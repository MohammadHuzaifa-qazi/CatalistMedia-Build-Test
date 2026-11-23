import sqlite3
import time

import pytest

from catalyst import config, llm, store


def make_llm(tmp_path) -> llm.LLM:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)
    limiter = llm.RateLimiter(path=tmp_path / "rate.json")
    return llm.LLM(
        conn=conn,
        limiter=limiter,
        metrics_path=tmp_path / "metrics.jsonl",
    )


def response(text: str = "ok", tool_calls=None) -> llm.ProviderResponse:
    return llm.ProviderResponse(
        text=text,
        tool_calls=tool_calls or [],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "thoughts_tokens": 3,
            "total_tokens": 15,
        },
        finish_reason="STOP",
    )


def test_cache_hit_skips_provider(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    calls = []

    def fake(model, *args, **kwargs):
        calls.append(model)
        return response("hello")

    monkeypatch.setattr(engine, "_call_gemini", fake)

    first = engine.chat([{"role": "user", "content": "hi"}])
    second = engine.chat([{"role": "user", "content": "hi"}])

    assert first.text == "hello" and not first.cached
    assert second.text == "hello" and second.cached
    assert len(calls) == 1


def test_fallback_uses_second_gemini_model(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    calls = []

    def fake(model, *args, **kwargs):
        calls.append(model)
        if model == config.GEMINI_MODEL:
            raise llm.ProviderError("503", status=503, retryable=False)
        return response("fallback")

    monkeypatch.setattr(engine, "_call_gemini", fake)

    result = engine.chat([{"role": "user", "content": "hi"}])

    assert result.model == config.GEMINI_FALLBACK_MODEL
    assert result.fallback_used
    assert result.text == "fallback"
    assert result.errors
    assert calls == [config.GEMINI_MODEL, config.GEMINI_FALLBACK_MODEL]


def test_tool_calls_are_returned(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)

    def fake(model, *args, **kwargs):
        return response(
            "",
            [{"id": "call_0", "name": "search", "arguments": {"query": "blue"}}],
        )

    monkeypatch.setattr(engine, "_call_gemini", fake)

    result = engine.chat([{"role": "user", "content": "find blue"}])
    assert result.tool_calls[0]["name"] == "search"
    assert result.tool_calls[0]["arguments"] == {"query": "blue"}


def test_prompt_budget_rejects_large_prompt(tmp_path):
    engine = make_llm(tmp_path)
    with pytest.raises(llm.PromptTooLarge):
        engine.chat([{"role": "user", "content": "x" * 30_000}])


def test_gemini_usage_bills_thought_tokens():
    usage = llm._gemini_usage(
        {
            "promptTokenCount": 7,
            "candidatesTokenCount": 1,
            "thoughtsTokenCount": 16,
            "totalTokenCount": 24,
        }
    )
    assert usage["thoughts_tokens"] == 16
    assert usage["billed_output_tokens"] == 17
    assert usage["total_tokens"] == 24


def test_groq_usage_maps_reasoning_tokens():
    class Usage:
        def model_dump(self):
            return {
                "prompt_tokens": 78,
                "completion_tokens": 18,
                "total_tokens": 96,
                "completion_tokens_details": {"reasoning_tokens": 8},
            }

    usage = llm._groq_usage(Usage())
    assert usage["thoughts_tokens"] == 8
    assert usage["billed_output_tokens"] == 18


def test_rate_limiter_daily_request_budget(tmp_path):
    limiter = llm.RateLimiter(
        path=tmp_path / "rate2.json",
        limits={
            "groq": {
                "rpm": 30,
                "tpm": 8000,
                "rpd": 1,
                "tpd": 200_000,
            }
        },
    )
    limiter.acquire("groq", 10)
    with pytest.raises(llm.DailyBudgetExceeded):
        limiter.acquire("groq", 10)


def test_rate_limiter_rejects_oversized_request(tmp_path):
    limiter = llm.RateLimiter(
        path=tmp_path / "rate3.json",
        limits={
            "groq": {
                "rpm": 30,
                "tpm": 100,
                "rpd": 1000,
                "tpd": 200_000,
            }
        },
    )
    with pytest.raises(llm.RequestTooLarge):
        limiter.acquire("groq", 101)


def test_parse_json_text_handles_fences():
    assert llm.parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json_text('prefix [1, 2] suffix') == [1, 2]
    assert llm.parse_json_text("not json") is None


def test_gemini_rate_limit_matches_free_tier():
    # Gemini free tier is 5 RPM. A higher local limit lets the client burst
    # past the provider limit and burn requests on guaranteed 429s.
    assert config.RATE_LIMITS["gemini"]["rpm"] == 5


def test_rate_limit_429_sets_short_cooldown_without_strike(tmp_path):
    limiter = llm.RateLimiter(path=tmp_path / "rate429.json")
    limiter.record_failure("model-x", status=429)

    circuit = limiter.state["circuits"]["model-x"]
    assert circuit["failures"] == 0
    assert circuit["disabled_until"] > time.time()
    assert circuit["disabled_until"] <= time.time() + 61
    with pytest.raises(llm.ProviderDisabled):
        limiter.check_circuit("model-x")


def test_other_client_errors_do_not_trip_breaker(tmp_path):
    limiter = llm.RateLimiter(path=tmp_path / "rate400.json")

    for _ in range(config.CIRCUIT_BREAKER_FAILURES + 2):
        limiter.record_failure("model-x", status=400)

    limiter.check_circuit("model-x")
    assert limiter.state["circuits"]["model-x"]["failures"] == 0


def test_server_errors_trip_breaker(tmp_path):
    limiter = llm.RateLimiter(path=tmp_path / "rate503.json")

    for _ in range(config.CIRCUIT_BREAKER_FAILURES):
        limiter.record_failure("model-x", status=503)

    circuit = limiter.state["circuits"]["model-x"]
    assert circuit["failures"] == config.CIRCUIT_BREAKER_FAILURES
    assert circuit["disabled_until"] > time.time()
    with pytest.raises(llm.ProviderDisabled):
        limiter.check_circuit("model-x")


def test_groq_json_failure_retries_once_at_low_effort(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    efforts = []
    seen_max_tokens = []

    def fake(
        model,
        messages,
        tools,
        response_schema,
        max_tokens,
        temperature,
        reasoning_effort,
        timeout,
    ):
        efforts.append(reasoning_effort)
        seen_max_tokens.append(max_tokens)
        if len(efforts) == 1:
            raise llm.ProviderError(
                "json_validate_failed",
                status=400,
                retryable=False,
                code="json_validate_failed",
            )
        return response('{"ok": true}')

    monkeypatch.setattr(engine, "_call_groq", fake)

    result = engine.chat(
        [{"role": "user", "content": "extract"}],
        response_schema={"type": "object"},
        max_tokens=1024,
        tier="final",
        model_override=config.GROQ_FINAL_MODEL,
    )

    assert efforts == [config.GROQ_FINAL_REASONING, "low"]
    assert seen_max_tokens[1] > seen_max_tokens[0]
    assert result.text == '{"ok": true}'
    assert result.errors
    assert result.errors[0]["code"] == "json_validate_failed"


def test_groq_json_failure_retries_bounded_then_drops_format(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    calls = []

    def fake(
        model,
        messages,
        tools,
        response_schema,
        max_tokens,
        temperature,
        reasoning_effort,
        timeout,
    ):
        calls.append((reasoning_effort, response_schema is not None))
        raise llm.ProviderError(
            "json_validate_failed",
            status=400,
            retryable=False,
            code="json_validate_failed",
        )

    monkeypatch.setattr(engine, "_call_groq", fake)

    with pytest.raises(llm.AllProvidersFailed):
        engine.chat(
            [{"role": "user", "content": "extract"}],
            response_schema={"type": "object"},
            tier="final",
            model_override=config.GROQ_FINAL_MODEL,
        )

    assert calls == [
        (config.GROQ_FINAL_REASONING, True),
        ("low", True),
        ("low", False),
    ]


def test_groq_json_failure_drops_response_format_and_parses_json(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    calls = []

    def fake(
        model,
        messages,
        tools,
        response_schema,
        max_tokens,
        temperature,
        reasoning_effort,
        timeout,
    ):
        calls.append((reasoning_effort, response_schema is not None))
        if response_schema is not None:
            raise llm.ProviderError(
                "json_validate_failed",
                status=400,
                retryable=False,
                code="json_validate_failed",
            )
        return response('```json\n{"ok": true}\n```')

    monkeypatch.setattr(engine, "_call_groq", fake)

    result = engine.chat(
        [{"role": "user", "content": "extract"}],
        response_schema={"type": "object"},
        tier="final",
        model_override=config.GROQ_FINAL_MODEL,
    )

    assert calls == [
        (config.GROQ_FINAL_REASONING, True),
        ("low", True),
        ("low", False),
    ]
    assert llm.parse_json_text(result.text) == {"ok": True}


def test_groq_json_drop_fallback_rejects_non_json(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)

    def fake(
        model,
        messages,
        tools,
        response_schema,
        max_tokens,
        temperature,
        reasoning_effort,
        timeout,
    ):
        if response_schema is not None:
            raise llm.ProviderError(
                "json_validate_failed",
                status=400,
                retryable=False,
                code="json_validate_failed",
            )
        return response("sorry, I cannot output json")

    monkeypatch.setattr(engine, "_call_groq", fake)

    with pytest.raises(llm.AllProvidersFailed):
        engine.chat(
            [{"role": "user", "content": "extract"}],
            response_schema={"type": "object"},
            tier="bulk",
            model_override=config.GROQ_BULK_MODEL,
        )
    assert engine.conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0


def test_extract_retry_after_parses_headers_and_gemini_body():
    assert llm._extract_retry_after(headers={"Retry-After": "17"}) == 17.0
    assert (
        llm._extract_retry_after(
            body_obj={
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "23s",
                        }
                    ]
                }
            }
        )
        == 23.0
    )
    assert (
        llm._extract_retry_after(body_text="Quota exceeded. Please retry in 14.5s.")
        == 14.5
    )
    assert (
        llm._extract_retry_after(headers={"x-ratelimit-reset-requests": "1m26.4s"})
        == 86.4
    )
    assert llm._extract_retry_after(headers={"Retry-After": "999"}) == 120.0


def test_rate_limit_429_honours_retry_after(tmp_path):
    limiter = llm.RateLimiter(path=tmp_path / "rate429_delay.json")
    before = time.time()
    limiter.record_failure("gemini-2.5-flash", status=429, retry_after=15.0)
    circuit = limiter.state["circuits"]["gemini-2.5-flash"]
    assert circuit["failures"] == 0
    assert before + 14.0 <= circuit["disabled_until"] <= before + 17.0


def test_groq_clamps_max_tokens_to_tpm(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    seen = []

    def fake(
        model,
        messages,
        tools,
        response_schema,
        max_tokens,
        temperature,
        reasoning_effort,
        timeout,
    ):
        seen.append(max_tokens)
        return response('{"ok": true}')

    monkeypatch.setattr(engine, "_call_groq", fake)

    # ~5000 prompt tokens + 4096 requested max_tokens > 8000 TPM -> clamped
    engine.chat(
        [{"role": "user", "content": "a" * 20_000}],
        max_tokens=4096,
        tier="final",
        model_override=config.GROQ_FINAL_MODEL,
    )
    assert len(seen) == 1
    assert 512 <= seen[0] < 4096


def test_to_groq_tools_allows_null_for_optional_params():
    sample = [
        {
            "type": "function",
            "function": {
                "name": "get_turns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "meeting_id": {"type": "string"},
                        "speaker": {"type": "string"},
                        "seq_start": {"type": "integer"},
                    },
                    "required": ["meeting_id"],
                },
            },
        }
    ]
    out = llm._to_groq_tools(sample)
    props = out[0]["function"]["parameters"]["properties"]
    assert props["meeting_id"]["type"] == "string"
    assert props["speaker"]["type"] == ["string", "null"]
    assert props["seq_start"]["type"] == ["integer", "null"]
    # Original schema is not mutated
    assert sample[0]["function"]["parameters"]["properties"]["speaker"]["type"] == "string"


def test_failed_call_refunds_reserved_tokens(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)

    def boom(*args, **kwargs):
        raise llm.ProviderError("connection failed", status=503, retryable=False)

    monkeypatch.setattr(engine, "_call_groq", boom)

    with pytest.raises(llm.AllProvidersFailed):
        engine.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=1000,
            tier="final",
            model_override=config.GROQ_FINAL_MODEL,
        )

    bucket = f"groq:{config.GROQ_FINAL_MODEL}"
    assert engine.limiter.state["providers"][bucket]["tokens_today"] == 0


def test_per_model_buckets_fallback_to_20b_when_120b_exhausted(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    # Disable Gemini models and exhaust 120b's daily budget
    engine.limiter.record_failure(config.GEMINI_MODEL, status=429, retry_after=3600)
    engine.limiter.record_failure(config.GEMINI_FALLBACK_MODEL, status=429, retry_after=3600)
    b120 = f"groq:{config.GROQ_FINAL_MODEL}"
    engine.limiter._provider(b120)["tokens_today"] = config.RATE_LIMITS["groq"]["tpd"]

    seen = []

    def fake_groq(model, *args, **kwargs):
        seen.append(model)
        return response('{"ok": true}')

    monkeypatch.setattr(engine, "_call_groq", fake_groq)
    res = engine.chat([{"role": "user", "content": "hi"}], tier="final")
    assert res.model == config.GROQ_BULK_MODEL
    assert res.fallback_used is True
    assert seen == [config.GROQ_BULK_MODEL]


def test_extract_retry_after_parses_groq_tpd_compound_duration():
    msg = (
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
        "Limit 200000, Used 199500. Please try again in 15m23.4s."
    )
    assert llm._extract_retry_after(body_text=msg, max_seconds=3600.0) == 923.4
    # Default clamp remains 120s for short RPM/TPM limits
    assert llm._extract_retry_after(body_text=msg) == 120.0


def test_429_fails_over_without_wasteful_retries_and_respects_cooldown(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    calls = []

    def fake_gemini(model, *args, **kwargs):
        calls.append(("gemini", model))
        raise llm.ProviderError("HTTP 429: quota", status=429, retryable=False, retry_after=45.0)

    def fake_groq(model, *args, **kwargs):
        calls.append(("groq", model))
        return response('{"ok": true}')

    monkeypatch.setattr(engine, "_call_gemini", fake_gemini)
    monkeypatch.setattr(engine, "_call_groq", fake_groq)

    # First call hits each Gemini model ONCE (no wasteful retries), records cooldown, falls over to Groq 120b
    r1 = engine.chat([{"role": "user", "content": "q1"}], tier="final")
    assert r1.model == config.GROQ_FINAL_MODEL
    assert calls == [
        ("gemini", config.GEMINI_MODEL),
        ("gemini", config.GEMINI_FALLBACK_MODEL),
        ("groq", config.GROQ_FINAL_MODEL),
    ]

    # Second uncached call SKIPS both cooled-down Gemini models immediately
    r2 = engine.chat([{"role": "user", "content": "q2"}], tier="final")
    assert r2.model == config.GROQ_FINAL_MODEL
    assert calls[3:] == [("groq", config.GROQ_FINAL_MODEL)]


def test_cached_work_preserved_when_all_providers_fail(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)

    monkeypatch.setattr(engine, "_call_gemini", lambda *a, **kw: response("cached-answer"))
    first = engine.chat([{"role": "user", "content": "remember me"}], tier="final")
    assert first.text == "cached-answer" and not first.cached

    # Now disable ALL providers
    for model in (
        config.GEMINI_MODEL,
        config.GEMINI_FALLBACK_MODEL,
        config.GROQ_FINAL_MODEL,
        config.GROQ_BULK_MODEL,
        config.GEMINI_TERTIARY_MODEL,
        config.GROQ_TERTIARY_MODEL,
        config.GEMINI_QUATERNARY_MODEL,
    ):
        engine.limiter.record_failure(model, status=429, retry_after=3600.0)

    # Cached prompt STILL succeeds without hitting any provider
    second = engine.chat([{"role": "user", "content": "remember me"}], tier="final")
    assert second.text == "cached-answer" and second.cached

    # Uncached prompt raises clear AllProvidersFailed without fabricating anything
    with pytest.raises(llm.AllProvidersFailed) as exc_info:
        engine.chat([{"role": "user", "content": "brand new uncached prompt"}], tier="final")
    msg = str(exc_info.value)
    assert "no fallback response was fabricated" in msg
    assert len(exc_info.value.errors) == 7


def test_rate_limiter_date_reset_clears_expired_circuits_keeps_active(tmp_path):
    path = tmp_path / "rate_reset.json"
    limiter = llm.RateLimiter(path=path)
    limiter.state["date"] = "2020-01-01"
    limiter.state["providers"] = {"groq:test": {"requests_today": 99, "tokens_today": 999, "requests": [], "token_events": []}}
    limiter.state["circuits"] = {
        "expired-model": {"failures": 3, "disabled_until": time.time() - 10},
        "active-model": {"failures": 0, "disabled_until": time.time() + 300},
    }
    limiter._save()
    limiter._reset()
    assert limiter.state["providers"] == {}
    assert "expired-model" not in limiter.state["circuits"]
    assert "active-model" in limiter.state["circuits"]


def test_gemini_uses_header_auth_and_redacts_secrets_on_network_error(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    fake_secret = "FAKE_SECRET_KEY_1234567890"
    monkeypatch.setattr(config, "GEMINI_API_KEY", fake_secret)
    seen_calls = []

    def fake_post(url, *, headers=None, params=None, json=None, timeout=None):
        seen_calls.append({"url": url, "headers": headers, "params": params})
        raise llm.requests.ConnectionError(
            f"Connection failed to {url}?key={fake_secret}"
        )

    monkeypatch.setattr(llm.requests, "post", fake_post)
    with pytest.raises(llm.AllProvidersFailed) as exc_info:
        engine.chat(
            [{"role": "user", "content": "hi"}],
            tier="final",
            model_override=config.GEMINI_FALLBACK_MODEL,
        )
    assert seen_calls
    assert seen_calls[0]["params"] is None
    assert seen_calls[0]["headers"] == {"x-goog-api-key": fake_secret}
    assert fake_secret not in str(exc_info.value)
    metrics_text = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
    assert fake_secret not in metrics_text


def test_groq_tertiary_caps_max_tokens_under_otpm_limit(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)
    seen_max: list[int] = []

    def fake_groq(model, messages, tools, schema, max_tokens, *args, **kwargs):
        seen_max.append(max_tokens)
        return response('{"ok": true}')

    monkeypatch.setattr(engine, "_call_groq", fake_groq)
    res = engine.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=2048,
        tier="final",
        model_override=config.GROQ_TERTIARY_MODEL,
    )
    assert res.model == config.GROQ_TERTIARY_MODEL
    assert seen_max == [750]


def test_5xx_error_refunds_daily_request_slot(tmp_path, monkeypatch):
    engine = make_llm(tmp_path)

    def fail_503(*args, **kwargs):
        raise llm.ProviderError("HTTP 503: high demand", status=503, retryable=False)

    monkeypatch.setattr(engine, "_call_gemini", fail_503)
    with pytest.raises(llm.AllProvidersFailed):
        engine.chat(
            [{"role": "user", "content": "hi"}],
            tier="final",
            model_override=config.GEMINI_FALLBACK_MODEL,
        )
    bucket = f"gemini:{config.GEMINI_FALLBACK_MODEL}"
    assert engine.limiter._provider(bucket)["requests_today"] == 0
