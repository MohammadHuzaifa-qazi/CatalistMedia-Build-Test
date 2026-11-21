import sqlite3

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
