from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import config, store


class LLMError(Exception):
    pass


class PromptTooLarge(LLMError):
    pass


class AllProvidersFailed(LLMError):
    def __init__(self, message: str, *, errors: list[dict] | None = None) -> None:
        super().__init__(message)
        self.errors = list(errors or [])


class ProviderDisabled(LLMError):
    pass


class DailyBudgetExceeded(LLMError):
    pass


class RequestTooLarge(LLMError):
    pass


class ProviderError(LLMError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.code = code
        self.retry_after = retry_after


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[dict]
    usage: dict
    finish_reason: str | None = None


@dataclass
class LLMResult:
    text: str
    tool_calls: list[dict]
    usage: dict
    provider: str
    model: str
    fallback_used: bool = False
    cached: bool = False
    finish_reason: str | None = None
    latency_sec: float = 0.0
    errors: list[dict] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_secrets(text: str) -> str:
    """Strip configured API keys and any ?key=... query tokens from logs/errors."""
    out = str(text or "")
    for secret in (config.GEMINI_API_KEY, config.GROQ_API_KEY):
        if secret:
            out = out.replace(secret, "<redacted>")
    out = re.sub(r"([?&]key=)[^&\s'\"\\)]+", r"\1<redacted>", out, flags=re.IGNORECASE)
    return out


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _canonical_prompt(
    messages: list[dict],
    tools: list[dict] | None,
    response_schema: dict | None,
    max_tokens: int,
    temperature: float,
    tier: str,
) -> str:
    return json.dumps(
        {
            "messages": messages,
            "tools": tools,
            "response_schema": response_schema,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tier": tier,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _cache_key(provider: str, model: str, canonical: str) -> str:
    return hashlib.sha256(
        f"{provider}|{model}|{canonical}".encode("utf-8")
    ).hexdigest()


def _gemini_usage(raw: dict) -> dict:
    prompt = raw.get("promptTokenCount", 0) or 0
    completion = raw.get("candidatesTokenCount", 0) or 0
    thoughts = raw.get("thoughtsTokenCount", 0) or 0
    total = raw.get("totalTokenCount", prompt + completion + thoughts) or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "thoughts_tokens": thoughts,
        "total_tokens": total,
        "billed_output_tokens": completion + thoughts,
        "raw": raw,
    }


def _groq_usage(usage: Any) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = dict(usage)
    else:
        data = {}
    details = data.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
    completion = data.get("completion_tokens", 0) or 0
    return {
        "prompt_tokens": data.get("prompt_tokens", 0) or 0,
        "completion_tokens": completion,
        "thoughts_tokens": reasoning or 0,
        "total_tokens": data.get("total_tokens", 0) or 0,
        "billed_output_tokens": completion,
        "raw": data,
    }


def _to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            system_parts.append(str(content))
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            {"role": gemini_role, "parts": [{"text": str(content)}]}
        )
    merged: list[dict] = []
    for item in contents:
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1]["parts"].extend(item["parts"])
        else:
            merged.append(item)
    if not merged:
        merged = [{"role": "user", "parts": [{"text": ""}]}]
    return "\n\n".join(system_parts), merged


def _to_gemini_tools(tools: list[dict]) -> list[dict]:
    declarations = []
    for tool in tools:
        fn = tool.get("function", tool)
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return declarations


def _to_groq_tools(tools: list[dict]) -> list[dict]:
    """Allow explicit null for optional parameters so Groq tool validation accepts them."""
    converted: list[dict] = []
    for tool in tools:
        entry = json.loads(json.dumps(tool))
        fn = entry.get("function") if isinstance(entry, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(params, dict):
            required = set(params.get("required") or [])
            props = params.get("properties")
            if isinstance(props, dict):
                for prop_name, schema in props.items():
                    if prop_name in required or not isinstance(schema, dict):
                        continue
                    ptype = schema.get("type")
                    if isinstance(ptype, str):
                        schema["type"] = [ptype, "null"]
                    elif isinstance(ptype, list) and "null" not in ptype:
                        schema["type"] = [*ptype, "null"]
                    if "enum" in schema and isinstance(schema["enum"], list):
                        if None not in schema["enum"]:
                            schema["enum"] = [*schema["enum"], None]
        converted.append(entry)
    return converted


def parse_json_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                continue
    return None


def _parse_duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        num = float(text)
        return num if num > 0 else None
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)(?![a-z])", text):
        matched = True
        val = float(amount)
        if unit == "ms":
            total += val / 1000.0
        elif unit == "s":
            total += val
        elif unit == "m":
            total += val * 60.0
        elif unit == "h":
            total += val * 3600.0
    return total if matched and total > 0 else None


def _extract_retry_after(
    headers: Any = None,
    body_text: str = "",
    body_obj: dict | None = None,
    *,
    max_seconds: float = 120.0,
) -> float | None:
    delay: float | None = None
    if headers:
        lower = {str(k).lower(): v for k, v in dict(headers).items()}
        for key in (
            "retry-after",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-reset-requests",
        ):
            if key in lower:
                delay = _parse_duration_seconds(lower[key])
                if delay is not None:
                    break
    if delay is None and isinstance(body_obj, dict):
        err = body_obj.get("error")
        details = err.get("details") if isinstance(err, dict) else None
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict) and "retryDelay" in item:
                    delay = _parse_duration_seconds(item["retryDelay"])
                    if delay is not None:
                        break
    if delay is None and body_text:
        m = re.search(
            r"(?:retry|try\s+again)\s+(?:in|after)\s+((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)(?![a-z])\s*)+)",
            body_text,
            re.IGNORECASE,
        )
        if m:
            delay = _parse_duration_seconds(m.group(1))
    if delay is None:
        return None
    return round(max(1.0, min(delay, max_seconds)), 2)


class RateLimiter:
    def __init__(
        self,
        path: Path | None = None,
        limits: dict | None = None,
    ) -> None:
        self.path = path or config.RATE_STATE_PATH
        self.limits = limits or config.RATE_LIMITS
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    state.setdefault("providers", {})
                    state.setdefault("circuits", {})
                    return state
            except Exception:
                pass
        return {"date": _utc_date(), "providers": {}, "circuits": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _reset(self) -> None:
        today = _utc_date()
        if self.state.get("date") != today:
            now = time.time()
            active_circuits = {
                model: c
                for model, c in (self.state.get("circuits") or {}).items()
                if isinstance(c, dict) and c.get("disabled_until", 0.0) > now
            }
            self.state = {
                "date": today,
                "providers": {},
                "circuits": active_circuits,
            }
            self._save()

    def _provider(self, provider: str) -> dict:
        self.state.setdefault("providers", {})
        return self.state["providers"].setdefault(
            provider,
            {
                "requests_today": 0,
                "tokens_today": 0,
                "requests": [],
                "token_events": [],
            },
        )

    def _circuit(self, model: str) -> dict:
        self.state.setdefault("circuits", {})
        return self.state["circuits"].setdefault(
            model, {"failures": 0, "disabled_until": 0.0}
        )

    def check_circuit(self, model: str) -> None:
        self._reset()
        circuit = self._circuit(model)
        if circuit.get("disabled_until", 0.0) > time.time():
            raise ProviderDisabled(
                f"{model} disabled until {circuit['disabled_until']}"
            )

    def acquire(
        self, provider: str, estimated_tokens: int, *, bucket: str | None = None
    ) -> None:
        self._reset()
        limits = dict(self.limits[provider])
        if bucket and ":" in bucket:
            model_name = bucket.split(":", 1)[1]
            override = getattr(config, "MODEL_RATE_LIMITS", {}).get(model_name)
            if override:
                limits.update(override)
        state = self._provider(bucket or provider)
        if estimated_tokens > limits["tpm"]:
            raise RequestTooLarge(
                f"{provider}: {estimated_tokens} > TPM {limits['tpm']}"
            )
        if state["requests_today"] + 1 > limits["rpd"]:
            raise DailyBudgetExceeded(f"{bucket or provider}: daily request limit")
        if state["tokens_today"] + estimated_tokens > limits["tpd"]:
            raise DailyBudgetExceeded(f"{bucket or provider}: daily token limit")

        now = time.time()
        cutoff = now - 60
        state["requests"] = [ts for ts in state["requests"] if ts > cutoff]
        state["token_events"] = [
            [ts, n] for ts, n in state["token_events"] if ts > cutoff
        ]

        while len(state["requests"]) + 1 > limits["rpm"]:
            wait = state["requests"][0] + 60 - now
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            cutoff = now - 60
            state["requests"] = [ts for ts in state["requests"] if ts > cutoff]

        while (
            sum(n for _, n in state["token_events"]) + estimated_tokens
            > limits["tpm"]
        ):
            if not state["token_events"]:
                break
            wait = state["token_events"][0][0] + 60 - now
            if wait > 0:
                time.sleep(wait)
            now = time.time()
            cutoff = now - 60
            state["token_events"] = [
                [ts, n] for ts, n in state["token_events"] if ts > cutoff
            ]

        state["requests"].append(now)
        state["requests_today"] += 1
        state["token_events"].append([now, estimated_tokens])
        state["tokens_today"] += estimated_tokens
        self._save()

    def record_tokens(
        self,
        provider: str,
        actual_tokens: int,
        estimated_tokens: int,
        *,
        bucket: str | None = None,
    ) -> None:
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return
        self._reset()
        state = self._provider(bucket or provider)
        state["tokens_today"] = max(0, state["tokens_today"] + diff)
        if state["token_events"]:
            state["token_events"][-1][1] = max(
                0, state["token_events"][-1][1] + diff
            )
        self._save()

    def refund_request(
        self,
        provider: str,
        *,
        bucket: str | None = None,
    ) -> None:
        """Refund a reserved request slot when a server 5xx/timeout prevented execution."""
        self._reset()
        state = self._provider(bucket or provider)
        state["requests_today"] = max(0, state.get("requests_today", 0) - 1)
        if state.get("requests"):
            state["requests"].pop()
        self._save()

    def record_failure(
        self,
        model: str,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self._reset()
        circuit = self._circuit(model)
        if status == 429:
            # Rate limits say "slow down", not "this model is broken". Honour
            # the server retry delay (or a short default cooldown) without
            # counting a health strike or opening a 10-minute breaker.
            cooldown = (
                retry_after
                if retry_after is not None and retry_after > 0
                else config.CIRCUIT_BREAKER_RATE_LIMIT_SECONDS
            )
            circuit["disabled_until"] = max(
                circuit.get("disabled_until", 0.0),
                time.time() + cooldown,
            )
            self._save()
            return
        if status is not None and 400 <= status < 500:
            # Other client errors (bad schema, oversized request, auth) are
            # request problems; retrying the same request will not fix them.
            return
        circuit["failures"] = circuit.get("failures", 0) + 1
        if circuit["failures"] >= config.CIRCUIT_BREAKER_FAILURES:
            circuit["disabled_until"] = (
                time.time() + config.CIRCUIT_BREAKER_SECONDS
            )
        self._save()

    def record_success(self, model: str) -> None:
        self._reset()
        circuit = self._circuit(model)
        circuit["failures"] = 0
        circuit["disabled_until"] = 0.0
        self._save()


class LLM:
    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        *,
        cache: bool = True,
        limiter: RateLimiter | None = None,
        metrics_path: Path | None = None,
    ) -> None:
        self.conn = conn or store.connect()
        self.cache = cache
        self.limiter = limiter or RateLimiter()
        self.metrics_path = metrics_path or config.LLM_METRICS_PATH
        self._ensure_cache_table()

    def _ensure_cache_table(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
              cache_key     TEXT PRIMARY KEY,
              provider      TEXT NOT NULL,
              model         TEXT NOT NULL,
              prompt_hash   TEXT NOT NULL,
              response_json TEXT NOT NULL,
              usage_json    TEXT,
              created_at    TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _candidates(
        self, tier: str, model_override: str | None
    ) -> list[tuple[str, str, dict]]:
        if model_override:
            if model_override == config.GEMINI_MODEL:
                return [
                    (
                        "gemini",
                        model_override,
                        {
                            "thinking": {
                                "thinkingLevel": config.GEMINI_38_THINKING_BULK
                            },
                            "attempts": 1,
                        },
                    )
                ]
            if model_override == config.GEMINI_FALLBACK_MODEL:
                return [
                    (
                        "gemini",
                        model_override,
                        {
                            "thinking": {
                                "thinkingBudget": config.GEMINI_25_BUDGET_BULK
                            },
                            "attempts": 2,
                        },
                    )
                ]
            if model_override == config.GROQ_BULK_MODEL:
                return [
                    (
                        "groq",
                        model_override,
                        {
                            "reasoning_effort": config.GROQ_BULK_REASONING,
                            "attempts": 3,
                        },
                    )
                ]
            if model_override == config.GROQ_FINAL_MODEL:
                return [
                    (
                        "groq",
                        model_override,
                        {
                            "reasoning_effort": config.GROQ_FINAL_REASONING,
                            "attempts": 3,
                        },
                    )
                ]
            if model_override in {
                config.GEMINI_TERTIARY_MODEL,
                config.GEMINI_QUATERNARY_MODEL,
            }:
                return [("gemini", model_override, {"attempts": 2})]
            if model_override == config.GROQ_TERTIARY_MODEL:
                return [("groq", model_override, {"attempts": 2})]
            raise ValueError(f"unknown model override: {model_override}")

        if tier == "bulk":
            return [
                (
                    "gemini",
                    config.GEMINI_MODEL,
                    {
                        "thinking": {
                            "thinkingLevel": config.GEMINI_38_THINKING_BULK
                        },
                        "attempts": 1,
                    },
                ),
                (
                    "gemini",
                    config.GEMINI_FALLBACK_MODEL,
                    {
                        "thinking": {
                            "thinkingBudget": config.GEMINI_25_BUDGET_BULK
                        },
                        "attempts": 2,
                    },
                ),
                (
                    "groq",
                    config.GROQ_BULK_MODEL,
                    {
                        "reasoning_effort": config.GROQ_BULK_REASONING,
                        "attempts": 3,
                    },
                ),
            ]

        return [
            (
                "gemini",
                config.GEMINI_MODEL,
                {
                    "thinking": {
                        "thinkingLevel": config.GEMINI_38_THINKING_FINAL
                    },
                    "attempts": 1,
                },
            ),
            (
                "gemini",
                config.GEMINI_FALLBACK_MODEL,
                {
                    "thinking": {
                        "thinkingBudget": config.GEMINI_25_BUDGET_FINAL
                    },
                    "attempts": 2,
                },
            ),
            (
                "groq",
                config.GROQ_FINAL_MODEL,
                {
                    "reasoning_effort": config.GROQ_FINAL_REASONING,
                    "attempts": 3,
                },
            ),
            (
                "groq",
                config.GROQ_BULK_MODEL,
                {
                    "reasoning_effort": config.GROQ_BULK_REASONING,
                    "attempts": 3,
                },
            ),
            (
                "gemini",
                config.GEMINI_TERTIARY_MODEL,
                {
                    "attempts": 2,
                },
            ),
            (
                "groq",
                config.GROQ_TERTIARY_MODEL,
                {
                    "attempts": 2,
                },
            ),
            (
                "gemini",
                config.GEMINI_QUATERNARY_MODEL,
                {
                    "attempts": 2,
                },
            ),
        ]

    def _cache_get(
        self, provider: str, model: str, canonical: str
    ) -> LLMResult | None:
        key = _cache_key(provider, model, canonical)
        row = self.conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["response_json"])
        except Exception:
            return None
        return LLMResult(
            text=data.get("text", ""),
            tool_calls=data.get("tool_calls", []),
            usage=data.get("usage", {}),
            provider=provider,
            model=model,
            cached=True,
            finish_reason=data.get("finish_reason"),
        )

    def _cache_put(
        self,
        provider: str,
        model: str,
        canonical: str,
        prompt_hash: str,
        result: LLMResult,
    ) -> None:
        key = _cache_key(provider, model, canonical)
        payload = {
            "text": result.text,
            "tool_calls": result.tool_calls,
            "usage": result.usage,
            "finish_reason": result.finish_reason,
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(cache_key, provider, model, prompt_hash, response_json, usage_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                provider,
                model,
                prompt_hash,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result.usage, ensure_ascii=False),
                _now_iso(),
            ),
        )
        self.conn.commit()

    def _log_metric(
        self,
        result: LLMResult,
        *,
        tier: str,
        status: str,
        billed: bool,
    ) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _now_iso(),
            "provider": result.provider,
            "model": result.model,
            "tier": tier,
            "cached": result.cached,
            "billed": billed,
            "fallback_used": result.fallback_used,
            "prompt_tokens": result.usage.get("prompt_tokens", 0),
            "completion_tokens": result.usage.get("completion_tokens", 0),
            "thoughts_tokens": result.usage.get("thoughts_tokens", 0),
            "total_tokens": result.usage.get("total_tokens", 0),
            "latency_sec": result.latency_sec,
            "status": status,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_error(
        self,
        provider: str,
        model: str,
        tier: str,
        error: str,
        latency: float,
    ) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": _now_iso(),
            "provider": provider,
            "model": model,
            "tier": tier,
            "cached": False,
            "billed": False,
            "fallback_used": False,
            "latency_sec": round(latency, 3),
            "status": "error",
            "error": _redact_secrets(error),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        tier: str = "bulk",
        model_override: str | None = None,
        timeout: float | None = None,
        max_prompt_tokens: int | None = None,
    ) -> LLMResult:
        if tier not in {"bulk", "final", "judge"}:
            raise ValueError(f"bad tier: {tier}")
        max_tokens = max_tokens or (
            config.LLM_MAX_TOKENS_FINAL
            if tier != "bulk"
            else config.LLM_MAX_TOKENS_BULK
        )
        timeout = timeout or (
            config.LLM_TIMEOUT_FINAL
            if tier != "bulk"
            else config.LLM_TIMEOUT_BULK
        )
        max_prompt_tokens = max_prompt_tokens or config.PROMPT_BUDGET_TOKENS
        canonical = _canonical_prompt(
            messages, tools, response_schema, max_tokens, temperature, tier
        )
        prompt_tokens = max(1, len(canonical) // 4)
        if prompt_tokens > max_prompt_tokens:
            raise PromptTooLarge(
                f"prompt ~{prompt_tokens} tokens > {max_prompt_tokens}"
            )
        prompt_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        candidates = self._candidates(tier, model_override)

        if self.cache:
            for index, (provider, model, _) in enumerate(candidates):
                cached = self._cache_get(provider, model, canonical)
                if cached is not None:
                    cached.fallback_used = index > 0
                    self._log_metric(
                        cached, tier=tier, status="ok", billed=False
                    )
                    return cached

        errors: list[dict] = []
        for index, (provider, model, options) in enumerate(candidates):
            try:
                self.limiter.check_circuit(model)
            except ProviderDisabled as exc:
                errors.append(
                    {
                        "provider": provider,
                        "model": model,
                        "skipped": True,
                        "error": str(exc),
                    }
                )
                continue
            call_max_tokens = max_tokens
            if provider == "groq":
                if model == config.GROQ_TERTIARY_MODEL:
                    # qwen/qwen3.8-27b enforces a 1,000 output-tokens-per-minute
                    # (OTPM) cap on Groq's free tier; cap max_tokens cleanly under it.
                    call_max_tokens = min(call_max_tokens, 750)
                tpm = self.limiter.limits["groq"]["tpm"]
                if prompt_tokens + call_max_tokens > tpm and prompt_tokens + 512 <= tpm:
                    call_max_tokens = max(512, tpm - prompt_tokens - 128)
            estimated = prompt_tokens + call_max_tokens
            if provider == "groq":
                # Reserve a small BPE/JSON-schema cushion inside Groq's 60s TPM
                # window without falsely triggering RequestTooLarge.
                tpm = self.limiter.limits["groq"]["tpm"]
                if estimated <= tpm:
                    estimated = min(tpm, int(estimated * 1.15))
            attempts = options.get("attempts", 2)
            reasoning_effort = options.get("reasoning_effort")
            use_json_mode = response_schema is not None
            json_effort_retry_done = False
            json_drop_retry_done = False
            bucket = f"{provider}:{model}"
            attempt = 0
            while attempt < attempts:
                attempt += 1
                try:
                    self.limiter.acquire(provider, estimated, bucket=bucket)
                except (
                    DailyBudgetExceeded,
                    RequestTooLarge,
                    ProviderDisabled,
                ) as exc:
                    errors.append(
                        {
                            "provider": provider,
                            "model": model,
                            "skipped": True,
                            "error": str(exc),
                        }
                    )
                    break
                started = time.time()
                try:
                    if provider == "gemini":
                        response = self._call_gemini(
                            model,
                            messages,
                            tools,
                            response_schema,
                            call_max_tokens,
                            temperature,
                            options.get("thinking"),
                            timeout,
                        )
                    else:
                        groq_schema = response_schema if use_json_mode else None
                        response = self._call_groq(
                            model,
                            messages,
                            tools,
                            groq_schema,
                            call_max_tokens,
                            temperature,
                            reasoning_effort,
                            timeout,
                        )
                        if (
                            response_schema is not None
                            and not use_json_mode
                            and parse_json_text(response.text) is None
                        ):
                            raise ProviderError(
                                "groq non-json fallback returned non-JSON output",
                                status=422,
                                retryable=False,
                                code="invalid_json_output",
                            )
                    latency = time.time() - started
                    actual_tokens = response.usage.get("total_tokens", estimated)
                    self.limiter.record_tokens(
                        provider, actual_tokens, estimated, bucket=bucket
                    )
                    self.limiter.record_success(model)
                    result = LLMResult(
                        text=response.text,
                        tool_calls=response.tool_calls,
                        usage=response.usage,
                        provider=provider,
                        model=model,
                        fallback_used=index > 0,
                        cached=False,
                        finish_reason=response.finish_reason,
                        latency_sec=round(latency, 3),
                        errors=errors,
                    )
                    if self.cache:
                        self._cache_put(
                            provider, model, canonical, prompt_hash, result
                        )
                    self._log_metric(
                        result, tier=tier, status="ok", billed=True
                    )
                    return result
                except ProviderError as exc:
                    latency = time.time() - started
                    # Refund reserved estimated tokens so failed calls do not
                    # burn phantom daily budget; also refund the daily request
                    # slot on 5xx/network/timeout errors that never executed.
                    self.limiter.record_tokens(provider, 0, estimated, bucket=bucket)
                    if exc.status is None or exc.status >= 500:
                        self.limiter.refund_request(provider, bucket=bucket)
                    self.limiter.record_failure(
                        model, status=exc.status, retry_after=exc.retry_after
                    )
                    record = {
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                        "status": exc.status,
                        "error": str(exc),
                        "latency_sec": round(latency, 3),
                    }
                    if exc.code:
                        record["code"] = exc.code
                    if exc.retry_after is not None:
                        record["retry_after"] = exc.retry_after
                    errors.append(record)
                    self._log_error(
                        provider, model, tier, str(exc), latency
                    )
                    if (
                        provider == "groq"
                        and response_schema is not None
                        and exc.code in {"json_validate_failed", "tool_use_failed"}
                    ):
                        # Verified empirically: Groq counts reasoning tokens
                        # against max_tokens, leaving empty content when budget
                        # runs out. First lower reasoning_effort to "low" and
                        # expand max_tokens within TPM headroom; if json_object
                        # mode still fails, drop response_format and parse JSON
                        # ourselves via parse_json_text().
                        tpm = self.limiter.limits["groq"]["tpm"]
                        headroom = max(512, tpm - prompt_tokens - 128)
                        bumped = min(max(call_max_tokens, 2048), headroom)
                        if bumped > call_max_tokens:
                            call_max_tokens = bumped
                            estimated = prompt_tokens + call_max_tokens
                        if reasoning_effort != "low" and not json_effort_retry_done:
                            json_effort_retry_done = True
                            reasoning_effort = "low"
                            attempts += 1
                            continue
                        if use_json_mode and not json_drop_retry_done:
                            json_drop_retry_done = True
                            use_json_mode = False
                            attempts += 1
                            continue
                    if exc.retryable and attempt < attempts:
                        time.sleep(2 ** attempt)
                        continue
                    break
                except Exception as exc:
                    latency = time.time() - started
                    self.limiter.record_tokens(provider, 0, estimated, bucket=bucket)
                    self.limiter.record_failure(model)
                    message = f"{type(exc).__name__}: {exc}"
                    errors.append(
                        {
                            "provider": provider,
                            "model": model,
                            "attempt": attempt,
                            "error": message,
                            "latency_sec": round(latency, 3),
                        }
                    )
                    self._log_error(provider, model, tier, message, latency)
                    break

        summary_lines = [
            f"All {len(candidates)} LLM provider candidates failed (tier={tier}); "
            "no fallback response was fabricated. Check API keys, quotas, or cooldowns."
        ]
        for err in errors:
            model = err.get("model", "?")
            status = err.get("status")
            tag = f"HTTP {status}" if status else ("skipped" if err.get("skipped") else "error")
            retry_note = f", retry_after={err['retry_after']}s" if err.get("retry_after") else ""
            summary_lines.append(f"  - {err.get('provider')}:{model} ({tag}{retry_note}): {str(err.get('error', ''))[:180]}")
        raise AllProvidersFailed(
            "\n".join(summary_lines)
            + "\nDetails:\n"
            + json.dumps(errors, indent=2, ensure_ascii=False),
            errors=errors,
        )

    def _call_gemini(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_schema: dict | None,
        max_tokens: int,
        temperature: float,
        thinking: dict | None,
        timeout: float,
    ) -> ProviderResponse:
        system, contents = _to_gemini_contents(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if thinking:
            body["generationConfig"]["thinkingConfig"] = thinking
        if tools:
            body["tools"] = [
                {"functionDeclarations": _to_gemini_tools(tools)}
            ]
            body["toolConfig"] = {
                "functionCallingConfig": {"mode": "AUTO"}
            }
        if response_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {"x-goog-api-key": config.GEMINI_API_KEY}
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise ProviderError(_redact_secrets(f"timeout: {exc}"), retryable=True) from exc
        except requests.RequestException as exc:
            raise ProviderError(_redact_secrets(f"request error: {exc}"), retryable=True) from exc

        if response.status_code == 400 and thinking:
            body["generationConfig"].pop("thinkingConfig", None)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                raise ProviderError(
                    _redact_secrets(f"request error: {exc}"), retryable=True
                ) from exc

        if response.status_code != 200:
            status = response.status_code
            # 429 is handled by failing over to the next candidate; retrying
            # in-provider just burns the local rate budget.
            retryable = status in (500, 502, 503, 504)
            retry_after = None
            if status == 429:
                try:
                    parsed_err = response.json()
                except Exception:
                    parsed_err = None
                if "per day" in response.text.lower() or "generate_content_free_tier_requests" in response.text:
                    # Daily quota exhausted; cool down for 1 hour rather than
                    # hammering a dead daily limit every minute.
                    retry_after = 3600.0
                else:
                    retry_after = _extract_retry_after(
                        response.headers, response.text, parsed_err
                    )
            detail = _redact_secrets(response.text[:500])
            raise ProviderError(
                f"HTTP {status}: {detail}",
                status=status,
                retryable=retryable,
                retry_after=retry_after,
            )
        return self._parse_gemini(response.json())

    def _parse_gemini(self, data: dict) -> ProviderResponse:
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(
            part.get("text", "") for part in parts if "text" in part
        )
        tool_calls = []
        for index, part in enumerate(parts):
            call = part.get("functionCall")
            if call:
                tool_calls.append(
                    {
                        "id": f"call_{index}",
                        "name": call.get("name"),
                        "arguments": call.get("args", {}),
                    }
                )
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            usage=_gemini_usage(data.get("usageMetadata", {})),
            finish_reason=candidate.get("finishReason"),
        )

    def _call_groq(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        response_schema: dict | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        timeout: float,
    ) -> ProviderResponse:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY, timeout=timeout)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _to_groq_tools(tools)
            kwargs["tool_choice"] = "auto"
            # Tool-calling steps don't need heavy reasoning; low effort avoids
            # burning the completion budget and keeps TPD usage lean.
            if reasoning_effort == "medium":
                reasoning_effort = "low"
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
            completion = raw.parse()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            name = type(exc).__name__
            code = None
            body = getattr(exc, "body", None)
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except Exception:
                    body = None
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    code = error.get("code")
            retry_after = None
            if status == 429:
                resp_obj = getattr(exc, "response", None)
                resp_headers = getattr(resp_obj, "headers", None)
                exc_lower = str(exc).lower()
                is_daily = any(
                    tok in exc_lower
                    for tok in ("tokens per day", "(tpd)", "requests per day", "(rpd)", "per day")
                )
                retry_after = _extract_retry_after(
                    resp_headers,
                    str(exc),
                    body if isinstance(body, dict) else None,
                    max_seconds=3600.0 if is_daily else 120.0,
                )
                if is_daily and (retry_after is None or retry_after < 300.0):
                    retry_after = max(retry_after or 0.0, 1800.0)
            retryable = status in (500, 502, 503, 504) or name in {
                "APITimeoutError",
                "APIConnectionError",
            }
            message = f"{name}: {exc}"
            if config.GROQ_API_KEY:
                message = message.replace(config.GROQ_API_KEY, "<redacted>")
            raise ProviderError(
                message,
                status=status,
                retryable=retryable,
                code=code,
                retry_after=retry_after,
            ) from exc

        choice = completion.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls = []
        for call in message.tool_calls or []:
            raw_args = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except Exception:
                arguments = {"_raw": raw_args}
            tool_calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": arguments,
                }
            )
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            usage=_groq_usage(completion.usage),
            finish_reason=choice.finish_reason,
        )
