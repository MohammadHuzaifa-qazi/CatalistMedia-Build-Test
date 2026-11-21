from __future__ import annotations

import json
import time

import requests

from . import config

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_FALLBACK = "gemini-2.5-flash"


def _safe(text: str) -> str:
    for key in (config.GEMINI_API_KEY, config.GROQ_API_KEY):
        if key:
            text = text.replace(key, "<redacted>")
    return text


def _gemini_post(
    model: str,
    generation_config: dict,
    timeout: int = 25,
    attempts: int = 2,
) -> dict:
    url = f"{GEMINI_BASE}/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
        "generationConfig": generation_config,
    }
    last: dict = {}
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            response = requests.post(
                url,
                params={"key": config.GEMINI_API_KEY},
                json=body,
                timeout=timeout,
            )
            raw = response.text
            try:
                parsed = response.json()
            except Exception:
                parsed = {}
            text = ""
            candidates = parsed.get("candidates") or []
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
            result = {
                "model": model,
                "attempt": attempt,
                "status": response.status_code,
                "ok": response.status_code == 200,
                "text": text[:200],
                "usage": parsed.get("usageMetadata", {}),
                "model_version": parsed.get("modelVersion"),
                "elapsed_sec": round(time.time() - started, 2),
            }
            if response.status_code != 200:
                result["error"] = _safe(raw)[:500]
            last = result
            if response.status_code == 200:
                return result
            if response.status_code not in (429, 500, 502, 503, 504):
                return result
        except Exception as exc:
            last = {
                "model": model,
                "attempt": attempt,
                "status": "error",
                "ok": False,
                "error": _safe(f"{type(exc).__name__}: {exc}")[:500],
                "elapsed_sec": round(time.time() - started, 2),
            }
        if attempt < attempts:
            time.sleep(5 * attempt)
    return last


def gemini_probe() -> dict:
    target = _gemini_post(config.GEMINI_MODEL, {"maxOutputTokens": 128}, attempts=2)

    thinking_results: list[dict] = []
    if target.get("ok"):
        for level in ("MINIMAL", "LOW", "MEDIUM", "HIGH", "minimal"):
            thinking_results.append(
                _gemini_post(
                    config.GEMINI_MODEL,
                    {
                        "maxOutputTokens": 128,
                        "thinkingConfig": {"thinkingLevel": level},
                    },
                    attempts=1,
                )
            )
    else:
        thinking_results.append(
            {
                "status": "skipped",
                "reason": "target model unavailable; thinking levels not tested",
            }
        )

    fallback = _gemini_post(GEMINI_FALLBACK, {"maxOutputTokens": 64}, attempts=1)
    fallback_budget = _gemini_post(
        GEMINI_FALLBACK,
        {"maxOutputTokens": 64, "thinkingConfig": {"thinkingBudget": 0}},
        attempts=1,
    )
    fallback_level = _gemini_post(
        GEMINI_FALLBACK,
        {"maxOutputTokens": 64, "thinkingConfig": {"thinkingLevel": "MINIMAL"}},
        attempts=1,
    )

    return {
        "provider": "gemini",
        "target_model": config.GEMINI_MODEL,
        "target": target,
        "thinking_levels": thinking_results,
        "fallback_model": GEMINI_FALLBACK,
        "fallback": fallback,
        "fallback_thinking_budget_zero": fallback_budget,
        "fallback_thinking_level_minimal": fallback_level,
        "documented_limits": {
            "status": "NOT VERIFIED",
            "note": "Free-tier RPM/RPD/TPM are only shown in AI Studio.",
        },
        "rate_limit_429": "not triggered in probe",
    }


def groq_probe() -> dict:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY, timeout=30.0)
    results = []
    for model in (config.GROQ_BULK_MODEL, config.GROQ_FINAL_MODEL):
        started = time.time()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Reply with the single word: ok"}
                ],
                max_tokens=128,
                temperature=0,
                reasoning_effort="low",
            )
            completion = raw.parse()
            headers = {
                key: value
                for key, value in raw.headers.items()
                if "ratelimit" in key.lower() or "retry-after" in key.lower()
            }
            results.append(
                {
                    "model": model,
                    "ok": True,
                    "content": completion.choices[0].message.content,
                    "usage": completion.usage.model_dump()
                    if completion.usage
                    else {},
                    "headers": headers,
                    "elapsed_sec": round(time.time() - started, 2),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model": model,
                    "ok": False,
                    "error": _safe(f"{type(exc).__name__}: {exc}")[:500],
                    "elapsed_sec": round(time.time() - started, 2),
                }
            )

    return {
        "provider": "groq",
        "documented_limits": {
            "rpm": 30,
            "rpd": 1000,
            "tpm": 8000,
            "tpd": 200000,
            "status": "VERIFIED",
            "source": "official Groq docs",
        },
        "rate_limit_429": "not triggered in probe",
        "results": results,
    }


def main() -> int:
    config.ensure_work_dir()
    result = {"gemini": gemini_probe(), "groq": groq_probe()}
    out = config.WORK_DIR / "step0_probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "gemini_target": {
            "model": result["gemini"]["target_model"],
            "status": result["gemini"]["target"].get("status"),
            "ok": result["gemini"]["target"].get("ok"),
        },
        "gemini_fallback": {
            "model": result["gemini"]["fallback_model"],
            "status": result["gemini"]["fallback"].get("status"),
            "ok": result["gemini"]["fallback"].get("ok"),
        },
        "gemini_thinking_levels": [
            {
                "status": entry.get("status"),
                "ok": entry.get("ok"),
                "level": entry.get("thinking_level"),
            }
            for entry in result["gemini"]["thinking_levels"]
        ],
        "groq": [
            {
                "model": entry["model"],
                "ok": entry["ok"],
                "content": entry.get("content"),
            }
            for entry in result["groq"]["results"]
        ],
        "saved": str(out),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
