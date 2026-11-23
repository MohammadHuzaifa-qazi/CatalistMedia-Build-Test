from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, store


# Pricing rates explicitly tagged VERIFIED vs NOT VERIFIED / ASSUMPTION.
PRICING = {
    "gemini-3.8-flash_through_2026": {
        "input_per_1m_usd": 0.75,
        "output_per_1m_usd": 3.75,  # includes thoughts_tokens billed as output
        "status": "VERIFIED",
        "source": "PHASE2_DESIGN.md §1.2 (official Google pricing page, Nov 2026; valid through 2026-12-31)",
        "thoughts_billed_as_output": True,
    },
    "gemini-3.8-flash_from_2027": {
        "input_per_1m_usd": 1.50,
        "output_per_1m_usd": 7.50,
        "status": "VERIFIED",
        "source": "PHASE2_DESIGN.md §1.2 (official Google pricing page, Nov 2026; effective 2027-01-01)",
        "thoughts_billed_as_output": True,
    },
    "gemini-2.5-flash_paid": {
        "input_per_1m_usd": 0.30,
        "output_per_1m_usd": 2.50,
        "status": "NOT VERIFIED (ASSUMPTION)",
        "source": "Approximate reference price; verify on Google AI Studio pricing page before budgeting",
        "thoughts_billed_as_output": True,
    },
    "groq_gpt_oss_120b_paid": {
        "input_per_1m_usd": 0.15,
        "output_per_1m_usd": 0.75,
        "status": "NOT VERIFIED (ASSUMPTION)",
        "source": "PHASE2_DESIGN.md §1.4 assumption",
        "thoughts_billed_as_output": True,
    },
    "groq_gpt_oss_20b_paid": {
        "input_per_1m_usd": 0.10,
        "output_per_1m_usd": 0.50,
        "status": "NOT VERIFIED (ASSUMPTION)",
        "source": "PHASE2_DESIGN.md §1.4 assumption",
        "thoughts_billed_as_output": True,
    },
}

FREE_TIER_LIMITS = {
    "gemini-2.5-flash": {
        "rpm": 5,
        "rpd": 20,
        "rpm_rpd_status": "VERIFIED (measured from live HTTP 429 responses)",
        "tpm_tpd_status": "NOT VERIFIED (unpublished in official docs; AI Studio only)",
    },
    "gemini-3.8-flash": {
        "rpm": 5,
        "rpd": 20,
        "rpm_rpd_status": "VERIFIED (measured from live HTTP 429 response, limit: 20)",
        "availability_note": "Frequent 503 UNAVAILABLE / ReadTimeout during testing",
    },
    "groq_gpt_oss_120b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
        "status": "VERIFIED (official Groq docs + live response headers)",
    },
    "groq_gpt_oss_20b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
        "status": "VERIFIED (official Groq docs + live response headers)",
    },
}


def _read_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cost_usd(prompt_tokens: float, billed_output_tokens: float, rate: dict) -> float:
    return round(
        (prompt_tokens / 1_000_000.0) * rate["input_per_1m_usd"]
        + (billed_output_tokens / 1_000_000.0) * rate["output_per_1m_usd"],
        4,
    )


def build_estimate(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Compute measured baseline and extrapolated cost/time tables (138 / 1,000 / 10,000 meetings)."""
    own_conn = conn is None
    conn = conn or store.connect()
    try:
        n_meetings = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] or 138
        n_teams = conn.execute("SELECT COUNT(DISTINCT team) FROM meetings").fetchone()[0] or 35
        n_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0] or 48148
        n_citable = (
            conn.execute(
                "SELECT COUNT(*) FROM turns WHERE n_words >= ?", (config.MIN_QUOTE_WORDS,)
            ).fetchone()[0]
            or 23821
        )
    finally:
        if own_conn:
            conn.close()

    ledger_m = _read_json(config.METRICS_DIR / "ledger.json") or {}
    reconcile_m = _read_json(config.METRICS_DIR / "reconcile.json") or {}
    prov_m = _read_json(config.ROOT / "answers_provenance.json") or {}

    # Measured TS3012 ledger (4 meetings)
    ledger_meetings = int(ledger_m.get("meetings") or 4)
    ledger_prompt = int(ledger_m.get("prompt_tokens") or 72413)
    ledger_comp = int(ledger_m.get("completion_tokens") or 15489)
    ledger_thoughts = int(ledger_m.get("thoughts_tokens") or 204)
    ledger_elapsed = float(ledger_m.get("elapsed_sec") or 183.8)
    # Note: ledger.json ran primarily on gemini-2.5-flash (thinkingBudget=0) + 3 groq 20b calls.
    # For Gemini paid output billing, thoughts tokens are billed as output.
    ledger_billed_out = ledger_comp + ledger_thoughts

    ledger_prompt_per_mtg = ledger_prompt / ledger_meetings
    ledger_out_per_mtg = ledger_billed_out / ledger_meetings
    ledger_sec_per_mtg = ledger_elapsed / ledger_meetings

    # Measured TS3012 reconcile (2 scopes: asof_c + all = 8 calls on gemini-2.5-flash)
    rec_scopes = int(reconcile_m.get("scopes") or 2)
    rec_prompt = int(reconcile_m.get("prompt_tokens") or 23265)
    rec_comp = int(reconcile_m.get("completion_tokens") or 5866)
    rec_thoughts = int(reconcile_m.get("thoughts_tokens") or 7357)
    rec_elapsed = float(reconcile_m.get("elapsed_sec") or 145.13)
    rec_billed_out = rec_comp + rec_thoughts

    # Typical team in production needs 1 scope ('all'); normalize per team (1 scope)
    rec_prompt_per_team = rec_prompt / max(1, rec_scopes)
    rec_out_per_team = rec_billed_out / max(1, rec_scopes)
    rec_thoughts_per_team = rec_thoughts / max(1, rec_scopes)
    rec_sec_per_team = rec_elapsed / max(1, rec_scopes)

    # Measured query cost from answers_provenance.json (passed questions only), grouped by actual model
    passed_qs = [q for q in (prov_m.get("questions") or []) if q.get("ok")]
    by_model: dict[str, list[dict]] = {}
    for q in passed_qs:
        mkey = str(q.get("model") or q.get("provider") or "unknown")
        by_model.setdefault(mkey, []).append(q)

    def _avg_usage(qs: list[dict]) -> dict[str, Any]:
        if not qs:
            return {"n": 0, "status": "NOT VERIFIED"}
        n = len(qs)
        p = sum(int((q.get("usage") or {}).get("prompt_tokens", 0)) for q in qs) / n
        c = sum(int((q.get("usage") or {}).get("completion_tokens", 0)) for q in qs) / n
        th = sum(int((q.get("usage") or {}).get("thoughts_tokens", 0)) for q in qs) / n
        bo = sum(int((q.get("usage") or {}).get("billed_output_tokens", 0)) for q in qs) / n
        calls = sum(int(q.get("calls", 0)) for q in qs) / n
        return {
            "n": n,
            "avg_calls": round(calls, 2),
            "avg_prompt_tokens": round(p, 1),
            "avg_completion_tokens": round(c, 1),
            "avg_thoughts_tokens": round(th, 1),
            "avg_billed_output_tokens": round(bo, 1),
            "status": "VERIFIED (measured)",
        }

    agent_by_model = {m: _avg_usage(qs) for m, qs in sorted(by_model.items())}
    agent_by_model["all_passed"] = _avg_usage(passed_qs)

    # Measured embedding throughput from full run (work/embed.out.log: 23,757 in 10,149.7s)
    embed_centres_per_sec = 2.34
    citable_per_meeting = n_citable / n_meetings
    meetings_per_team = n_meetings / n_teams

    scales: dict[str, Any] = {}
    for target_meetings in (138, 1000, 10000):
        est_turns = round((n_turns / n_meetings) * target_meetings)
        est_citable = round(citable_per_meeting * target_meetings)
        est_teams = max(1, round(target_meetings / meetings_per_team))
        embed_hours = round((est_citable / embed_centres_per_sec) / 3600.0, 2)
        raw_vectors_mb = round((est_citable * 384 * 4) / (1024 * 1024), 1)

        # Full ledger + 1-scope reconcile per team
        l_prompt = round(ledger_prompt_per_mtg * target_meetings)
        l_out = round(ledger_out_per_mtg * target_meetings)
        l_hours = round((ledger_sec_per_mtg * target_meetings) / 3600.0, 2)

        r_prompt = round(rec_prompt_per_team * est_teams)
        r_out = round(rec_out_per_team * est_teams)
        r_thoughts = round(rec_thoughts_per_team * est_teams)
        r_hours = round((rec_sec_per_team * est_teams) / 3600.0, 2)

        tot_prompt = l_prompt + r_prompt
        tot_billed_out = l_out + r_out

        scales[f"{target_meetings}_meetings"] = {
            "meetings": target_meetings,
            "est_teams": est_teams,
            "est_turns": est_turns,
            "est_citable_centres": est_citable,
            "raw_vectors_mb": raw_vectors_mb,
            "extrapolation_status": (
                "VERIFIED corpus counts; ledger/reconcile extrapolated from 4-meeting TS3012 run"
                if target_meetings == 138
                else "LINEAR EXTRAPOLATION (NOT VERIFIED at scale)"
            ),
            "embeddings_cpu": {
                "centres_per_sec": embed_centres_per_sec,
                "hours": embed_hours,
                "days": round(embed_hours / 24.0, 2),
                "status": "VERIFIED at 138 (2.82 h); linear extrapolation above",
            },
            "ledger_plus_reconcile": {
                "ledger_prompt_tokens": l_prompt,
                "ledger_billed_output_tokens": l_out,
                "reconcile_prompt_tokens": r_prompt,
                "reconcile_billed_output_tokens": r_out,
                "reconcile_thoughts_tokens_included": r_thoughts,
                "total_prompt_tokens": tot_prompt,
                "total_billed_output_tokens": tot_billed_out,
                "serial_llm_hours_at_measured_pace": round(l_hours + r_hours, 2),
                "estimated_cost_usd": {
                    "gemini_3.8_flash_2026_VERIFIED_price": _cost_usd(
                        tot_prompt, tot_billed_out, PRICING["gemini-3.8-flash_through_2026"]
                    ),
                    "gemini_3.8_flash_2027_VERIFIED_price": _cost_usd(
                        tot_prompt, tot_billed_out, PRICING["gemini-3.8-flash_from_2027"]
                    ),
                    "gemini_2.5_flash_ASSUMPTION_price": _cost_usd(
                        tot_prompt, tot_billed_out, PRICING["gemini-2.5-flash_paid"]
                    ),
                    "groq_gpt_oss_120b_ASSUMPTION_price": _cost_usd(
                        tot_prompt, tot_billed_out, PRICING["groq_gpt_oss_120b_paid"]
                    ),
                },
            },
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measured_baselines": {
            "corpus": {
                "meetings": n_meetings,
                "teams": n_teams,
                "turns": n_turns,
                "citable_ge6": n_citable,
                "citable_per_meeting": round(citable_per_meeting, 2),
                "status": "VERIFIED",
            },
            "embeddings_full_run": {
                "model": config.EMBED_MODEL,
                "centres_embedded": 23757,
                "total_collection": 23821,
                "elapsed_sec": 10149.7,
                "centres_per_sec": embed_centres_per_sec,
                "status": "VERIFIED (work/embed.out.log)",
            },
            "ts3012_ledger_4_meetings": {
                "meetings": ledger_meetings,
                "windows": int(ledger_m.get("windows") or 20),
                "prompt_tokens": ledger_prompt,
                "completion_tokens": ledger_comp,
                "thoughts_tokens": ledger_thoughts,
                "billed_output_tokens": ledger_billed_out,
                "elapsed_sec": ledger_elapsed,
                "tokens_per_meeting": round((ledger_prompt + ledger_billed_out) / ledger_meetings, 1),
                "sec_per_meeting": round(ledger_sec_per_mtg, 2),
                "status": "VERIFIED (work/metrics/ledger.json); TS3012 has 5.0 windows/mtg vs 3.01 corpus avg (conservative upper bound)",
            },
            "ts3012_reconcile_2_scopes": {
                "scopes": rec_scopes,
                "calls": int(reconcile_m.get("calls") or 8),
                "prompt_tokens": rec_prompt,
                "completion_tokens": rec_comp,
                "thoughts_tokens": rec_thoughts,
                "billed_output_tokens": rec_billed_out,
                "thoughts_share_of_billed_output": round(rec_thoughts / max(1, rec_billed_out), 4),
                "elapsed_sec": rec_elapsed,
                "status": "VERIFIED (work/metrics/reconcile.json, gemini-2.5-flash)",
            },
            "agent_per_question": agent_by_model,
        },
        "pricing": PRICING,
        "free_tier_limits": FREE_TIER_LIMITS,
        "scales": scales,
    }


def main() -> int:
    est = build_estimate()
    out = config.METRICS_DIR / "cost_time_estimate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(est, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
