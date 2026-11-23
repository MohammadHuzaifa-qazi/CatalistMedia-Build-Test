from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, llm, store, tools, validate

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "bullets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "meeting_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["meeting_id", "quote"],
                        },
                    },
                },
                "required": ["summary", "citations"],
            },
        }
    },
    "required": ["bullets"],
}

SECTION_TITLES = {
    "decided": "DECIDED:",
    "changed": "WHAT CHANGED:",
    "open_actions": "OPEN ACTIONS AND OWNERS:",
    "open_disagreements": "UNRESOLVED DISAGREEMENTS:",
}

SYSTEM_PROMPT = """You write one section of a factual pre-meeting brief for an AMI product design team using ONLY verified records from meetings a, b, and c.

STRICT RULES:
1. NEVER reference meeting d (the final meeting) anywhere.
2. NEVER make predictions or forecasts about the upcoming/final meeting, and NEVER use future tense ("will", "next meeting", "likely", "expected to", "going to") in your prose summary. Use past or present factual tense only (e.g., "The team agreed on...", "Target audience is...").
3. Deduplicate overlapping records into 4 to 8 crisp, high-signal bullets (or fewer if fewer distinct facts exist).
4. Every bullet MUST include 1 or 2 citations copied VERBATIM from the provided records' `quote` fields (preserving exact ASCII spelling and punctuation, >=6 words, no Unicode hyphens or curly quotes).
5. Always identify speakers by letter (Speaker A, B, C, D); include their explicit role ONLY if listed in the provided role mapping. Use plain ASCII characters only (use `-` and `'`, never Unicode dashes).
6. Return ONLY a raw JSON object matching {"bullets": [{"summary": "...", "citations": [{"meeting_id": "TS3012b", "quote": "..."}]}]}.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_explicit_roles(conn: sqlite3.Connection, ctx: tools.RunContext) -> dict[str, str]:
    """Best-effort extraction of explicitly stated roles from the first in-scope meeting."""
    first_meeting = conn.execute(
        "SELECT meeting_id FROM meetings WHERE team = ? AND letter <= ? ORDER BY letter LIMIT 1",
        (ctx.team, ctx.as_of or "z"),
    ).fetchone()
    if first_meeting is None:
        return {}
    turns = tools.get_turns(
        conn, ctx, first_meeting["meeting_id"], seq_start=0, seq_end=24, max_turns=25
    )
    pattern = re.compile(
        r"(?:i am|i'm|i'll be|i will be|my role is|you are)\s+"
        r"(?:the|your|a|an)?\s*(?:(?!not\b)[a-z']+\s+){0,5}?"
        r"(project manager|industrial designer|user interface(?:\s+designer)?|marketing(?:\s+expert|\s+director|\s+manager)?)",
        re.IGNORECASE,
    )
    direct_intro = re.compile(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(project manager|industrial designer|user interface(?:\s+designer)?|marketing(?:\s+expert|\s+director|\s+manager)?)",
        re.IGNORECASE,
    )
    roles: dict[str, str] = {}
    for t in turns:
        spk = t["speaker"].replace("Speaker ", "").strip()
        text = t["text"]
        m = pattern.search(text) or direct_intro.search(text)
        if m and spk not in roles:
            # Avoid attributing "you are ..." to the speaker who asked it
            if m.group(0).lower().startswith("you are"):
                continue
            roles[spk] = m.group(1).strip().title()
    return roles


def _dedupe_records(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in records:
        key = (r.get("turn_id", ""), (r.get("quote") or "").strip().lower()[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def gather_section_inputs(
    conn: sqlite3.Connection, ctx: tools.RunContext
) -> dict[str, Any]:
    """Load strictly as_of-scoped ledger records for the 4 brief sections.

    Never touches meeting d when ctx.as_of == 'c'.
    """
    scope = "asof_c" if ctx.as_of == "c" else "all"
    as_of = ctx.as_of or "z"

    decisions = tools.ledger(conn, ctx, kind="decisions")
    actions = tools.ledger(conn, ctx, kind="actions")
    disagreements = tools.ledger(conn, ctx, kind="disagreements")
    figures = tools.ledger(conn, ctx, kind="figures")
    roles = _extract_explicit_roles(conn, ctx)

    # 1. Decided: current decisions + non-superseded top-meeting decisions + figures
    # Prioritize settled ('current') and later-meeting records so key decisions never truncate.
    decided = _dedupe_records(
        sorted(
            [d for d in decisions if d["status"] in {"current", "disputed"}],
            key=lambda d: (0 if d["status"] == "current" else 1, -ord(d["meeting_letter"][0]), d["id"]),
        )
    )

    # 2. Changed: explicit supersede edges + superseded decisions paired with newer peers
    superseded_ids = {d["id"] for d in decisions if d["status"] == "superseded"}
    by_id = {d["id"]: d for d in decisions}
    changed_pairs: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    edge_rows = conn.execute(
        "SELECT older_id, newer_id FROM decision_supersedes "
        "WHERE scope = ? AND superseding_letter <= ? "
        "AND newer_id IN (SELECT decision_id FROM decisions WHERE team = ? AND meeting_letter <= ?)",
        (scope, as_of, ctx.team, as_of),
    ).fetchall()
    for er in edge_rows:
        old_d = by_id.get(er["older_id"])
        new_d = by_id.get(er["newer_id"])
        if old_d and new_d:
            key = (old_d["id"], new_d["id"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                changed_pairs.append({"old": old_d, "new": new_d, "source": "supersede_edge"})

    # Also pair superseded decisions sharing topic_key with their latest in-scope peer
    by_topic: dict[str, list[dict]] = {}
    for d in decisions:
        tk = d.get("topic_key") or d.get("topic") or ""
        by_topic.setdefault(tk, []).append(d)
    for tk, group in by_topic.items():
        if len(group) < 2:
            continue
        latest_letter = max(d["meeting_letter"] for d in group)
        latests = [d for d in group if d["meeting_letter"] == latest_letter]
        for old_d in group:
            if old_d["id"] in superseded_ids and old_d["meeting_letter"] < latest_letter:
                new_d = latests[0]
                key = (old_d["id"], new_d["id"])
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    changed_pairs.append({"old": old_d, "new": new_d, "source": "topic_progression"})

    # 3. Open actions + owners
    open_actions = _dedupe_records([a for a in actions if a["status"] == "open"])

    # 4. Open disagreements
    open_disagreements = _dedupe_records(
        [g for g in disagreements if g["status"] == "open"]
    )

    return {
        "roles": roles,
        "decided": decided,
        "figures": _dedupe_records(figures),
        "changed": changed_pairs,
        "open_actions": open_actions,
        "open_disagreements": open_disagreements,
    }


def _format_roles_line(roles: dict[str, str]) -> str:
    if not roles:
        return "Explicit speaker roles: none verified; use speaker letters A-D only."
    pairs = ", ".join(f"Speaker {k} = {v}" for k, v in sorted(roles.items()))
    return f"Explicit speaker roles (from meeting a): {pairs}"


def _build_section_prompt(
    section: str, team: str, as_of: str | None, inputs: dict[str, Any], max_chars: int = 12000
) -> str:
    roles_line = _format_roles_line(inputs["roles"])
    scope_line = f"Team: {team} | In-scope meetings: a through {as_of or 'd'} ONLY."
    lines: list[str] = [scope_line, roles_line]

    if section == "decided":
        lines.append(
            "Task: Summarize the team's settled decisions up through meeting "
            f"{as_of or 'd'} into 5-8 comprehensive, factual bullets covering target audience, "
            "product scope, core features (e.g. locator, no speech recognition, LCD status), "
            "button count/layout, branding/materials, and budget/selling-price figures. "
            "Use past or present tense ONLY (never 'will'). Every bullet must have 1-2 verbatim quotes."
        )
        lines.append("\nDecisions in scope:")
        used = sum(len(x) for x in lines)
        for d in inputs["decided"]:
            line = (
                f'- [{d["meeting_id"]} | {d["status"]} | topic={d["topic"]} | Speaker {d["speaker"]}] '
                f'{d["text"]} | quote="{d["quote"]}"'
            )
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        if inputs["figures"]:
            lines.append("\nFigures in scope:")
            for f in inputs["figures"]:
                line = (
                    f'- [{f["meeting_id"]} | figure] {f.get("topic") or ""}: {f["text"]} | '
                    f'quote="{f["quote"]}"'
                )
                if used + len(line) > max_chars:
                    break
                lines.append(line)
                used += len(line) + 1

    elif section == "changed":
        lines.append(
            "Task: Summarize what changed across meetings up through meeting "
            f"{as_of or 'd'}. For each change, clearly state the earlier position (with meeting letter) "
            "and the later position (with meeting letter), citing exact quotes for old and/or new."
        )
        lines.append("\nChanged decision pairs:")
        used = sum(len(x) for x in lines)
        for pair in inputs["changed"]:
            old_d, new_d = pair["old"], pair["new"]
            line = (
                f'- ({pair["source"]}) OLD [{old_d["meeting_id"]}]: {old_d["text"]} '
                f'| old_quote="{old_d["quote"]}" -> '
                f'NEW [{new_d["meeting_id"]}]: {new_d["text"]} | new_quote="{new_d["quote"]}"'
            )
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1

    elif section == "open_actions":
        lines.append(
            "Task: List the team's open (unclosed) action items as of meeting "
            f"{as_of or 'd'} with each item's true owner (Speaker letter + verified role). "
            "Check each quote carefully: if Speaker A (Project Manager) assigns a task to 'you', "
            "the team, Industrial Designer, or Marketing Expert, attribute ownership accurately "
            "rather than listing Speaker A as doing every task. Merge duplicates into 5-8 concise bullets "
            "in present/past factual tense (never 'will')."
        )
        lines.append("\nOpen actions in scope:")
        used = sum(len(x) for x in lines)
        for a in inputs["open_actions"]:
            line = (
                f'- [{a["meeting_id"]} | owner=Speaker {a["speaker"]}] '
                f'{a["text"]} | quote="{a["quote"]}"'
            )
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1

    elif section == "open_disagreements":
        lines.append(
            "Task: Summarize the unresolved disagreements as of meeting "
            f"{as_of or 'd'}, naming who (Speaker letter + role) holds each position. "
            "Group related points by topic into 4-8 concise bullets. Do NOT predict how they will be resolved."
        )
        lines.append("\nOpen disagreements in scope:")
        used = sum(len(x) for x in lines)
        for g in inputs["open_disagreements"]:
            line = (
                f'- [{g["meeting_id"]} | topic={g["topic"]} | Speaker {g["speaker"]}] '
                f'{g["text"]} | quote="{g["quote"]}"'
            )
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
    else:
        raise ValueError(f"unknown brief section: {section}")

    return "\n".join(lines)


_ASCII_TRANS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)


def _format_and_verify_section(
    conn: sqlite3.Connection,
    ctx: tools.RunContext,
    section: str,
    raw_text: str,
) -> tuple[list[dict], list[str]]:
    """Parse JSON bullets, verify every citation and brief rule deterministically."""
    data = llm.parse_json_text(raw_text)
    if not isinstance(data, dict) or not isinstance(data.get("bullets"), list):
        return [], [f"{section}:invalid_json"]

    valid_bullets: list[dict] = []
    errors: list[str] = []
    forbidden_id = f"{ctx.team}d" if ctx.team and ctx.as_of == "c" else None

    for idx, item in enumerate(data["bullets"]):
        if not isinstance(item, dict):
            errors.append(f"{section}[{idx}]:bad_bullet")
            continue
        summary = " ".join(str(item.get("summary") or "").strip().split()).translate(_ASCII_TRANS)
        # Strip any inline citation tags the model may have already inserted in summary
        summary = validate.CITE.sub("", summary).strip().rstrip(" ;")
        if not summary:
            errors.append(f"{section}[{idx}]:empty_summary")
            continue
        if forbidden_id and re.search(rf"\b{re.escape(forbidden_id)}\b", summary):
            errors.append(f"{section}[{idx}]:asof_violation")
            continue
        if validate.PREDICTIVE.search(summary):
            errors.append(f"{section}[{idx}]:predicts_final_meeting")
            continue

        raw_cites = item.get("citations") or []
        verified_cites: list[dict] = []
        cite_errors: list[str] = []
        for c in raw_cites:
            if not isinstance(c, dict):
                continue
            mid = str(c.get("meeting_id") or "").strip()
            if "#" in mid:
                mid = mid.split("#", 1)[0].strip()
            quote = str(c.get("quote") or "").strip()
            if forbidden_id and mid.startswith(forbidden_id):
                cite_errors.append("asof_violation")
                continue
            vr = validate.verify_quote(conn, ctx, mid, quote)
            if not vr.ok and not quote.isascii():
                ascii_q = quote.translate(_ASCII_TRANS)
                vr_ascii = validate.verify_quote(conn, ctx, mid, ascii_q)
                if vr_ascii.ok:
                    quote, vr = ascii_q, vr_ascii
            if vr.ok:
                verified_cites.append(
                    {"meeting_id": mid, "quote": quote, "turn_id": vr.turn_id}
                )
            else:
                cite_errors.append(f"citation:{vr.reason}")

        if not verified_cites:
            errors.append(
                f"{section}[{idx}]:no_verified_citation({','.join(cite_errors) or 'none'})"
            )
            continue

        cite_str = " ".join(
            f'[{c["meeting_id"]}: "{c["quote"]}"]' for c in verified_cites[:2]
        )
        bullet_line = f"- {summary} {cite_str}"
        valid_bullets.append(
            {
                "section": section,
                "text": bullet_line,
                "citations": verified_cites[:2],
                "turn_ids": [c["turn_id"] for c in verified_cites[:2]],
            }
        )

    return valid_bullets, errors


def generate_brief(
    conn: sqlite3.Connection,
    engine: llm.LLM,
    *,
    team: str = "TS3012",
    as_of: str = "c",
    tier: str = "final",
) -> dict:
    ctx = tools.RunContext(team=team, as_of=as_of, mode="brief")
    started = time.time()
    run_id = f"brief_{uuid.uuid4().hex[:12]}"
    inputs = gather_section_inputs(conn, ctx)

    provenance: dict[str, Any] = {
        "run_id": run_id,
        "team": team,
        "as_of": as_of,
        "calls": 0,
        "providers": [],
        "models": [],
        "last_provider": None,
        "last_model": None,
        "fallback_used": False,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0,
            "billed_output_tokens": 0,
        },
        "section_details": {},
    }

    sections_out: dict[str, list[dict]] = {}
    section_order = ("decided", "changed", "open_actions", "open_disagreements")
    fatal_errors: list[str] = []

    for section in section_order:
        user_prompt = _build_section_prompt(section, team, as_of, inputs)
        # Enforce < 6K prompt budget strictly
        budget_chars = 12000
        while (
            _estimate_tokens(SYSTEM_PROMPT + user_prompt)
            > config.PROMPT_BUDGET_TOKENS - 400
            and budget_chars > 2000
        ):
            budget_chars = int(budget_chars * 0.75)
            user_prompt = _build_section_prompt(
                section, team, as_of, inputs, max_chars=budget_chars
            )

        # Use a larger completion cap for sections synthesizing large record sets
        # (decided/open_actions/open_disagreements) so Gemini thinkingBudget=1024
        # never truncates JSON output, while preserving cached changed responses.
        sec_max_tokens = (
            2560 if section in {"decided", "open_actions", "open_disagreements"} else 1536
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = engine.chat(
                messages,
                response_schema=SECTION_SCHEMA,
                max_tokens=sec_max_tokens,
                temperature=0.1,
                tier=tier,
            )
        except llm.LLMError as exc:
            fatal_errors.append(f"{section}:llm_error:{type(exc).__name__}")
            provenance["section_details"][section] = {"error": str(exc)[:600]}
            continue

        _accumulate_usage(provenance, resp)
        bullets, sec_errors = _format_and_verify_section(conn, ctx, section, resp.text)

        # One bounded repair attempt if section produced 0 valid bullets or had dropped bullets
        if not bullets and inputs[section]:
            repair_msg = (
                user_prompt
                + "\n\nPREVIOUS OUTPUT HAD VALIDATION ERRORS:\n"
                + "\n".join(f"- {e}" for e in sec_errors[:10])
                + "\nFix every error: copy quotes verbatim (>=6 words, ASCII only) from the listed records, "
                "use past/present tense only (no predictive words like 'will', 'likely', 'next meeting'), "
                "and never reference meeting d."
            )
            try:
                resp2 = engine.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": repair_msg[:18000]},
                    ],
                    response_schema=SECTION_SCHEMA,
                    max_tokens=2560,
                    temperature=0.1,
                    tier=tier,
                )
                _accumulate_usage(provenance, resp2)
                bullets, sec_errors = _format_and_verify_section(
                    conn, ctx, section, resp2.text
                )
            except llm.LLMError as exc:
                sec_errors.append(f"repair_llm_error:{type(exc).__name__}")

        sections_out[section] = bullets
        provenance["section_details"][section] = {
            "provider": resp.provider,
            "model": resp.model,
            "cached": resp.cached,
            "prompt_est_tokens": _estimate_tokens(SYSTEM_PROMPT + user_prompt),
            "valid_bullets": len(bullets),
            "dropped_errors": sec_errors,
        }
        if not bullets and inputs[section]:
            fatal_errors.append(f"{section}:no_verified_bullets")

    all_bullets: list[dict] = []
    text_blocks: list[str] = []
    for section in section_order:
        sec_bullets = sections_out.get(section) or []
        if not sec_bullets:
            continue
        text_blocks.append(SECTION_TITLES[section])
        for b in sec_bullets:
            text_blocks.append(b["text"])
            all_bullets.append(b)
        text_blocks.append("")

    raw_text = "\n".join(text_blocks).strip()

    # Validate BOTH with explicit bullets and directly on raw_text
    ok_bullets, errs_bullets = validate.validate_brief(
        conn, ctx, {"raw_text": raw_text, "bullets": [dict(b) for b in all_bullets]}
    )
    ok_raw, errs_raw = validate.validate_brief(conn, ctx, {"raw_text": raw_text})

    all_errors = list(fatal_errors)
    if not raw_text:
        all_errors.append("empty_brief")
    all_errors.extend(errs_bullets)
    for e in errs_raw:
        if e not in all_errors:
            all_errors.append(e)

    ok = len(all_errors) == 0 and ok_bullets and ok_raw and len(all_bullets) > 0
    provenance["elapsed_sec"] = round(time.time() - started, 2)

    if ok:
        _log_brief_run(conn, run_id, all_bullets, provenance)

    return {
        "ok": ok,
        "team": team,
        "as_of": as_of,
        "brief": raw_text,
        "raw_text": raw_text,
        "bullets": all_bullets,
        "sections": sections_out,
        "errors": all_errors,
        "provenance": provenance,
    }


def _accumulate_usage(provenance: dict, resp: llm.LLMResult) -> None:
    provenance["calls"] += 1
    if resp.provider not in provenance["providers"]:
        provenance["providers"].append(resp.provider)
    if resp.model not in provenance["models"]:
        provenance["models"].append(resp.model)
    provenance["last_provider"] = resp.provider
    provenance["last_model"] = resp.model
    if resp.fallback_used:
        provenance["fallback_used"] = True
    usage = resp.usage or {}
    for k in provenance["usage"]:
        provenance["usage"][k] += int(usage.get(k, 0) or 0)


def _log_brief_run(
    conn: sqlite3.Connection,
    run_id: str,
    bullets: list[dict],
    provenance: dict,
) -> None:
    provider = provenance.get("last_provider") or "unknown"
    model = provenance.get("last_model") or "unknown"
    conn.execute(
        "INSERT OR REPLACE INTO answer_runs "
        "(run_id, question_id, kind, provider, model, fallback_used, usage_json, created_at) "
        "VALUES (?, NULL, 'brief', ?, ?, ?, ?, ?)",
        (
            run_id,
            provider,
            model,
            1 if provenance.get("fallback_used") else 0,
            json.dumps(provenance.get("usage", {}), ensure_ascii=False),
            _now_iso(),
        ),
    )
    for bullet in bullets:
        for cite in bullet.get("citations") or []:
            conn.execute(
                "INSERT INTO citations_log "
                "(run_id, question_id, meeting_id, turn_id, quote, verified, method) "
                "VALUES (?, NULL, ?, ?, ?, 1, 'word_boundary_single_turn_match')",
                (
                    run_id,
                    cite["meeting_id"],
                    cite["turn_id"],
                    cite["quote"],
                ),
            )
    conn.commit()


def run(
    *,
    team: str = "TS3012",
    as_of: str = "c",
    out_path: str | Path | None = None,
) -> dict:
    save_path = Path(out_path) if out_path else (config.WORK_DIR / f"brief_{team}.json")
    conn = store.connect()
    engine = llm.LLM(conn=conn)
    try:
        result = generate_brief(conn, engine, team=team, as_of=as_of)
    finally:
        conn.close()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve any existing verified brief if a re-run fails (e.g. providers unavailable).
    if not result["ok"] and save_path.exists():
        try:
            prev = json.loads(save_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and prev.get("ok"):
                fail_path = save_path.with_suffix(".failed.json")
                fail_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                return {
                    "ok": False,
                    "team": team,
                    "as_of": as_of,
                    "bullets": 0,
                    "errors": result["errors"],
                    "calls": result["provenance"]["calls"],
                    "model": result["provenance"]["last_model"],
                    "preserved_previous": str(save_path),
                    "out_path": str(fail_path),
                }
        except Exception:
            pass
    save_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "ok": result["ok"],
        "team": team,
        "as_of": as_of,
        "bullets": len(result["bullets"]),
        "errors": result["errors"],
        "calls": result["provenance"]["calls"],
        "model": result["provenance"]["last_model"],
        "out_path": str(save_path),
    }
