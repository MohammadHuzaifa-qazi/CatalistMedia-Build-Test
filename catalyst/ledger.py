from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone

from . import config, llm, store, tools, validate

PROGRESS_PATH = config.WORK_DIR / "ledger_progress.json"
METRICS_PATH = config.METRICS_DIR / "ledger.json"

SYSTEM_PROMPT = (
    "You extract structured meeting memory from AMI design-meeting transcripts. "
    "Return JSON only and use exactly the four top-level arrays in the schema. "
    "Extract only information explicitly stated in the supplied transcript window; "
    "never invent, infer, complete missing fields, or use outside knowledge. "
    "Every record must point to one speaker turn in this window and include "
    "that turn's exact turn_id and an exact quote copied verbatim from that same turn. "
    "Quotes must contain at least 6 words. Do not combine words from different turns. "
    "If a category has no qualifying evidence, return an empty array for that category. "
    "Do not create an action from a status update, suggestion, question, or meeting-time "
    "announcement unless someone explicitly undertakes a concrete task. "
    "Do not assign an owner unless the transcript explicitly identifies the person."
)


LEDGER_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short subject explicitly discussed in the cited turn."},
                    "decision": {"type": "string", "description": "Agreed choice or decision, not an unaccepted proposal."},
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["topic", "decision", "turn_id", "quote"],
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Explicitly identified responsible person; omit if not explicit."},
                    "task": {"type": "string", "description": "Concrete task someone explicitly commits or is assigned to do."},
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["task", "turn_id", "quote"],
            },
        },
        "disagreements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "who": {"type": "string"},
                    "position": {"type": "string", "description": "Conflicting position explicitly expressed in the cited turn."},
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["topic", "position", "turn_id", "quote"],
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": "Numeric value explicitly stated in the cited turn, represented as text."},
                    "unit": {"type": "string", "description": "Unit explicitly stated, such as Euro or minutes."},
                    "component": {"type": "string", "description": "What the number refers to, such as selling price or target revenue."},
                    "role": {
                        "type": "string",
                        "enum": [
                            "total",
                            "limit",
                            "unit_price",
                            "conversion",
                        ],
                    },
                    "turn_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["value", "role", "turn_id", "quote"],
            },
        },
    },
    "required": ["decisions", "actions", "disagreements", "figures"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "general").lower()).strip("_")
    return (slug or "general")[:80]


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _line(turn: dict) -> str:
    return f'[{turn["turn_id"]}] {turn["speaker"]}: {turn["text"]}'


def build_windows(
    turns: list[dict],
    max_tokens: int = 2500,
    overlap_tokens: int = 300,
) -> list[dict]:
    windows: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for turn in turns:
        line = _line(turn)
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_tokens:
            windows.append(current)
            overlap: list[dict] = []
            overlap_total = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(_line(previous))
                if overlap_total + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_total += previous_tokens
            current = overlap
            current_tokens = overlap_total
        current.append(turn)
        current_tokens += line_tokens

    if current:
        windows.append(current)

    result = []
    for index, window in enumerate(windows):
        text = "\n".join(_line(turn) for turn in window)
        result.append(
            {
                "index": index,
                "turns": window,
                "text": text,
                "turn_ids": {turn["turn_id"] for turn in window},
            }
        )
    return result


def _question_teams() -> list[str]:
    data = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    teams = {question["team"] for question in data["questions"]}
    teams.add(data["brief_task"]["team"])
    return sorted(teams)


def _meeting_turns(
    conn: sqlite3.Connection, meeting_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM turns WHERE meeting_id = ? ORDER BY seq", (meeting_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def _select_meetings(
    conn: sqlite3.Connection,
    teams: list[str] | None,
    all_teams: bool,
    limit: int | None,
) -> list[dict]:
    if all_teams:
        rows = conn.execute(
            "SELECT * FROM meetings ORDER BY meeting_id"
        ).fetchall()
    else:
        selected = teams or _question_teams()
        placeholders = ",".join("?" for _ in selected)
        rows = conn.execute(
            f"SELECT * FROM meetings WHERE team IN ({placeholders}) "
            "ORDER BY meeting_id",
            selected,
        ).fetchall()
    if limit is not None:
        rows = rows[:limit]
    return [dict(row) for row in rows]


def _speaker(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    match = re.search(r"\bSpeaker\s+([A-D])\b", text, re.I)
    if match:
        return match.group(1).upper()
    if re.fullmatch(r"[A-D]", text.upper()):
        return text.upper()
    return text[:40] or None


def _record_id(
    kind: str, meeting_id: str, turn_id: str, text: str
) -> str:
    digest = hashlib.sha1(
        f"{kind}|{meeting_id}|{turn_id}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{kind}_{digest}"


def _drop(
    conn: sqlite3.Connection,
    team: str,
    meeting_id: str,
    kind: str,
    reason: str,
    raw: dict,
    created_at: str,
) -> None:
    conn.execute(
        "INSERT INTO ledger_drops "
        "(team, meeting_id, kind, reason, raw_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            team,
            meeting_id,
            kind,
            reason,
            json.dumps(raw, ensure_ascii=False),
            created_at,
        ),
    )


def _verify(
    conn: sqlite3.Connection,
    ctx: tools.RunContext,
    meeting_id: str,
    raw: dict,
    window: dict,
) -> tuple[str | None, str | None, str | None, str | None]:
    quote = str(raw.get("quote", "")).strip()
    if not quote:
        return None, None, None, "missing_quote"
    result = validate.verify_quote(conn, ctx, meeting_id, quote)
    if not result.ok:
        return None, None, None, f"verify:{result.reason}"
    if result.turn_id in window["turn_ids"]:
        turn = conn.execute(
            "SELECT speaker FROM turns WHERE turn_id = ?",
            (result.turn_id,),
        ).fetchone()
        speaker = turn["speaker"] if turn else None
        return quote, result.turn_id, speaker, None
    normalized_quote = validate.normalize(quote)
    pattern = r"(?<!\w)" + re.escape(normalized_quote) + r"(?!\w)"
    for turn in window["turns"]:
        if re.search(pattern, validate.normalize(turn["text"])):
            return quote, turn["turn_id"], turn["speaker"], None
    return None, None, None, "quote_not_in_window"


def store_window(
    conn: sqlite3.Connection,
    meeting: dict,
    window: dict,
    data: dict,
    result: llm.LLMResult,
) -> dict:
    team = meeting["team"]
    meeting_id = meeting["meeting_id"]
    letter = meeting["letter"]
    ctx = tools.RunContext(team=team, as_of=None, mode="answer")
    extracted_at = _now_iso()
    provider = result.provider
    model = result.model
    counts = {
        "decisions": 0,
        "actions": 0,
        "disagreements": 0,
        "figures": 0,
        "drops": 0,
    }

    for raw in data.get("decisions") or []:
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic", "")).strip()
        decision = str(raw.get("decision", "")).strip()
        if not topic or not decision:
            _drop(
                conn,
                team,
                meeting_id,
                "decision",
                "missing_fields",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        quote, turn_id, speaker, reason = _verify(
            conn, ctx, meeting_id, raw, window
        )
        if reason or not turn_id:
            _drop(
                conn,
                team,
                meeting_id,
                "decision",
                reason or "verify:not_found",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        decision_id = _record_id("dec", meeting_id, turn_id, decision)
        conn.execute(
            "INSERT OR REPLACE INTO decisions "
            "(decision_id, team, topic, topic_key, decision, meeting_id, "
            "meeting_letter, turn_id, speaker, quote, provider, model, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision_id,
                team,
                topic,
                normalize_topic(topic),
                decision,
                meeting_id,
                letter,
                turn_id,
                _speaker(speaker),
                quote,
                provider,
                model,
                extracted_at,
            ),
        )
        scopes = ["all"] + (["asof_c"] if letter <= "c" else [])
        for scope in scopes:
            conn.execute(
                "INSERT OR REPLACE INTO topic_key_map "
                "(team, scope, decision_id, topic_key, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    team,
                    scope,
                    decision_id,
                    normalize_topic(topic),
                    topic,
                ),
            )
        counts["decisions"] += 1

    for raw in data.get("actions") or []:
        if not isinstance(raw, dict):
            continue
        task = str(raw.get("task", "")).strip()
        if not task:
            _drop(
                conn,
                team,
                meeting_id,
                "action",
                "missing_fields",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        quote, turn_id, speaker, reason = _verify(
            conn, ctx, meeting_id, raw, window
        )
        if reason or not turn_id:
            _drop(
                conn,
                team,
                meeting_id,
                "action",
                reason or "verify:not_found",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        action_id = _record_id("act", meeting_id, turn_id, task)
        conn.execute(
            "INSERT OR REPLACE INTO actions "
            "(action_id, team, owner, task, meeting_id, meeting_letter, "
            "turn_id, quote, provider, model, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                action_id,
                team,
                _speaker(raw.get("owner")),
                task,
                meeting_id,
                letter,
                turn_id,
                quote,
                provider,
                model,
                extracted_at,
            ),
        )
        counts["actions"] += 1

    for raw in data.get("disagreements") or []:
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic", "")).strip()
        position = str(raw.get("position", "")).strip()
        if not topic or not position:
            _drop(
                conn,
                team,
                meeting_id,
                "disagreement",
                "missing_fields",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        quote, turn_id, speaker, reason = _verify(
            conn, ctx, meeting_id, raw, window
        )
        if reason or not turn_id:
            _drop(
                conn,
                team,
                meeting_id,
                "disagreement",
                reason or "verify:not_found",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        disagreement_id = _record_id(
            "dis", meeting_id, turn_id, position
        )
        conn.execute(
            "INSERT OR REPLACE INTO disagreements "
            "(disagreement_id, team, topic, who, position, meeting_id, "
            "meeting_letter, turn_id, quote, provider, model, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                disagreement_id,
                team,
                topic,
                _speaker(raw.get("who") or speaker),
                position,
                meeting_id,
                letter,
                turn_id,
                quote,
                provider,
                model,
                extracted_at,
            ),
        )
        counts["disagreements"] += 1

    for raw in data.get("figures") or []:
        if not isinstance(raw, dict):
            continue
        value = str(raw.get("value", "")).strip()
        role = str(raw.get("role", "")).strip()
        if not value or role not in {
            "total",
            "limit",
            "unit_price",
            "conversion",
        }:
            _drop(
                conn,
                team,
                meeting_id,
                "figure",
                "bad_fields",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        quote, turn_id, _speaker_value, reason = _verify(
            conn, ctx, meeting_id, raw, window
        )
        if reason or not turn_id:
            _drop(
                conn,
                team,
                meeting_id,
                "figure",
                reason or "verify:not_found",
                raw,
                extracted_at,
            )
            counts["drops"] += 1
            continue
        figure_id = _record_id(
            "fig", meeting_id, turn_id, f"{value}|{role}"
        )
        conn.execute(
            "INSERT OR REPLACE INTO figures "
            "(figure_id, team, meeting_id, meeting_letter, value, unit, "
            "component, role, turn_id, quote, provider, model, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                figure_id,
                team,
                meeting_id,
                letter,
                value,
                str(raw.get("unit", "")).strip() or None,
                str(raw.get("component", "")).strip() or None,
                role,
                turn_id,
                quote,
                provider,
                model,
                extracted_at,
            ),
        )
        counts["figures"] += 1

    return counts


def _user_prompt(meeting: dict, window_text: str) -> str:
    return (
        f"Meeting: {meeting['meeting_id']} (team {meeting['team']}, "
        f"meeting {meeting['letter']}).\\n"
        "Extract only evidence present in the window. Return exactly these four "
        "top-level arrays: decisions, actions, disagreements, figures.\\n\\n"
        "CATEGORY RULES:\\n"
        "1. decisions: include an agreed choice or explicit decision, not merely an idea. "
        "Provide topic and a concise decision description.\\n"
        "2. actions: include a concrete task someone explicitly agrees or is assigned "
        "to do. Provide task. Provide owner only when the person is explicitly identified; "
        "otherwise omit owner. Do not treat meeting scheduling, general discussion, "
        "or suggestions as actions.\\n"
        "3. disagreements: include clearly conflicting positions. Provide topic and "
        "the position expressed in this turn. Do not invent the other side.\\n"
        "4. figures: extract each materially distinct numeric amount/quantity explicitly "
        "mentioned. Provide value as a string, role from total/limit/unit_price/conversion, "
        "and unit/component when stated. Use total for a target or overall amount, "
        "limit for a cap/maximum, unit_price for a per-item selling price, and conversion "
        "for an explicitly stated conversion or equivalent. Do not calculate or derive "
        "numbers that are not spoken. Preserve approximate/uncertain wording in the "
        "component if needed; do not present an estimate as exact.\\n\\n"
        "CITATION RULES:\\n"
        "Every record must include turn_id and an exact quote of at least 6 words "
        "copied verbatim from one single speaker turn displayed below. The quote and "
        "turn_id must refer to the same turn. Never paraphrase the quote, combine turns, "
        "or invent a turn_id. If evidence is insufficient, omit that record.\\n\\n"
        "Return each record as a JSON object matching the schema exactly. "
        "For a spoken price such as twenty five Euro, extract the numeric value "
        "and unit only when the amount is explicitly spoken; identify the component "
        "and role from the surrounding words, without calculating new values.\\n\\n"
        "Window:\\n"
        f"{window_text}"
    )


def _valid_extraction_payload(data: object) -> tuple[bool, str | None]:
    """Validate the model's outer JSON shape before a window is checkpointed."""
    if not isinstance(data, dict):
        return False, "response_not_object"
    required = ("decisions", "actions", "disagreements", "figures")
    for key in required:
        value = data.get(key)
        if not isinstance(value, list):
            return False, f"{key}_not_array"
        if any(not isinstance(item, dict) for item in value):
            return False, f"{key}_contains_non_object"
    return True, None


def _load_progress() -> set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return set(data.get("done", []))
    except Exception:
        return set()


def _save_progress(done: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps({"done": sorted(done)}, indent=2),
        encoding="utf-8",
    )


def _reset(conn: sqlite3.Connection) -> None:
    for table in (
        "decision_supersedes",
        "action_closed_by",
        "disagreement_resolved_by",
        "topic_key_map",
        "decisions",
        "actions",
        "disagreements",
        "figures",
        "ledger_drops",
    ):
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()


def run(
    teams: list[str] | None = None,
    limit: int | None = None,
    max_windows: int | None = None,
    reset: bool = False,
    all_teams: bool = False,
    window_tokens: int = 2500,
) -> dict:
    conn = store.connect()
    if reset:
        _reset(conn)

    done = _load_progress()
    selected = _select_meetings(conn, teams, all_teams, limit)
    engine = llm.LLM(conn=conn)
    started = time.time()
    summary = {
        "meetings": len(selected),
        "windows": 0,
        "decisions": 0,
        "actions": 0,
        "disagreements": 0,
        "figures": 0,
        "drops": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
        "errors": [],
    }
    stop = False

    for meeting in selected:
        if stop:
            break
        turns = _meeting_turns(conn, meeting["meeting_id"])
        windows = build_windows(turns, max_tokens=window_tokens)
        for window in windows:
            key = f'{meeting["meeting_id"]}:{window["index"]}'
            if key in done:
                continue
            if max_windows is not None and summary["windows"] >= max_windows:
                stop = True
                break
            try:
                result = engine.chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _user_prompt(meeting, window["text"]),
                        },
                    ],
                    response_schema=LEDGER_SCHEMA,
                    max_tokens=2048,
                    temperature=0.1,
                    tier="bulk",
                )
            except llm.AllProvidersFailed as exc:
                summary["errors"].append(str(exc)[:1000])
                stop = True
                break

            data = llm.parse_json_text(result.text)
            payload_ok, payload_error = _valid_extraction_payload(data)
            if not payload_ok:
                reason = f"invalid_extraction_payload:{payload_error}"
                _drop(
                    conn,
                    meeting["team"],
                    meeting["meeting_id"],
                    "window",
                    reason,
                    {"text": result.text[:1000]},
                    _now_iso(),
                )
                conn.commit()
                counts = {
                    "decisions": 0,
                    "actions": 0,
                    "disagreements": 0,
                    "figures": 0,
                    "drops": 1,
                }
                # Do NOT checkpoint malformed model output as completed.
                # The next invocation will retry this window.
                summary["errors"].append(
                    f'{meeting["meeting_id"]} window {window["index"]}: {reason}'
                )
            else:
                try:
                    counts = store_window(
                        conn, meeting, window, data, result
                    )
                    conn.commit()
                    done.add(key)
                    _save_progress(done)
                except Exception as exc:
                    conn.rollback()
                    reason = f"store_window_error:{type(exc).__name__}:{exc}"
                    _drop(
                        conn,
                        meeting["team"],
                        meeting["meeting_id"],
                        "window",
                        reason[:500],
                        {"text": result.text[:1000]},
                        _now_iso(),
                    )
                    conn.commit()
                    counts = {
                        "decisions": 0,
                        "actions": 0,
                        "disagreements": 0,
                        "figures": 0,
                        "drops": 1,
                    }
                    # Keep the window pending so it can be retried.
                    summary["errors"].append(
                        f'{meeting["meeting_id"]} window {window["index"]}: {reason}'
                    )
            summary["windows"] += 1
            for field in (
                "decisions",
                "actions",
                "disagreements",
                "figures",
                "drops",
            ):
                summary[field] += counts[field]
            usage = result.usage or {}
            summary["prompt_tokens"] += usage.get("prompt_tokens", 0)
            summary["completion_tokens"] += usage.get(
                "completion_tokens", 0
            )
            summary["thoughts_tokens"] += usage.get(
                "thoughts_tokens", 0
            )
            summary["total_tokens"] += usage.get("total_tokens", 0)
            print(
                f'{meeting["meeting_id"]} w{window["index"]} '
                f'{result.provider}/{result.model} '
                f'fallback={result.fallback_used} '
                f'dec={counts["decisions"]} act={counts["actions"]} '
                f'dis={counts["disagreements"]} fig={counts["figures"]} '
                f'drop={counts["drops"]} '
                f'checkpointed={key in done}',
                flush=True,
            )

    summary["elapsed_sec"] = round(time.time() - started, 2)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {"ts": _now_iso(), **summary}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalyst.ledger")
    parser.add_argument("--teams", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--window-tokens", type=int, default=2500)
    args = parser.parse_args(argv)
    teams = [
        team.strip() for team in args.teams.split(",") if team.strip()
    ] or None
    result = run(
        teams=teams,
        limit=args.limit,
        max_windows=args.max_windows,
        reset=args.reset,
        all_teams=args.all,
        window_tokens=args.window_tokens,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
