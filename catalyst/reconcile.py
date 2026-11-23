from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

from . import config, ledger, llm, store, tools, validate

SYSTEM_PROMPT = (
    "You reconcile meeting memory. Return JSON only. You are given records "
    "with ids, meeting letters, text, and exact quotes. Identify topic "
    "clusters and explicit supersession, closure, and resolution edges. Do not "
    "infer. A bare mention never creates an edge. Only create an edge when a "
    "later record explicitly changes, closes, or resolves an earlier one. "
    "Meeting order is a<b<c<d. Every quote must be exact, at least 6 words, "
    "and copied from one speaker turn in the target meeting. Use only IDs "
    "shown."
)

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "topic_key": {"type": "string"},
                    "decision_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["topic", "decision_ids"],
            },
        },
    },
    "required": ["topics"],
}

SUPERSEDE_SCHEMA = {
    "type": "object",
    "properties": {
        "supersedes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "newer_id": {"type": "string"},
                    "older_id": {"type": "string"},
                },
                "required": ["newer_id", "older_id"],
            },
        },
    },
    "required": ["supersedes"],
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_closures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["action_id", "quote"],
            },
        },
    },
    "required": ["action_closures"],
}

DISAGREEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "disagreement_resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "disagreement_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["disagreement_id", "quote"],
            },
        },
    },
    "required": ["disagreement_resolutions"],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scope_as_of(scope: str) -> str | None:
    if scope not in {"all", "asof_c"}:
        raise ValueError(f"bad scope: {scope}")
    return "c" if scope == "asof_c" else None


def records_for_scope(
    conn: sqlite3.Connection, team: str, scope: str
) -> list[dict]:
    ctx = tools.RunContext(
        team=team, as_of=scope_as_of(scope), mode="answer"
    )
    records: list[dict] = []
    for kind in ("decisions", "actions", "disagreements"):
        records.extend(tools.ledger(conn, ctx, kind=kind))
    return records


def _decision_lines(records: list[dict]) -> str:
    return "\n".join(
        f'{record["id"]} | {record["meeting_letter"]} | {record.get("topic") or ""} | {record["text"]}'
        for record in records
        if record["kind"] == "decision"
    )


def _later_evidence_lines(
    decisions: list[dict], figures: list[dict], min_letter: str, max_chars: int = 8000
) -> str:
    lines: list[str] = []
    used = 0
    for record in decisions:
        if record["meeting_letter"] <= min_letter:
            continue
        line = (
            f'DEC {record["id"]} | {record["meeting_letter"]} | '
            f'{record["text"]} | "{record["quote"]}"'
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    for record in figures:
        if record["meeting_letter"] <= min_letter:
            continue
        line = (
            f'FIG {record["id"]} | {record["meeting_letter"]} | '
            f'{record.get("topic") or ""}: {record["text"]} | "{record["quote"]}"'
        )
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _action_lines(records: list[dict]) -> str:
    return "\n".join(
        f'{record["id"]} | {record["meeting_letter"]} | '
        f'{record["text"]} | {record["quote"]}'
        for record in records
        if record["kind"] == "action"
    )


def _disagreement_lines(records: list[dict]) -> str:
    return "\n".join(
        f'{record["id"]} | {record["meeting_letter"]} | '
        f'{record["text"]} | {record["quote"]}'
        for record in records
        if record["kind"] == "disagreement"
    )


def _verify_in_meeting(
    conn: sqlite3.Connection,
    team: str,
    meeting_letter: str,
    quote: str,
    as_of: str | None,
) -> dict | None:
    meeting = conn.execute(
        "SELECT meeting_id FROM meetings WHERE team = ? AND letter = ?",
        (team, meeting_letter),
    ).fetchone()
    if meeting is None:
        return None
    ctx = tools.RunContext(team=team, as_of=as_of, mode="answer")
    result = validate.verify_quote(conn, ctx, meeting["meeting_id"], quote)
    if not result.ok:
        return None
    return {
        "meeting_id": meeting["meeting_id"],
        "letter": meeting_letter,
        "turn_id": result.turn_id,
    }


def _find_quote_turn(
    conn: sqlite3.Connection,
    team: str,
    quote: str,
    as_of: str | None,
    min_letter: str | None = None,
) -> dict | None:
    sql = "SELECT meeting_id, letter FROM meetings WHERE team = ?"
    params: list = [team]
    if as_of:
        sql += " AND letter <= ?"
        params.append(as_of)
    if min_letter:
        sql += " AND letter > ?"
        params.append(min_letter)
    sql += " ORDER BY letter"
    ctx = tools.RunContext(team=team, as_of=as_of, mode="answer")
    for meeting in conn.execute(sql, params):
        result = validate.verify_quote(conn, ctx, meeting["meeting_id"], quote)
        if result.ok:
            return {
                "meeting_id": meeting["meeting_id"],
                "letter": meeting["letter"],
                "turn_id": result.turn_id,
            }
    return None


def _drop(drops: list[dict], kind: str, reason: str, raw: dict) -> None:
    drops.append({"kind": kind, "reason": reason, "raw": raw})


def _topic_parts(
    topic, decisions: dict[str, dict]
) -> tuple[list[str], str, str] | None:
    key = ""
    if isinstance(topic, list):
        ids = [str(value) for value in topic if isinstance(value, str)]
        label = ""
    elif isinstance(topic, dict):
        raw_ids = (
            topic.get("decision_ids")
            or topic.get("decisions")
            or topic.get("ids")
            or topic.get("records")
        )
        if raw_ids is None and isinstance(topic.get("topic"), list):
            raw_ids = topic["topic"]
        if not isinstance(raw_ids, list):
            return None
        ids = [str(value) for value in raw_ids if isinstance(value, str)]
        label = ""
        for field in ("topic", "name", "label", "subject"):
            value = topic.get(field)
            if isinstance(value, str) and value.strip():
                label = value.strip()
                break
        key = str(topic.get("topic_key") or "").strip()
    else:
        return None
    if not ids:
        return None
    if not label:
        first = decisions.get(ids[0])
        label = (first.get("topic") if first else "") or ""
    key = ledger.normalize_topic(key or label)
    return ids, label or key, key


def apply_proposals(
    conn: sqlite3.Connection,
    team: str,
    scope: str,
    proposals: dict,
    result: llm.LLMResult | None = None,
) -> dict:
    as_of = scope_as_of(scope)
    records = records_for_scope(conn, team, scope)
    decisions = {
        record["id"]: record
        for record in records
        if record["kind"] == "decision"
    }
    actions = {
        record["id"]: record
        for record in records
        if record["kind"] == "action"
    }
    disagreements = {
        record["id"]: record
        for record in records
        if record["kind"] == "disagreement"
    }
    drops: list[dict] = []
    mapping: dict[str, tuple[str, str]] = {}

    for topic in proposals.get("topics") or []:
        parts = _topic_parts(topic, decisions)
        if parts is None:
            _drop(drops, "topic", "bad_schema", {"topic": topic})
            continue
        ids, label, key = parts
        for decision_id in ids:
            if decision_id not in decisions:
                _drop(drops, "topic", "unknown_id", topic)
                continue
            if decision_id in mapping:
                _drop(drops, "topic", "duplicate_id", topic)
                continue
            mapping[decision_id] = (key, label or key)

    for decision_id, record in decisions.items():
        if decision_id in mapping:
            continue
        row = conn.execute(
            "SELECT topic, topic_key FROM decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if row:
            mapping[decision_id] = (
                row["topic_key"] or ledger.normalize_topic(row["topic"]),
                row["topic"] or row["topic_key"],
            )

    supersedes: list[tuple[str, str, str]] = []
    supersede_items = (
        proposals.get("supersedes")
        or proposals.get("supersede")
        or proposals.get("decision_supersedes")
        or []
    )
    for edge in supersede_items:
        if not isinstance(edge, dict):
            _drop(drops, "supersede", "bad_schema", {"edge": edge})
            continue
        newer_id = (
            edge.get("newer_id")
            or edge.get("later_id")
            or edge.get("newer")
        )
        older_id = (
            edge.get("older_id")
            or edge.get("earlier_id")
            or edge.get("older")
        )
        newer = decisions.get(newer_id) if isinstance(newer_id, str) else None
        older = decisions.get(older_id) if isinstance(older_id, str) else None
        if newer is None or older is None:
            _drop(drops, "supersede", "unknown_id", edge)
            continue
        if newer["meeting_letter"] <= older["meeting_letter"]:
            _drop(drops, "supersede", "letter_order", edge)
            continue
        quote = str(
            edge.get("quote")
            or edge.get("evidence")
            or newer.get("quote")
            or ""
        ).strip()
        verified = _verify_in_meeting(
            conn,
            team,
            newer["meeting_letter"],
            quote,
            as_of,
        )
        if verified is None:
            _drop(drops, "supersede", "quote", edge)
            continue
        supersedes.append(
            (newer["id"], older["id"], newer["meeting_letter"])
        )

    action_closures: list[tuple[str, str, str]] = []
    closure_items = (
        proposals.get("action_closures")
        or proposals.get("closures")
        or []
    )
    for edge in closure_items:
        if not isinstance(edge, dict):
            _drop(drops, "action_closure", "bad_schema", {"edge": edge})
            continue
        action_id = (
            edge.get("action_id")
            or edge.get("earlier_id")
            or edge.get("action")
        )
        action = actions.get(action_id) if isinstance(action_id, str) else None
        if action is None:
            _drop(drops, "action_closure", "unknown_id", edge)
            continue
        quote = str(edge.get("quote") or edge.get("evidence") or "").strip()
        verified = _find_quote_turn(
            conn,
            team,
            quote,
            as_of,
            min_letter=action["meeting_letter"],
        )
        if verified is None:
            _drop(drops, "action_closure", "quote", edge)
            continue
        action_closures.append(
            (action["id"], verified["turn_id"], verified["letter"])
        )

    resolutions: list[tuple[str, str, str]] = []
    resolution_items = (
        proposals.get("disagreement_resolutions")
        or proposals.get("resolutions")
        or []
    )
    for edge in resolution_items:
        if not isinstance(edge, dict):
            _drop(
                drops,
                "disagreement_resolution",
                "bad_schema",
                {"edge": edge},
            )
            continue
        disagreement_id = (
            edge.get("disagreement_id")
            or edge.get("earlier_id")
            or edge.get("disagreement")
        )
        disagreement = (
            disagreements.get(disagreement_id)
            if isinstance(disagreement_id, str)
            else None
        )
        if disagreement is None:
            _drop(drops, "disagreement_resolution", "unknown_id", edge)
            continue
        quote = str(edge.get("quote") or edge.get("evidence") or "").strip()
        verified = _find_quote_turn(
            conn,
            team,
            quote,
            as_of,
            min_letter=disagreement["meeting_letter"],
        )
        if verified is None:
            _drop(drops, "disagreement_resolution", "quote", edge)
            continue
        resolutions.append(
            (
                disagreement["id"],
                verified["turn_id"],
                verified["letter"],
            )
        )

    provider = result.provider if result else "test"
    model = result.model if result else "test"
    created_at = _now_iso()

    with conn:
        conn.execute(
            "DELETE FROM topic_key_map WHERE team = ? AND scope = ?",
            (team, scope),
        )
        conn.execute(
            "DELETE FROM decision_supersedes WHERE scope = ? AND newer_id IN "
            "(SELECT decision_id FROM decisions WHERE team = ?)",
            (scope, team),
        )
        conn.execute(
            "DELETE FROM action_closed_by WHERE scope = ? AND action_id IN "
            "(SELECT action_id FROM actions WHERE team = ?)",
            (scope, team),
        )
        conn.execute(
            "DELETE FROM disagreement_resolved_by WHERE scope = ? AND "
            "disagreement_id IN "
            "(SELECT disagreement_id FROM disagreements WHERE team = ?)",
            (scope, team),
        )

        for decision_id, (topic_key, topic) in mapping.items():
            conn.execute(
                "INSERT OR REPLACE INTO topic_key_map "
                "(team, scope, decision_id, topic_key, topic) "
                "VALUES (?, ?, ?, ?, ?)",
                (team, scope, decision_id, topic_key, topic),
            )
        for newer_id, older_id, letter in supersedes:
            conn.execute(
                "INSERT OR REPLACE INTO decision_supersedes "
                "(newer_id, older_id, superseding_letter, scope, provider, "
                "model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    newer_id,
                    older_id,
                    letter,
                    scope,
                    provider,
                    model,
                    created_at,
                ),
            )
        for action_id, turn_id, letter in action_closures:
            conn.execute(
                "INSERT OR REPLACE INTO action_closed_by "
                "(action_id, closer_turn_id, closing_letter, scope, provider, "
                "model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    action_id,
                    turn_id,
                    letter,
                    scope,
                    provider,
                    model,
                    created_at,
                ),
            )
        for disagreement_id, turn_id, letter in resolutions:
            conn.execute(
                "INSERT OR REPLACE INTO disagreement_resolved_by "
                "(disagreement_id, resolver_turn_id, resolving_letter, scope, "
                "provider, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    disagreement_id,
                    turn_id,
                    letter,
                    scope,
                    provider,
                    model,
                    created_at,
                ),
            )

    return {
        "records": len(records),
        "topics": len(mapping),
        "supersedes": len(supersedes),
        "action_closures": len(action_closures),
        "disagreement_resolutions": len(resolutions),
        "drops": len(drops),
        "drop_details": drops[:20],
    }


def _boundary(scope: str) -> str:
    if scope == "asof_c":
        return "Scope: asof_c. Only meetings a-c are in scope; never reference d."
    return "Scope: all. Meetings a-d are in scope."


def _decision_prompt(team: str, scope: str, text: str) -> str:
    return (
        f"Team: {team}\n{_boundary(scope)}\n"
        "All decisions are listed. Group decisions into topics only when at "
        "least two decisions address the same subject; omit singleton "
        "decisions because they are handled separately. Never put one decision "
        "in more than one topic. Use only the IDs shown. Return JSON with key "
        "topics.\n\nDecisions:\n"
        f"{text}"
    )


def _supersede_prompt(team: str, scope: str, text: str) -> str:
    return (
        f"Team: {team}\n{_boundary(scope)}\n"
        "All decisions are listed. Create supersede edges only when a newer "
        "decision explicitly replaces an older one; newer_id must have a later "
        "meeting letter than older_id. Do not create an edge for a bare "
        "mention, a restatement, or a same-meeting change. Use only the IDs "
        "shown. Return JSON with key supersedes.\n\nDecisions:\n"
        f"{text}"
    )


def _action_prompt(
    team: str, scope: str, text: str, later_evidence: str = ""
) -> str:
    evidence_block = (
        f"\n\nLater-meeting decisions/figures (copy exact quotes from here when closing an action):\n{later_evidence}"
        if later_evidence
        else ""
    )
    return (
        f"Team: {team}\n{_boundary(scope)}\n"
        "All actions are listed. Create action closures only when a later "
        "meeting record explicitly shows the action was completed or its "
        "results were reported; the quote must be copied verbatim (>=6 words) "
        "from a later meeting than the action. Do not infer from silence. "
        "Use only the action IDs shown. Return JSON with key action_closures."
        f"\n\nActions:\n{text}{evidence_block}"
    )


def _disagreement_prompt(
    team: str, scope: str, text: str, later_evidence: str = ""
) -> str:
    evidence_block = (
        f"\n\nLater-meeting decisions/figures (copy exact quotes from here when resolving a disagreement):\n{later_evidence}"
        if later_evidence
        else ""
    )
    return (
        f"Team: {team}\n{_boundary(scope)}\n"
        "All disagreements are listed. Create disagreement resolutions only "
        "when a later meeting record explicitly resolves the disagreement "
        "(e.g. a later decision settling that exact topic); the quote must be "
        "copied verbatim (>=6 words) from a later meeting than the "
        "disagreement. Do not infer from silence. Use only the disagreement "
        "IDs shown. Return JSON with key disagreement_resolutions."
        f"\n\nDisagreements:\n{text}{evidence_block}"
    )


def run(
    teams: list[str] | None = None,
    scopes: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    conn = store.connect()
    selected = teams or ledger._question_teams()
    if limit is not None:
        selected = selected[:limit]
    engine = llm.LLM(conn=conn)
    started = time.time()
    summary = {
        "teams": len(selected),
        "scopes": 0,
        "calls": 0,
        "records": 0,
        "topics": 0,
        "supersedes": 0,
        "action_closures": 0,
        "disagreement_resolutions": 0,
        "drops": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
        "errors": [],
        "details": [],
    }

    for team in selected:
        team_scopes = scopes or (
            ["all", "asof_c"] if team == "TS3012" else ["all"]
        )
        for scope in team_scopes:
            records = records_for_scope(conn, team, scope)
            if not records:
                counts = apply_proposals(conn, team, scope, {}, None)
                summary["scopes"] += 1
                summary["details"].append(
                    {"team": team, "scope": scope, **counts}
                )
                continue

            decisions = [
                record
                for record in records
                if record["kind"] == "decision"
            ]
            actions = [
                record
                for record in records
                if record["kind"] == "action"
            ]
            disagreements = [
                record
                for record in records
                if record["kind"] == "disagreement"
            ]
            ctx_scope = tools.RunContext(
                team=team, as_of=scope_as_of(scope), mode="answer"
            )
            figures = tools.ledger(conn, ctx_scope, kind="figures")
            combined = {
                "topics": [],
                "supersedes": [],
                "action_closures": [],
                "disagreement_resolutions": [],
            }
            calls: list[tuple[dict, str, list[str]]] = []
            if decisions:
                decision_text = _decision_lines(decisions)
                calls.append(
                    (
                        DECISION_SCHEMA,
                        _decision_prompt(team, scope, decision_text),
                        ["topics"],
                    )
                )
                calls.append(
                    (
                        SUPERSEDE_SCHEMA,
                        _supersede_prompt(team, scope, decision_text),
                        ["supersedes"],
                    )
                )
            if actions:
                min_act_letter = min(r["meeting_letter"] for r in actions)
                later_for_act = _later_evidence_lines(
                    decisions, figures, min_act_letter
                )
                calls.append(
                    (
                        ACTION_SCHEMA,
                        _action_prompt(
                            team, scope, _action_lines(actions), later_for_act
                        ),
                        ["action_closures"],
                    )
                )
            if disagreements:
                min_dis_letter = min(r["meeting_letter"] for r in disagreements)
                later_for_dis = _later_evidence_lines(
                    decisions, figures, min_dis_letter
                )
                calls.append(
                    (
                        DISAGREEMENT_SCHEMA,
                        _disagreement_prompt(
                            team,
                            scope,
                            _disagreement_lines(disagreements),
                            later_for_dis,
                        ),
                        ["disagreement_resolutions"],
                    )
                )

            last_result: llm.LLMResult | None = None
            failed = False

            for schema, prompt, keys in calls:
                try:
                    result = engine.chat(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        response_schema=schema,
                        max_tokens=4096,
                        temperature=0.1,
                        tier="final",
                    )
                except llm.AllProvidersFailed as exc:
                    summary["errors"].append(str(exc)[:1000])
                    failed = True
                    break

                data = llm.parse_json_text(result.text)
                if not isinstance(data, dict):
                    summary["errors"].append(
                        f"{team}/{scope}: parse_error"
                    )
                    failed = True
                    break

                for key in keys:
                    combined[key].extend(data.get(key) or [])
                last_result = result
                summary["calls"] += 1
                usage = result.usage or {}
                summary["prompt_tokens"] += usage.get("prompt_tokens", 0)
                summary["completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                summary["thoughts_tokens"] += usage.get(
                    "thoughts_tokens", 0
                )
                summary["total_tokens"] += usage.get("total_tokens", 0)

            if failed:
                continue

            counts = apply_proposals(
                conn, team, scope, combined, last_result
            )
            summary["scopes"] += 1
            summary["records"] += counts["records"]
            summary["topics"] += counts["topics"]
            summary["supersedes"] += counts["supersedes"]
            summary["action_closures"] += counts["action_closures"]
            summary["disagreement_resolutions"] += counts[
                "disagreement_resolutions"
            ]
            summary["drops"] += counts["drops"]
            summary["details"].append(
                {
                    "team": team,
                    "scope": scope,
                    **counts,
                    "provider": last_result.provider if last_result else None,
                    "model": last_result.model if last_result else None,
                    "fallback": (
                        last_result.fallback_used if last_result else None
                    ),
                }
            )
            print(
                f"{team}/{scope} records={counts['records']} "
                f"topics={counts['topics']} "
                f"supersedes={counts['supersedes']} "
                f"closures={counts['action_closures']} "
                f"resolutions={counts['disagreement_resolutions']} "
                f"drops={counts['drops']}",
                flush=True,
            )

    summary["elapsed_sec"] = round(time.time() - started, 2)
    path = config.METRICS_DIR / "reconcile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"ts": _now_iso(), **summary}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return summary
