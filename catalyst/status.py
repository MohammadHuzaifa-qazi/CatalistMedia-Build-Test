from __future__ import annotations

import sqlite3


def _scope(ctx) -> str:
    return "asof_c" if ctx.as_of == "c" else "all"


def _as_of(ctx) -> str:
    return ctx.as_of or "z"


def decision_status(
    conn: sqlite3.Connection,
    ctx,
    decision_id: str,
    team: str,
    meeting_letter: str,
    topic_key: str,
) -> str:
    scope = _scope(ctx)
    as_of = _as_of(ctx)

    edge = conn.execute(
        "SELECT 1 FROM decision_supersedes "
        "WHERE older_id = ? AND scope = ? AND superseding_letter <= ? LIMIT 1",
        (decision_id, scope, as_of),
    ).fetchone()
    if edge:
        return "superseded"

    peers = conn.execute(
        "SELECT d.decision_id, d.meeting_letter "
        "FROM decisions d "
        "JOIN topic_key_map m ON m.decision_id = d.decision_id "
        "AND m.team = d.team AND m.scope = ? "
        "WHERE d.team = ? AND d.meeting_letter <= ? AND m.topic_key = ?",
        (scope, team, as_of, topic_key),
    ).fetchall()
    if not peers:
        return "unknown"

    top_letter = max(row["meeting_letter"] for row in peers)
    top = [row for row in peers if row["meeting_letter"] == top_letter]
    if len(top) > 1:
        return "disputed"
    return "current" if top[0]["decision_id"] == decision_id else "superseded"


def action_status(
    conn: sqlite3.Connection, ctx, action_id: str, meeting_letter: str
) -> str:
    scope = _scope(ctx)
    as_of = _as_of(ctx)
    edge = conn.execute(
        "SELECT 1 FROM action_closed_by "
        "WHERE action_id = ? AND scope = ? AND closing_letter <= ? LIMIT 1",
        (action_id, scope, as_of),
    ).fetchone()
    if edge:
        return "done"
    return "open" if meeting_letter <= as_of else "out_of_scope"


def disagreement_status(
    conn: sqlite3.Connection, ctx, disagreement_id: str, meeting_letter: str
) -> str:
    scope = _scope(ctx)
    as_of = _as_of(ctx)
    edge = conn.execute(
        "SELECT 1 FROM disagreement_resolved_by "
        "WHERE disagreement_id = ? AND scope = ? AND resolving_letter <= ? LIMIT 1",
        (disagreement_id, scope, as_of),
    ).fetchone()
    if edge:
        return "resolved"
    return "open" if meeting_letter <= as_of else "out_of_scope"
