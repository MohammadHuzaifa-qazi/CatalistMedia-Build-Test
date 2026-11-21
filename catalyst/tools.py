from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import config, status, store
from .validate import VerifyResult, normalize, verify_quote


class ScopeError(Exception):
    pass


@dataclass
class RunContext:
    team: str | None
    as_of: str | None = None
    mode: str = "answer"

    def __post_init__(self) -> None:
        if self.mode not in {"answer", "brief", "eval"}:
            raise ValueError(f"bad mode: {self.mode}")
        if self.mode != "eval" and self.team is None:
            raise ValueError("team required outside eval mode")


def _meeting(conn: sqlite3.Connection, meeting_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT team, letter FROM meetings WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()
    if row is None:
        raise ScopeError(f"unknown meeting: {meeting_id}")
    return row


def _check_scope(conn: sqlite3.Connection, ctx: RunContext, meeting_id: str) -> sqlite3.Row:
    meeting = _meeting(conn, meeting_id)
    if ctx.team is not None and meeting["team"] != ctx.team:
        raise ScopeError(f"{meeting_id} outside team {ctx.team}")
    if ctx.as_of and meeting["letter"] > ctx.as_of:
        raise ScopeError(f"{meeting_id} beyond as_of={ctx.as_of}")
    return meeting


def get_turns(
    conn: sqlite3.Connection,
    ctx: RunContext,
    meeting_id: str,
    seq_start: int | None = None,
    seq_end: int | None = None,
    speaker: str | None = None,
) -> list[dict]:
    _check_scope(conn, ctx, meeting_id)

    sql = "SELECT * FROM turns WHERE meeting_id = ?"
    params: list = [meeting_id]
    if seq_start is not None:
        sql += " AND seq >= ?"
        params.append(seq_start)
    if seq_end is not None:
        sql += " AND seq <= ?"
        params.append(seq_end)
    if speaker:
        sql += " AND speaker = ?"
        params.append(speaker)
    sql += " ORDER BY seq"
    return [dict(row) for row in conn.execute(sql, params)]


def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.CHROMA_COLLECTION)


def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=config.EMBED_MODEL, threads=config.EMBED_THREADS
    )


_collection_cache = None
_embedder_cache = None


def _vector_search(ctx: RunContext, query: str, team: str | None, n: int) -> list[dict]:
    global _collection_cache, _embedder_cache
    try:
        if _collection_cache is None:
            _collection_cache = _collection()
        if _collection_cache.count() == 0:
            return []
        if _embedder_cache is None:
            _embedder_cache = _embedder()
        query_vector = list(_embedder_cache.embed([query], batch_size=1))[0].tolist()
        where = {"team": team} if team else None
        result = _collection_cache.query(
            query_embeddings=[query_vector], n_results=n, where=where
        )
        return [{"turn_id": turn_id} for turn_id in result.get("ids", [[]])[0]]
    except Exception:
        return []


def search(
    conn: sqlite3.Connection,
    ctx: RunContext,
    query: str,
    meeting_letter: str | None = None,
    k: int = 8,
) -> list[dict]:
    if meeting_letter and ctx.as_of and meeting_letter > ctx.as_of:
        raise ScopeError(f"meeting_letter {meeting_letter} beyond as_of={ctx.as_of}")

    team = None if ctx.mode == "eval" else ctx.team
    pool = max(k * 3, 20)

    fts_rows = store.fts_search(conn, query, team=team, limit=pool)
    vector_rows = _vector_search(ctx, query, team, pool)

    scores: dict[str, float] = {}
    for rank, row in enumerate(fts_rows):
        scores[row["turn_id"]] = scores.get(row["turn_id"], 0.0) + 1.0 / (60 + rank + 1)
    for rank, row in enumerate(vector_rows):
        scores[row["turn_id"]] = scores.get(row["turn_id"], 0.0) + 1.0 / (60 + rank + 1)

    hits: list[dict] = []
    for turn_id, score in sorted(scores.items(), key=lambda item: -item[1]):
        turn = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if turn is None:
            continue
        if team and turn["team"] != team:
            continue
        if meeting_letter and turn["letter"] != meeting_letter:
            continue
        if ctx.as_of and turn["letter"] > ctx.as_of:
            continue
        hit = dict(turn)
        hit["score"] = round(score, 6)
        hits.append(hit)
        if len(hits) >= k:
            break
    return hits


def ledger(
    conn: sqlite3.Connection,
    ctx: RunContext,
    topic: str | None = None,
    kind: str = "all",
) -> list[dict]:
    if ctx.team is None:
        raise ScopeError("ledger requires a team")

    as_of = ctx.as_of or "z"
    scope = "asof_c" if ctx.as_of == "c" else "all"
    records: list[dict] = []

    def topic_match(*values: str | None) -> bool:
        if not topic:
            return True
        needle = topic.lower()
        return any(value and needle in value.lower() for value in values)

    if kind in {"all", "decisions"}:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE team = ? AND meeting_letter <= ? "
            "ORDER BY meeting_letter, decision_id",
            (ctx.team, as_of),
        )
        for row in rows:
            if not topic_match(row["topic"], row["decision"]):
                continue
            records.append(
                {
                    "kind": "decision",
                    "id": row["decision_id"],
                    "topic": row["topic"],
                    "text": row["decision"],
                    "status": status.decision_status(
                        conn,
                        ctx,
                        row["decision_id"],
                        row["team"],
                        row["meeting_letter"],
                        row["topic_key"],
                    ),
                    "meeting_id": row["meeting_id"],
                    "meeting_letter": row["meeting_letter"],
                    "turn_id": row["turn_id"],
                    "speaker": row["speaker"],
                    "quote": row["quote"],
                    "scope": scope,
                }
            )

    if kind in {"all", "actions"}:
        rows = conn.execute(
            "SELECT * FROM actions WHERE team = ? AND meeting_letter <= ? "
            "ORDER BY meeting_letter, action_id",
            (ctx.team, as_of),
        )
        for row in rows:
            if not topic_match(row["task"]):
                continue
            records.append(
                {
                    "kind": "action",
                    "id": row["action_id"],
                    "topic": None,
                    "text": row["task"],
                    "status": status.action_status(
                        conn, ctx, row["action_id"], row["meeting_letter"]
                    ),
                    "meeting_id": row["meeting_id"],
                    "meeting_letter": row["meeting_letter"],
                    "turn_id": row["turn_id"],
                    "speaker": row["owner"],
                    "quote": row["quote"],
                    "scope": scope,
                }
            )

    if kind in {"all", "disagreements"}:
        rows = conn.execute(
            "SELECT * FROM disagreements WHERE team = ? AND meeting_letter <= ? "
            "ORDER BY meeting_letter, disagreement_id",
            (ctx.team, as_of),
        )
        for row in rows:
            if not topic_match(row["topic"], row["position"]):
                continue
            records.append(
                {
                    "kind": "disagreement",
                    "id": row["disagreement_id"],
                    "topic": row["topic"],
                    "text": row["position"],
                    "status": status.disagreement_status(
                        conn, ctx, row["disagreement_id"], row["meeting_letter"]
                    ),
                    "meeting_id": row["meeting_id"],
                    "meeting_letter": row["meeting_letter"],
                    "turn_id": row["turn_id"],
                    "speaker": row["who"],
                    "quote": row["quote"],
                    "scope": scope,
                }
            )

    if kind in {"all", "figures"}:
        rows = conn.execute(
            "SELECT * FROM figures WHERE team = ? AND meeting_letter <= ? "
            "ORDER BY meeting_letter, figure_id",
            (ctx.team, as_of),
        )
        for row in rows:
            if not topic_match(row["component"]):
                continue
            records.append(
                {
                    "kind": "figure",
                    "id": row["figure_id"],
                    "topic": row["component"],
                    "text": f'{row["value"]} {row["unit"] or ""}'.strip(),
                    "status": "current",
                    "meeting_id": row["meeting_id"],
                    "meeting_letter": row["meeting_letter"],
                    "turn_id": row["turn_id"],
                    "speaker": None,
                    "quote": row["quote"],
                    "scope": scope,
                }
            )

    return records


def verify(
    conn: sqlite3.Connection, ctx: RunContext, meeting_id: str, quote: str
) -> VerifyResult:
    return verify_quote(conn, ctx, meeting_id, quote)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Hybrid keyword + vector search over turns. Returns exact turn text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "meeting_letter": {"type": "string"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_turns",
            "description": "Fetch exact turns from one meeting for context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                    "seq_start": {"type": "integer"},
                    "seq_end": {"type": "integer"},
                    "speaker": {"type": "string"},
                },
                "required": ["meeting_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ledger",
            "description": "Query decisions, actions, disagreements, and figures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["all", "decisions", "actions", "disagreements", "figures"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": "Deterministically verify an exact quote inside one turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["meeting_id", "quote"],
            },
        },
    },
]


def execute_tool(
    conn: sqlite3.Connection, ctx: RunContext, name: str, args: dict
):
    if name == "search":
        return search(conn, ctx, **args)
    if name == "get_turns":
        return get_turns(conn, ctx, **args)
    if name == "ledger":
        return ledger(conn, ctx, **args)
    if name == "verify":
        return verify(conn, ctx, **args)
    raise ValueError(f"unknown tool: {name}")
