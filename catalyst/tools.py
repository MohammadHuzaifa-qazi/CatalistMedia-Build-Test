from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass

from . import config, status, store
from .validate import VerifyResult, normalize, verify_quote

logger = logging.getLogger(__name__)
VECTOR_SEARCH_FAILURES = 0

_TRIM_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "their", "them", "they", "this",
        "to", "was", "were", "what", "when", "where", "which", "who", "why",
        "will", "with",
    }
)


def reset_vector_search_failures() -> None:
    global VECTOR_SEARCH_FAILURES
    VECTOR_SEARCH_FAILURES = 0


def _query_terms(query: str | None) -> set[str]:
    if not query:
        return set()
    raw = re.findall(r"[A-Za-z0-9'_]+", query)
    filtered = [t for t in raw if t.lower() not in _TRIM_STOPWORDS and len(t) > 1]
    tokens = filtered or raw
    terms: set[str] = set()
    for token in tokens:
        terms.add(token.lower())
        if 2 <= len(token) <= 4 and token.isalpha() and token.isupper():
            terms.add(("_".join(token) + "_").lower())
    return terms


def trim_turn_text(
    text: str,
    query: str | None = None,
    max_words: int = config.MAX_DISPLAY_TURN_WORDS,
) -> tuple[str, bool]:
    """Return a verbatim contiguous word-boundary span of at most max_words.

    Never inserts ellipses or modifies characters inside the span, so any
    >=6-word substring copied from the returned text is guaranteed to verify
    against the stored full turn.
    """
    spans = list(re.finditer(r"\S+", text))
    if len(spans) <= max_words:
        return text, False

    terms = _query_terms(query)
    if not terms:
        return text[spans[0].start() : spans[max_words - 1].end()], True

    scores: list[int] = []
    for sp in spans:
        word = sp.group(0).lower()
        clean = re.sub(r"^[^a-z0-9_]+|[^a-z0-9_]+$", "", word)
        score = 0
        if clean in terms:
            score = 3
        else:
            for term in terms:
                if len(term) >= 4 and (term in clean or (len(clean) >= 4 and clean in term)):
                    score = 1
                    break
        scores.append(score)

    window_sum = sum(scores[:max_words])
    best_sum = window_sum
    best_start = 0
    for start in range(1, len(spans) - max_words + 1):
        window_sum += scores[start + max_words - 1] - scores[start - 1]
        if window_sum > best_sum:
            best_sum = window_sum
            best_start = start

    # If the best window has matching words, re-center around the midpoint of
    # the matched words within that window so surrounding context is balanced.
    if best_sum > 0:
        hit_indices = [
            idx
            for idx in range(best_start, best_start + max_words)
            if scores[idx] > 0
        ]
        if hit_indices:
            mid = (hit_indices[0] + hit_indices[-1]) // 2
            centered = max(0, min(mid - max_words // 2, len(spans) - max_words))
            if sum(scores[centered : centered + max_words]) == best_sum:
                best_start = centered

    best_end = best_start + max_words
    return text[spans[best_start].start() : spans[best_end - 1].end()], True


class ScopeError(Exception):
    pass


class VectorSearchError(Exception):
    """Raised when the vector index cannot be used.

    A silent empty result hides a missing dependency or an unbuilt collection
    behind a plausible-looking answer. Callers must see the failure.
    """


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
    query: str | None = None,
    max_turns: int = config.GET_TURNS_MAX_TURNS,
    max_words_per_turn: int = config.MAX_DISPLAY_TURN_WORDS,
    max_total_words: int = config.GET_TURNS_MAX_WORDS,
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
    if max_turns is not None and max_turns > 0:
        sql += " LIMIT ?"
        params.append(max_turns)

    turns: list[dict] = []
    words_used = 0
    for row in conn.execute(sql, params):
        item = dict(row)
        trimmed_text, truncated = trim_turn_text(
            item["text"], query=query, max_words=max_words_per_turn
        )
        shown_words = len(trimmed_text.split())
        if turns and max_total_words and words_used + shown_words > max_total_words:
            break
        item["text"] = trimmed_text
        item["truncated"] = truncated
        item["shown_words"] = shown_words
        turns.append(item)
        words_used += shown_words
    return turns


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


def _build_chroma_where(
    team: str | None,
    as_of: str | None,
    meeting_letter: str | None = None,
) -> dict | None:
    clauses: list[dict] = []
    if team:
        clauses.append({"team": team})
    if meeting_letter:
        clauses.append({"letter": meeting_letter})
    elif as_of:
        allowed = [ch for ch in "abcd" if ch <= as_of]
        if len(allowed) == 1:
            clauses.append({"letter": allowed[0]})
        else:
            clauses.append({"letter": {"$in": allowed}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _vector_search(
    ctx: RunContext,
    query: str,
    team: str | None,
    n: int,
    meeting_letter: str | None = None,
) -> list[dict]:
    global _collection_cache, _embedder_cache, VECTOR_SEARCH_FAILURES
    try:
        if _collection_cache is None:
            _collection_cache = _collection()
        if _collection_cache.count() == 0:
            VECTOR_SEARCH_FAILURES += 1
            logger.error(
                "vector search failed (failures=%d): collection %r is empty",
                VECTOR_SEARCH_FAILURES,
                config.CHROMA_COLLECTION,
            )
            raise VectorSearchError(
                f"vector collection {config.CHROMA_COLLECTION!r} is empty; "
                "build it with `python -m catalyst.pipeline embed`"
            )
        if _embedder_cache is None:
            _embedder_cache = _embedder()
        query_vector = list(_embedder_cache.embed([query], batch_size=1))[0].tolist()
        where = _build_chroma_where(team, ctx.as_of, meeting_letter=meeting_letter)
        result = _collection_cache.query(
            query_embeddings=[query_vector], n_results=n, where=where
        )
        return [{"turn_id": turn_id} for turn_id in result.get("ids", [[]])[0]]
    except VectorSearchError:
        raise
    except Exception as exc:
        VECTOR_SEARCH_FAILURES += 1
        logger.error(
            "vector search failed (failures=%d) for collection %r: %s: %s",
            VECTOR_SEARCH_FAILURES,
            config.CHROMA_COLLECTION,
            type(exc).__name__,
            exc,
        )
        raise VectorSearchError(
            f"vector search failed for collection "
            f"{config.CHROMA_COLLECTION!r}: {type(exc).__name__}: {exc}. "
            "Build the index with `python -m catalyst.pipeline embed`."
        ) from exc


def search(
    conn: sqlite3.Connection,
    ctx: RunContext,
    query: str,
    meeting_letter: str | None = None,
    k: int = 8,
    max_words_per_turn: int = config.MAX_DISPLAY_TURN_WORDS,
) -> list[dict]:
    if meeting_letter and ctx.as_of and meeting_letter > ctx.as_of:
        raise ScopeError(f"meeting_letter {meeting_letter} beyond as_of={ctx.as_of}")

    team = None if ctx.mode == "eval" else ctx.team
    pool = max(k * 3, 20)

    fts_rows = store.fts_search(
        conn,
        query,
        team=team,
        as_of=ctx.as_of,
        meeting_letter=meeting_letter,
        limit=pool,
    )
    vector_rows = _vector_search(
        ctx, query, team, pool, meeting_letter=meeting_letter
    )

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
        trimmed_text, truncated = trim_turn_text(
            hit["text"], query=query, max_words=max_words_per_turn
        )
        hit["text"] = trimmed_text
        hit["truncated"] = truncated
        hit["shown_words"] = len(trimmed_text.split())
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
            "SELECT d.*, COALESCE(m.topic, d.topic) AS mapped_topic, "
            "COALESCE(m.topic_key, d.topic_key) AS mapped_topic_key "
            "FROM decisions d "
            "LEFT JOIN topic_key_map m ON m.decision_id = d.decision_id "
            "AND m.team = d.team AND m.scope = ? "
            "WHERE d.team = ? AND d.meeting_letter <= ? "
            "ORDER BY d.meeting_letter, d.decision_id",
            (scope, ctx.team, as_of),
        )
        for row in rows:
            if not topic_match(row["mapped_topic"], row["topic"], row["decision"]):
                continue
            records.append(
                {
                    "kind": "decision",
                    "id": row["decision_id"],
                    "topic": row["mapped_topic"],
                    "topic_key": row["mapped_topic_key"],
                    "text": row["decision"],
                    "status": status.decision_status(
                        conn,
                        ctx,
                        row["decision_id"],
                        row["team"],
                        row["meeting_letter"],
                        row["mapped_topic_key"],
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
            "description": (
                "Hybrid keyword + vector search over turns. Long turns (>150 words) "
                "are trimmed to the best-matching verbatim ~150-word span; copy "
                "quotes ONLY from the displayed text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "meeting_letter": {
                        "type": "string",
                        "enum": ["a", "b", "c", "d"],
                        "description": (
                            "Optional single meeting letter ('a', 'b', 'c', or 'd'). "
                            "Omit to search all meetings of this team."
                        ),
                    },
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
            "description": (
                "Fetch exact turns from one meeting for context (capped at 25 turns; "
                "turns >150 words are trimmed to a verbatim ~150-word span). Copy "
                "quotes ONLY from the displayed text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {"type": "string"},
                    "seq_start": {"type": "integer"},
                    "seq_end": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "query": {"type": "string"},
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
    args = dict(args or {})
    if name == "search":
        raw_k = args.get("k")
        k = 8 if raw_k is None else max(1, min(int(raw_k), 20))
        raw_letter = args.get("meeting_letter")
        meeting_letter: str | None = None
        if raw_letter is not None and str(raw_letter).strip():
            cand = str(raw_letter).strip()
            if cand.lower() in {"a", "b", "c", "d"}:
                meeting_letter = cand.lower()
            elif (
                ctx.team
                and len(cand) == len(ctx.team) + 1
                and cand[:-1].upper() == ctx.team.upper()
                and cand[-1].lower() in {"a", "b", "c", "d"}
            ):
                meeting_letter = cand[-1].lower()
            else:
                raise ValueError(
                    f"invalid meeting_letter {cand!r}: must be a single letter "
                    "'a', 'b', 'c', or 'd', or omitted to search all meetings"
                )
        return search(
            conn,
            ctx,
            query=str(args["query"]),
            meeting_letter=meeting_letter,
            k=k,
        )
    if name == "get_turns":
        return get_turns(
            conn,
            ctx,
            meeting_id=str(args["meeting_id"]),
            seq_start=args.get("seq_start"),
            seq_end=args.get("seq_end"),
            speaker=args.get("speaker") or None,
            query=args.get("query") or None,
        )
    if name == "ledger":
        return ledger(
            conn,
            ctx,
            topic=args.get("topic") or None,
            kind=str(args.get("kind") or "all"),
        )
    if name == "verify":
        return verify(
            conn,
            ctx,
            meeting_id=str(args["meeting_id"]),
            quote=str(args["quote"]),
        )
    raise ValueError(f"unknown tool: {name}")
