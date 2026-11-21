from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from . import config

_ABBR = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|No|Fig|approx|dept|est|min|max|"
    r"a\.m|p\.m|e\.g|i\.e|U\.S|U\.K)\.",
    re.IGNORECASE,
)

CITE = re.compile(r'\[([A-Z]{2}\d{4}[a-d]): "([^"]+)"\]')
PREDICTIVE = re.compile(
    r"\b(next meeting|expected to|likely|going to|"
    r"will (?:be|decide|discuss|finali[sz]e))\b",
    re.IGNORECASE,
)


@dataclass
class VerifyResult:
    ok: bool
    turn_id: str | None
    reason: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def count_sentences(text: str) -> int:
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    protected = _ABBR.sub(lambda m: m.group(0)[:-1] + "<DOT>", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return sum(1 for part in parts if re.search(r"[A-Za-z0-9]", part))


def team_meetings(conn: sqlite3.Connection, team: str) -> set[str]:
    rows = conn.execute("SELECT meeting_id FROM meetings WHERE team = ?", (team,))
    return {row["meeting_id"] for row in rows}


def verify_quote(
    conn: sqlite3.Connection, ctx, meeting_id: str | None, quote: str
) -> VerifyResult:
    if not meeting_id:
        return VerifyResult(False, None, "bad_meeting")

    meeting = conn.execute(
        "SELECT team, letter FROM meetings WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()
    if meeting is None:
        return VerifyResult(False, None, "bad_meeting")
    if ctx.team is not None and meeting["team"] != ctx.team:
        return VerifyResult(False, None, "out_of_scope")
    if ctx.as_of and meeting["letter"] > ctx.as_of:
        return VerifyResult(False, None, "out_of_scope")

    normalized_quote = normalize(quote)
    if len(normalized_quote.split()) < config.MIN_QUOTE_WORDS:
        return VerifyResult(False, None, "too_short")

    pattern = r"(?<!\w)" + re.escape(normalized_quote) + r"(?!\w)"
    rows = conn.execute(
        "SELECT turn_id, text FROM turns WHERE meeting_id = ? ORDER BY seq",
        (meeting_id,),
    )
    for row in rows:
        if re.search(pattern, normalize(row["text"])):
            return VerifyResult(True, row["turn_id"], "ok")
    return VerifyResult(False, None, "not_found")


def validate_answer(
    conn: sqlite3.Connection,
    ctx,
    answer: dict,
    retrieved: set[str],
    question: str | None = None,
) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    text = answer.get("text", "")
    citations = answer.get("citations", [])

    if count_sentences(text) not in (1, 2):
        errors.append("length")
    if not (1 <= len(citations) <= 3):
        errors.append("citation_count")

    letters: set[str] = set()
    allowed_meetings = team_meetings(conn, ctx.team) if ctx.team else None

    for citation in citations:
        meeting_id = citation.get("meeting_id")
        quote = citation.get("quote", "")

        if allowed_meetings is not None and meeting_id not in allowed_meetings:
            errors.append("wrong_team")
        if ctx.as_of and meeting_id and meeting_id[-1] > ctx.as_of:
            errors.append("as_of_violation")

        result = verify_quote(conn, ctx, meeting_id, quote)
        if not result.ok:
            errors.append(f"citation:{result.reason}")
            continue

        citation["turn_id"] = result.turn_id
        if result.turn_id not in retrieved:
            errors.append("citation_not_retrieved")
        if meeting_id:
            letters.add(meeting_id[-1])

    if len(letters) < 2:
        warnings.append("single_meeting")

    if question and question.strip().lower().startswith("who"):
        if not re.search(r"\b(?:Speaker\s+)?[A-D]\b", text):
            warnings.append("who_missing_speaker")

    return (len(errors) == 0, errors, warnings)


def validate_brief(conn: sqlite3.Connection, ctx, brief: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    raw = brief.get("raw_text", "")

    bullets = brief.get("bullets")
    if bullets is None:
        bullets = [{"text": line} for line in raw.splitlines() if line.strip()]

    for bullet in bullets:
        text = bullet.get("text", "")
        refs = CITE.findall(text)
        if not refs:
            errors.append("uncited_bullet")
            continue

        verified = 0
        for meeting_id, quote in refs:
            if meeting_id.startswith("TS3012d"):
                errors.append("asof_violation")
                continue
            result = verify_quote(conn, ctx, meeting_id, quote)
            if result.ok:
                verified += 1
                bullet.setdefault("turn_ids", []).append(result.turn_id)
            else:
                errors.append(f"citation:{result.reason}")
        if verified < 1:
            errors.append("no_verified_citation")

    if re.search(r"\bTS3012d\b", raw):
        errors.append("asof_violation")

    prose = CITE.sub("", raw)
    for sentence in re.split(r"(?<=[.!?])\s+", prose):
        if re.search(r"\b(meeting|final|next)\b", sentence, re.IGNORECASE) and PREDICTIVE.search(
            sentence
        ):
            errors.append("predicts_final_meeting")
            break

    return (len(errors) == 0, errors)
