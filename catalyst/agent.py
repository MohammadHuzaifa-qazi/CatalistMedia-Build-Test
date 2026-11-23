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

SYSTEM_PROMPT = """You are a grounded meeting-memory research agent answering questions about AMI design-team transcripts.

TOOLS:
- `search(query, meeting_letter?, k?)`: hybrid keyword + semantic search over turns. `meeting_letter` MUST be a single letter ('a', 'b', 'c', or 'd') or omitted to search all meetings — NEVER pass a team ID.
- `get_turns(meeting_id, seq_start?, seq_end?, speaker?, query?)`: read consecutive turns around a hit.
- `ledger(topic?, kind?)`: ONLY if a precomputed ledger is available for this team.
- `verify(meeting_id, quote)`: deterministically verify an exact quote in one turn.

RESEARCH DISCIPLINE:
1. Initial hybrid search hits across all meetings of this team are already loaded in your Evidence Notebook. Usually you only need 1-3 targeted `get_turns()` or `search()` calls before answering — do not over-search.
2. Every question requires evidence from at least TWO meetings of this team (e.g., an earlier leaning/plan/commitment and a later switch/report/final decision). Check both early (a/b) and late (c/d) meetings.
3. ALWAYS READ SURROUNDING AND LATER TURNS BEFORE CONCLUDING: AMI transcripts are spoken multi-party meetings full of short interruptions and late reversals. Teams often quote a provisional total or defend a feature mid-discussion and then revise or drop it 5-25 turns later during budget/costing (`twelve point five` Euro limit) or evaluation. Call `get_turns(meeting_id, seq_start=seq-4, seq_end=seq+18)` around key hits in meetings c/d so you capture the TRUE final decision, final total, and exact reason.
4. Check EVERY sub-part of the question before concluding (e.g., initial leaning vs final choice, exact reason, reported figure, whether a feature stayed in the final total). Never leave a sub-part vague (name the exact material/feature/figure).
5. Numbers are spoken as words ("twenty five Euro", "two to three Euro", "twelve point five", "eight point two"). Prefer the final total/figure explicitly confirmed by the team; never invent numbers.
6. For speaker questions, ALWAYS verify who actually SPOKE the turn (`[meeting#seq | Speaker X]`) and do NOT confuse a speaker *addressed* in a quote with the person speaking. Give the speaker letter (Speaker A, B, C, or D), plus the role ONLY if explicitly verified in the notebook.
7. Long turns (>150 words) are trimmed to a verbatim span. Copy quotes ONLY from the exact displayed text in the Evidence Notebook. Never add "..." or merge turns.
8. Do NOT repeat identical tool calls. Once you have verified evidence from >=2 meetings answering all sub-parts, STOP calling tools immediately and return your final JSON answer.

FINAL OUTPUT FORMAT (STRICT — NO PROSE OUTSIDE JSON):
When done researching, stop calling tools and output ONLY a raw JSON object:
{"answer": "1 to 2 plain sentences directly answering every part of the question.", "citations": [{"meeting_id": "XXXXa", "quote": "exact verbatim quote of at least six words from one displayed turn"}]}

STRICT VALIDATION RULES:
- `answer`: strictly 1 or 2 plain sentences (plain ASCII characters only; no bullets, no markdown headings).
- `citations`: 1 to 3 items spanning at least 2 distinct meetings whenever possible.
- Every `quote`: >= 6 words, copied EXACTLY (preserving punctuation) from ONE turn displayed in the Evidence Notebook.
"""


SYNTHESIS_SYSTEM_PROMPT = """You are a grounded meeting-memory assistant answering a question from an Evidence Notebook of AMI design-team transcript turns.

RULES:
1. Answer EVERY sub-part of the question specifically and accurately using ONLY the turns in the Evidence Notebook (name exact materials, features, figures, meetings, and reasons — never answer vaguely).
2. CHRONOLOGY MATTERS: Read turns in chronological order (meeting a -> b -> c -> d, and lower seq -> higher seq). If the team discusses an early prototype or provisional sum and then revises/drops a feature or updates the total later in the same meeting (e.g., during the 12.50 Euro budget check), ALWAYS report the FINAL outcome.
3. Numbers are spoken as words ("twenty five Euro", "two to three Euro", "twelve point five", "eight point two"). Prefer the final explicitly confirmed figure; never invent numbers.
4. For speaker questions, ALWAYS verify who actually SPOKE the relevant turn (`Speaker A`, `B`, `C`, or `D`) rather than who was addressed. Include a role ONLY if explicitly confirmed in the Evidence Notebook.
5. Copy quotes VERBATIM (preserving punctuation) from ONE displayed turn in the Evidence Notebook. Never add "..." or merge two turns.
6. Every citation MUST directly support a specific claim in your answer — never cite an off-topic quote just to include a second meeting.
7. Do NOT call any tools. Output ONLY a raw JSON object with this exact schema:
{"answer": "1 to 2 plain sentences directly answering every part of the question.", "citations": [{"meeting_id": "XXXXa", "quote": "exact verbatim quote of at least six words from one displayed turn"}]}

STRICT CONSTRAINTS:
- `answer`: strictly 1 or 2 plain sentences (plain ASCII punctuation only — use `-` and `'`, never Unicode hyphens or curly quotes; no bullets or markdown).
- `citations`: 1 to 3 items spanning at least 2 distinct meetings whenever supported by the notebook.
- `meeting_id`: exact 7-character meeting ID (e.g. "IS1008d"), NOT a turn_id with "#".
- Every `quote`: >= 6 words, copied EXACTLY from ONE displayed turn in the Evidence Notebook (plain ASCII characters only).
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
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
    "required": ["answer", "citations"],
}


_UNICODE_TRANS_SPACE = str.maketrans(
    {
        "\u2010": " ",
        "\u2011": " ",
        "\u2012": " ",
        "\u2013": " ",
        "\u2014": " ",
        "\u2015": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
)
_UNICODE_TRANS_HYPHEN = str.maketrans(
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


def _clean_unicode_quotes(
    conn: sqlite3.Connection, ctx: tools.RunContext, draft: dict | None
) -> None:
    """Fix LLM Unicode typography (e.g. U+2011 non-breaking hyphen, curly quotes)
    in answer prose and in citations ONLY when the pure-ASCII equivalent verifies
    verbatim in the turn. AMI transcripts have 0 non-ASCII characters.
    """
    if not isinstance(draft, dict):
        return
    for key in ("answer", "text"):
        val = draft.get(key)
        if isinstance(val, str) and not val.isascii():
            draft[key] = val.translate(_UNICODE_TRANS_HYPHEN)
    for cite in draft.get("citations") or []:
        if not isinstance(cite, dict):
            continue
        mid = cite.get("meeting_id")
        quote = cite.get("quote")
        if not mid or not isinstance(quote, str) or quote.isascii():
            continue
        if validate.verify_quote(conn, ctx, mid, quote).ok:
            continue
        for table in (_UNICODE_TRANS_SPACE, _UNICODE_TRANS_HYPHEN):
            cand = quote.translate(table)
            if cand != quote and validate.verify_quote(conn, ctx, mid, cand).ok:
                cite["quote"] = cand
                break


_ROLE_SELF_RE = re.compile(
    r"\b(?:i\s+am|i'm|i'll\s+be|i\s+will\s+be|my\s+(?:role|function)\s+is)\s+"
    r"(?:the|your|a|an|supposed\s+to\s+be|the\s+pr\s+)?(?:(?!not\b)[a-z']+\s+){0,4}?"
    r"(project\s+(?:manager|leader)|industrial\s+designer|user\s+interface(?:\s+designer)?|marketing(?:\s+expert|\s+manager|\s+director|\s+person)?)\b",
    re.IGNORECASE,
)


def _extract_explicit_roles(conn: sqlite3.Connection, ctx: tools.RunContext) -> dict[str, str]:
    """Deterministically extract self-stated speaker roles from early meeting turns."""
    if not ctx.team:
        return {}
    roles: dict[str, str] = {}
    for letter in ("a", "b"):
        if ctx.as_of and letter > ctx.as_of:
            continue
        mid = f"{ctx.team}{letter}"
        rows = conn.execute(
            "SELECT speaker, text FROM turns WHERE meeting_id = ? AND seq <= 45 ORDER BY seq",
            (mid,),
        ).fetchall()
        for row in rows:
            spk = row["speaker"].replace("Speaker ", "").strip()
            if spk in roles:
                continue
            m = _ROLE_SELF_RE.search(row["text"])
            if m:
                roles[spk] = " ".join(m.group(1).split()).title()
        if len(roles) >= 4:
            break
    return roles


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _has_ledger(conn: sqlite3.Connection, team: str | None) -> bool:
    if not team:
        return False
    for table in ("decisions", "actions", "disagreements", "figures"):
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE team = ? LIMIT 1", (team,)
        ).fetchone()
        if row is not None:
            return True
    return False


def _meetings_in_scope(conn: sqlite3.Connection, ctx: tools.RunContext) -> list[str]:
    sql = "SELECT meeting_id FROM meetings WHERE 1=1"
    params: list[Any] = []
    if ctx.team is not None:
        sql += " AND team = ?"
        params.append(ctx.team)
    if ctx.as_of:
        sql += " AND letter <= ?"
        params.append(ctx.as_of)
    sql += " ORDER BY meeting_id"
    return [row["meeting_id"] for row in conn.execute(sql, params)]


def parse_final(text: str) -> dict | None:
    data = llm.parse_json_text(text or "")
    if not isinstance(data, dict):
        return None
    raw_answer = data.get("answer")
    if raw_answer is None:
        raw_answer = data.get("text")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        return None
    raw_citations = data.get("citations")
    if not isinstance(raw_citations, list):
        return None
    citations: list[dict] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        meeting_id = str(item.get("meeting_id") or "").strip()
        if "#" in meeting_id:
            meeting_id = meeting_id.split("#", 1)[0].strip()
        quote = str(item.get("quote") or "").strip()
        if meeting_id and quote:
            entry: dict[str, Any] = {"meeting_id": meeting_id, "quote": quote}
            if item.get("turn_id"):
                entry["turn_id"] = str(item["turn_id"]).strip()
            citations.append(entry)
    answer_text = " ".join(raw_answer.strip().split())
    return {
        "text": answer_text,
        "answer": answer_text,
        "citations": citations,
    }


class EvidenceNotebook:
    """Stateless per-step evidence store bounded well under the 4K-token budget."""

    def __init__(self, conn: sqlite3.Connection, ctx: tools.RunContext) -> None:
        self.ctx = ctx
        self.meetings = _meetings_in_scope(conn, ctx)
        self.has_ledger = _has_ledger(conn, ctx.team)
        self.roles = _extract_explicit_roles(conn, ctx)
        self.call_log: list[str] = []
        self.seen_calls: set[str] = set()
        self.turns: dict[str, dict] = {}
        self.turn_order: list[str] = []
        self.ledger_items: list[dict] = []
        self.verifications: list[str] = []

    def add_call(self, summary: str) -> None:
        self.call_log.append(summary)

    def add_turns(self, items: list[dict], *, explicit: bool = False) -> list[str]:
        added: list[str] = []
        for item in items:
            turn_id = item.get("turn_id")
            if not turn_id:
                continue
            existing = self.turns.get(turn_id)
            if existing is None:
                entry = dict(item)
                if explicit:
                    entry["_explicit"] = True
                self.turns[turn_id] = entry
                self.turn_order.append(turn_id)
            else:
                new_score = item.get("score")
                old_score = existing.get("score")
                updated = dict(existing)
                updated.update(item)
                if explicit:
                    updated["_explicit"] = True
                if old_score is not None and (new_score is None or old_score > new_score):
                    updated["score"] = old_score
                self.turns[turn_id] = updated
            added.append(turn_id)
        return added

    def add_ledger(self, items: list[dict]) -> list[str]:
        seen = {(r.get("kind"), r.get("id")) for r in self.ledger_items}
        added: list[str] = []
        for item in items:
            key = (item.get("kind"), item.get("id"))
            if key not in seen:
                seen.add(key)
                self.ledger_items.append(dict(item))
            if item.get("turn_id"):
                added.append(item["turn_id"])
        return added

    def add_verification(self, meeting_id: str, quote: str, res: validate.VerifyResult) -> None:
        short_q = quote[:70] + ("..." if len(quote) > 70 else "")
        self.verifications.append(
            f'verify({meeting_id}, "{short_q}") -> ok={res.ok} reason={res.reason} turn_id={res.turn_id}'
        )

    def _select_turns_for_budget(self, max_chars: int) -> list[dict]:
        if not self.turns:
            return []
        # Rank turns so each meeting gets representation and recent/high-score
        # turns survive trimming, then display chronologically.
        recency = {tid: idx for idx, tid in enumerate(self.turn_order)}
        by_meeting: dict[str, list[dict]] = {}
        for item in self.turns.values():
            by_meeting.setdefault(item["meeting_id"], []).append(item)

        def priority(item: dict) -> tuple[int, float, int]:
            return (
                1 if item.get("_explicit") else 0,
                float(item.get("score") or 0.0),
                recency.get(item["turn_id"], 0),
            )

        ordered: list[dict] = []
        buckets = [
            sorted(items, key=priority, reverse=True)
            for _, items in sorted(by_meeting.items())
        ]
        # Round-robin across meetings so no single meeting crowds out others.
        idx = 0
        while True:
            progressed = False
            for bucket in buckets:
                if idx < len(bucket):
                    ordered.append(bucket[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1

        chosen: list[dict] = []
        used = 0
        for item in ordered:
            line = self._format_turn(item)
            if chosen and used + len(line) + 1 > max_chars:
                continue
            chosen.append(item)
            used += len(line) + 1
            if used >= max_chars:
                break

        chosen.sort(key=lambda t: (t.get("meeting_id", ""), int(t.get("seq", 0))))
        return chosen

    @staticmethod
    def _format_turn(item: dict) -> str:
        trunc = " [trimmed]" if item.get("truncated") else ""
        return (
            f'[{item["turn_id"]} | seq={item.get("seq")} | '
            f'{item.get("speaker")}{trunc}]: {item.get("text", "")}'
        )

    def render(
        self, max_chars: int = 13500, *, include_call_log: bool = True
    ) -> tuple[str, list[str]]:
        lines: list[str] = []
        displayed_ids: list[str] = []
        team_label = self.ctx.team or "ALL (eval)"
        as_of_label = f", as_of<={self.ctx.as_of}" if self.ctx.as_of else ""
        lines.append(
            f"Team: {team_label}{as_of_label} | Meetings in scope: {', '.join(self.meetings)}"
        )
        if self.roles:
            role_str = ", ".join(f"Speaker {k} = {v}" for k, v in sorted(self.roles.items()))
            lines.append(f"Verified self-stated roles: {role_str}")
        if include_call_log:
            if not self.has_ledger:
                lines.append("Ledger: none for this team (use search and get_turns).")
            else:
                lines.append("Ledger: available for this team (decisions/actions/disagreements/figures).")

            if self.call_log:
                lines.append("Prior tool calls:")
                for entry in self.call_log[-10:]:
                    lines.append(f"  - {entry}")

        if self.turns:
            counts: dict[str, int] = {}
            for item in self.turns.values():
                mid = item.get("meeting_id", "?")
                counts[mid] = counts.get(mid, 0) + 1
            summary = ", ".join(f"{m}({c})" for m, c in sorted(counts.items()))
            lines.append(f"Retrieved turns by meeting: {summary}")

        if self.verifications:
            lines.append("Recent verifications:")
            for v in self.verifications[-6:]:
                lines.append(f"  - {v}")

        header = "\n".join(lines)
        remaining = max(1200, max_chars - len(header) - 200)

        parts = [header]

        if self.ledger_items:
            ledger_budget = min(2500, remaining // 3)
            ledger_lines = ["Ledger records:"]
            used = 0
            for rec in self.ledger_items:
                line = (
                    f'  [{rec.get("kind")}:{rec.get("status")} | {rec.get("turn_id")} | '
                    f'{rec.get("speaker")}] {rec.get("text")} | quote="{rec.get("quote")}"'
                )
                if used + len(line) + 1 > ledger_budget:
                    break
                ledger_lines.append(line)
                used += len(line) + 1
                if rec.get("turn_id"):
                    displayed_ids.append(rec["turn_id"])
            if len(ledger_lines) > 1:
                parts.append("\n".join(ledger_lines))
                remaining -= used

        chosen = self._select_turns_for_budget(remaining)
        if chosen:
            turn_lines = ["Evidence turns (copy quotes verbatim from displayed text):"]
            for item in chosen:
                turn_lines.append(self._format_turn(item))
                if item.get("turn_id"):
                    displayed_ids.append(item["turn_id"])
            parts.append("\n".join(turn_lines))
        else:
            parts.append("Evidence turns: (none retrieved yet)")

        return "\n\n".join(parts), displayed_ids


def _format_tool_result(
    name: str, args: dict, result: Any, max_chars: int = 4500
) -> tuple[str, list[str]]:
    """Format one tool result compactly and return (text, displayed_turn_ids)."""
    displayed_ids: list[str] = []
    if isinstance(result, str):
        return f"Tool {name}({json.dumps(args, ensure_ascii=False)}): {result}"[:max_chars], []

    if name == "verify" and isinstance(result, validate.VerifyResult):
        text = (
            f"Tool verify({args.get('meeting_id')}): "
            f"ok={result.ok}, reason={result.reason}, turn_id={result.turn_id}"
        )
        return text[:max_chars], []

    if name in {"search", "get_turns"} and isinstance(result, list):
        header = f"Tool {name}({json.dumps(args, ensure_ascii=False)}) -> {len(result)} items:"
        lines = [header]
        used = len(header)
        for item in result:
            line = EvidenceNotebook._format_turn(item)
            if len(lines) > 1 and used + len(line) + 1 > max_chars:
                lines.append(f"... ({len(result) - len(displayed_ids)} more omitted for budget)")
                break
            lines.append(line)
            used += len(line) + 1
            if item.get("turn_id"):
                displayed_ids.append(item["turn_id"])
        return "\n".join(lines), displayed_ids

    if name == "ledger" and isinstance(result, list):
        header = f"Tool ledger({json.dumps(args, ensure_ascii=False)}) -> {len(result)} records:"
        lines = [header]
        used = len(header)
        for rec in result:
            line = (
                f'[{rec.get("kind")}:{rec.get("status")} | {rec.get("turn_id")} | '
                f'{rec.get("speaker")}] {rec.get("text")} | quote="{rec.get("quote")}"'
            )
            if len(lines) > 1 and used + len(line) + 1 > max_chars:
                lines.append(f"... ({len(result) - len(displayed_ids)} more omitted for budget)")
                break
            lines.append(line)
            used += len(line) + 1
            if rec.get("turn_id"):
                displayed_ids.append(rec["turn_id"])
        return "\n".join(lines), displayed_ids

    return json.dumps(result, ensure_ascii=False)[:max_chars], []


def _format_error_feedback(errors: list[str], draft: dict | None) -> str:
    hints: list[str] = []
    for err in errors:
        if err == "length":
            text = (draft or {}).get("text", "")
            n = validate.count_sentences(text)
            hints.append(f"- length: answer has {n} sentences; must be strictly 1 or 2 plain sentences.")
        elif err == "citation_count":
            hints.append("- citation_count: provide between 1 and 3 citations.")
        elif err == "wrong_team":
            hints.append("- wrong_team: meeting_id must belong to this team.")
        elif err == "as_of_violation":
            hints.append("- as_of_violation: meeting_id exceeds allowed as_of scope.")
        elif err == "citation:too_short":
            hints.append("- citation:too_short: every quote must contain at least 6 words.")
        elif err == "citation:not_found":
            hints.append(
                "- citation:not_found: quote did not match verbatim inside a single turn of that meeting. "
                "Copy an EXACT contiguous >=6-word span (keeping punctuation) from ONE displayed turn."
            )
        elif err == "citation:bad_meeting":
            hints.append("- citation:bad_meeting: use an exact meeting_id from Meetings in scope.")
        elif err == "citation_not_retrieved":
            hints.append("- citation_not_retrieved: cite ONLY turns present in the Evidence Notebook.")
        elif err == "invalid_final_json":
            hints.append(
                '- invalid_final_json: return ONLY a valid JSON object: '
                '{"answer": "...", "citations": [{"meeting_id": "...", "quote": "..."}]}'
            )
        else:
            hints.append(f"- {err}")
    return "\n".join(hints)


def _build_stateless_messages(
    question: str,
    notebook: EvidenceNotebook,
    last_tool_result: str,
    retrieved: set[str],
    *,
    step_label: str,
    directive: str = "",
    previous_draft: dict | None = None,
    tools_schemas: list[dict] | None = None,
) -> list[dict]:
    """Build a strictly stateless 2-message prompt under PROMPT_BUDGET_TOKENS."""
    is_synthesis = tools_schemas is None
    sys_prompt = SYNTHESIS_SYSTEM_PROMPT if is_synthesis else SYSTEM_PROMPT
    tools_overhead = _estimate_tokens(json.dumps(tools_schemas or [], ensure_ascii=False))
    system_tokens = _estimate_tokens(sys_prompt)
    # Leave generous safety margin below 6K (and Groq 8K TPM with max_tokens).
    max_user_tokens = max(
        1000, config.PROMPT_BUDGET_TOKENS - system_tokens - tools_overhead - 400
    )
    notebook_budget_chars = min(
        9000 if is_synthesis else 7500,
        config.EVIDENCE_BUDGET_TOKENS * 4,
        int(max_user_tokens * 4 * 0.80),
    )
    last_result_budget_chars = max(500, min(1000, int(max_user_tokens * 4 * 0.14)))

    def compose(nb_chars: int, last_chars: int) -> tuple[str, list[str]]:
        nb_text, shown_ids = notebook.render(
            max_chars=nb_chars, include_call_log=not is_synthesis
        )
        sections = [
            f"Question: {question}",
            f"Step: {step_label}",
            "=== EVIDENCE NOTEBOOK ===\n" + nb_text,
        ]
        if last_tool_result and not is_synthesis:
            trimmed_last = last_tool_result[:last_chars]
            sections.append("=== LAST TOOL RESULT SUMMARY ===\n" + trimmed_last)
        if previous_draft is not None:
            compact_draft = {
                "answer": previous_draft.get("answer") or previous_draft.get("text", ""),
                "citations": [
                    {"meeting_id": c.get("meeting_id"), "quote": c.get("quote")}
                    for c in (previous_draft.get("citations") or [])
                ],
            }
            sections.append(
                "=== PREVIOUS DRAFT ===\n"
                + json.dumps(compact_draft, ensure_ascii=False)
            )
        if directive:
            sections.append("=== INSTRUCTION ===\n" + directive)
        return "\n\n".join(sections), shown_ids

    user_content, shown_ids = compose(notebook_budget_chars, last_result_budget_chars)
    while (
        _estimate_tokens(sys_prompt + user_content) + tools_overhead
        > config.PROMPT_BUDGET_TOKENS - 300
        and notebook_budget_chars > 1200
    ):
        notebook_budget_chars = int(notebook_budget_chars * 0.75)
        last_result_budget_chars = int(last_result_budget_chars * 0.75)
        user_content, shown_ids = compose(notebook_budget_chars, last_result_budget_chars)

    retrieved.update(shown_ids)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


def _execute_calls(
    conn: sqlite3.Connection,
    ctx: tools.RunContext,
    tool_calls: list[dict],
    notebook: EvidenceNotebook,
    retrieved: set[str],
    default_k: int,
) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    call_records: list[dict] = []

    for call in tool_calls:
        name = call.get("name", "")
        raw_args = call.get("arguments")
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        if name == "search" and args.get("k") is None:
            args["k"] = default_k

        call_key = json.dumps({"name": name, "args": args}, sort_keys=True, ensure_ascii=False)
        if call_key in notebook.seen_calls:
            dup_msg = (
                "DUPLICATE CALL SKIPPED — already ran this exact call. "
                "Try a different query/meeting or output your final JSON answer now."
            )
            blocks.append(f"Tool {name}: {dup_msg}")
            call_records.append({"name": name, "args": args, "skipped_duplicate": True})
            continue
        notebook.seen_calls.add(call_key)

        try:
            result = tools.execute_tool(conn, ctx, name, args)
        except tools.VectorSearchError:
            raise
        except Exception as exc:
            err_msg = f"ERROR ({type(exc).__name__}): {exc}"
            notebook.add_call(f"{name}({json.dumps(args, ensure_ascii=False)}) -> {err_msg}")
            blocks.append(f"Tool {name}: {err_msg}")
            call_records.append({"name": name, "args": args, "error": err_msg})
            continue

        if name in {"search", "get_turns"} and isinstance(result, list):
            # Add ALL returned turns to the notebook (already trimmed to <=150 words
            # and capped by tools.py) so parallel tool calls never starve each other.
            added_ids = notebook.add_turns(result, explicit=(name == "get_turns"))
            retrieved.update(added_ids)
            meetings = sorted({item["meeting_id"] for item in result if item.get("meeting_id")})
            summary_line = (
                f"{name}({json.dumps(args, ensure_ascii=False)}) -> "
                f"{len(added_ids)} turns ({','.join(meetings)}): {', '.join(added_ids[:8])}"
                + ("..." if len(added_ids) > 8 else "")
            )
            notebook.add_call(summary_line)
            blocks.append(f"Tool {summary_line} [all added to Evidence Notebook]")
            call_records.append(
                {"name": name, "args": args, "n_returned": len(added_ids), "turn_ids": added_ids}
            )
        elif name == "ledger" and isinstance(result, list):
            added_ids = notebook.add_ledger(result)
            retrieved.update(added_ids)
            summary_line = f"ledger({json.dumps(args, ensure_ascii=False)}) -> {len(result)} records"
            notebook.add_call(summary_line)
            blocks.append(f"Tool {summary_line} [added to Evidence Notebook]")
            call_records.append(
                {"name": name, "args": args, "n_returned": len(result), "turn_ids": added_ids}
            )
        elif name == "verify" and isinstance(result, validate.VerifyResult):
            notebook.add_verification(
                str(args.get("meeting_id", "")),
                str(args.get("quote", "")),
                result,
            )
            summary_line = f"verify({args.get('meeting_id')}) -> ok={result.ok} ({result.reason}, turn_id={result.turn_id})"
            notebook.add_call(summary_line)
            blocks.append(f"Tool {summary_line}")
            call_records.append(
                {
                    "name": name,
                    "args": args,
                    "ok": result.ok,
                    "reason": result.reason,
                    "turn_id": result.turn_id,
                }
            )

    return "\n".join(blocks), call_records


def _record_usage(provenance: dict, resp: llm.LLMResult) -> None:
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
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "thoughts_tokens",
        "total_tokens",
        "billed_output_tokens",
    ):
        provenance["usage"][key] += int(usage.get(key, 0) or 0)


def _tool_rounds(
    conn: sqlite3.Connection,
    engine: llm.LLM,
    question: str,
    ctx: tools.RunContext,
    notebook: EvidenceNotebook,
    retrieved: set[str],
    active_tools: list[dict],
    trace: list[dict],
    *,
    phase: str,
    directive: str,
    previous_draft: dict | None,
    max_rounds: int,
    default_k: int,
    tier: str,
    provenance: dict,
) -> llm.LLMResult | None:
    last_tool_result = ""
    for round_idx in range(1, max_rounds + 1):
        is_final_round = round_idx == max_rounds
        round_directive = directive
        if is_final_round:
            round_directive = (
                directive
                + "\nFINAL ROUND: Do not call more tools. Output your final JSON answer now."
            )
        round_tools = None if is_final_round else active_tools
        round_schema = ANSWER_SCHEMA if is_final_round else None
        messages = _build_stateless_messages(
            question,
            notebook,
            last_tool_result,
            retrieved,
            step_label=f"{phase} round {round_idx}/{max_rounds}",
            directive=round_directive,
            previous_draft=previous_draft,
            tools_schemas=round_tools,
        )
        resp = engine.chat(
            messages,
            tools=round_tools,
            response_schema=round_schema,
            max_tokens=2048 if is_final_round else 1536,
            temperature=0.1,
            tier=tier,
        )
        _record_usage(provenance, resp)
        step_entry: dict[str, Any] = {
            "phase": phase,
            "round": round_idx,
            "provider": resp.provider,
            "model": resp.model,
            "cached": resp.cached,
            "fallback_used": resp.fallback_used,
            "latency_sec": resp.latency_sec,
            "usage": {k: v for k, v in (resp.usage or {}).items() if k != "raw"},
        }
        if not resp.tool_calls:
            step_entry["final_text"] = resp.text
            trace.append(step_entry)
            return resp
        last_tool_result, call_records = _execute_calls(
            conn, ctx, resp.tool_calls, notebook, retrieved, default_k
        )
        step_entry["tool_calls"] = call_records
        trace.append(step_entry)
    return None


def _log_run_and_citations(
    conn: sqlite3.Connection,
    run_id: str,
    question_id: str | None,
    draft: dict,
    provenance: dict,
    verified: bool,
) -> None:
    provider = provenance.get("last_provider") or (
        provenance["providers"][-1] if provenance["providers"] else "unknown"
    )
    model = provenance.get("last_model") or (
        provenance["models"][-1] if provenance["models"] else "unknown"
    )
    conn.execute(
        "INSERT OR REPLACE INTO answer_runs "
        "(run_id, question_id, kind, provider, model, fallback_used, usage_json, created_at) "
        "VALUES (?, ?, 'question', ?, ?, ?, ?, ?)",
        (
            run_id,
            question_id,
            provider,
            model,
            1 if provenance.get("fallback_used") else 0,
            json.dumps(provenance.get("usage", {}), ensure_ascii=False),
            _now_iso(),
        ),
    )
    for cite in draft.get("citations") or []:
        conn.execute(
            "INSERT INTO citations_log "
            "(run_id, question_id, meeting_id, turn_id, quote, verified, method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                question_id,
                cite.get("meeting_id", ""),
                cite.get("turn_id"),
                cite.get("quote", ""),
                1 if verified else 0,
                "word_boundary_single_turn_match",
            ),
        )
    conn.commit()


def run(
    conn: sqlite3.Connection,
    engine: llm.LLM,
    question: str,
    ctx: tools.RunContext,
    *,
    question_id: str | None = None,
    max_steps: int = config.MAX_STEPS,
    default_k: int = 8,
    use_ledger: bool = True,
    seed_search: bool = True,
    tier: str = "final",
) -> dict:
    if ctx.mode != "eval" and ctx.team is None:
        raise ValueError("team required outside eval mode")

    started = time.time()
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    notebook = EvidenceNotebook(conn, ctx)
    if not use_ledger:
        notebook.has_ledger = False
    active_tools = (
        list(tools.TOOL_SCHEMAS)
        if use_ledger
        else [t for t in tools.TOOL_SCHEMAS if t["function"]["name"] != "ledger"]
    )

    retrieved: set[str] = set()
    trace: list[dict] = []

    if seed_search:
        seed_k = min(8, max(default_k, 8))
        seed_hits = tools.search(conn, ctx, question, k=seed_k)
        added = list(notebook.add_turns(seed_hits))
        hits_by_letter: dict[str, int] = {}
        for h in seed_hits:
            if h.get("letter"):
                hits_by_letter[h["letter"]] = hits_by_letter.get(h["letter"], 0) + 1
        # Ensure every in-scope meeting (respecting ctx.as_of) gets seed hits
        # (b/c/d get >=3 because design decisions/revisions live there; a gets >=2).
        if ctx.team is not None:
            for mid in notebook.meetings:
                letter = mid[-1]
                want = 3 if letter in {"b", "c", "d"} else 2
                if hits_by_letter.get(letter, 0) < want:
                    extra = tools.search(
                        conn, ctx, question, meeting_letter=letter, k=want
                    )
                    added.extend(notebook.add_turns(extra))
                    seed_hits.extend(extra)
                    if extra:
                        hits_by_letter[letter] = hits_by_letter.get(letter, 0) + len(extra)
        # Expand +/-5..+10 neighboring turns around top non-overlapping hits in
        # b/c/d (up to 2 regions >=15 turns apart in c and d, 1 in b, and 1 in a
        # if a is top-ranked or early-context is asked). Neighbors inherit half
        # of their anchor's score so they are not evicted ahead of low-scoring
        # kickoff turns. Zero LLM cost.
        hits_by_meeting: dict[str, list[dict]] = {}
        for h in seed_hits:
            mid = h.get("meeting_id")
            if mid:
                hits_by_meeting.setdefault(mid, []).append(h)

        top_by_meeting: dict[str, dict] = {}
        for mid, items in hits_by_meeting.items():
            items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
            top_by_meeting[mid] = items[0]

        def _expand_regions(mid: str, items: list[dict], max_regions: int, before: int = 5, after: int = 10) -> None:
            chosen_seqs: list[int] = []
            for hit in items:
                seq = int(hit.get("seq", 0))
                if any(abs(seq - prev_seq) < 15 for prev_seq in chosen_seqs):
                    continue
                chosen_seqs.append(seq)
                parent_score = float(hit.get("score") or 0.01)
                neighbors = tools.get_turns(
                    conn,
                    ctx,
                    mid,
                    seq_start=max(0, seq - before),
                    seq_end=seq + after,
                    query=question,
                    max_turns=before + after + 1,
                )
                for n in neighbors:
                    n["score"] = max(float(n.get("score") or 0.0), round(parent_score * 0.5, 6))
                added.extend(notebook.add_turns(neighbors))
                if len(chosen_seqs) >= max_regions:
                    break

        if hits_by_meeting:
            best_mid = max(
                top_by_meeting.keys(),
                key=lambda m: float(top_by_meeting[m].get("score") or 0.0),
            )
            early_cue = bool(
                re.search(
                    r"\b(first|initial|initially|meeting a|look into|check|ask|asked|originally)\b",
                    question,
                    re.IGNORECASE,
                )
            )
            for mid, items in sorted(hits_by_meeting.items()):
                letter = mid[-1]
                if letter in {"c", "d"}:
                    _expand_regions(mid, items, max_regions=2)
                elif letter == "b":
                    _expand_regions(mid, items, max_regions=1)
                elif letter == "a" and (mid == best_mid or early_cue):
                    _expand_regions(mid, items, max_regions=1)

        # General finance/tradeoff coverage: when a question asks about costing,
        # budget, totals, prices, cuts, replacements, or dropped features, also
        # seed the latest meeting's costing discussion (at 0 LLM cost) because
        # AMI totals are spoken as words ("twelve point five", "fifteen point five").
        if ctx.team is not None and notebook.meetings and re.search(
            r"\b(cost|costing|costed|budget|total|limit|price|euro|dollar|money|afford|cut|drop|dropped|remove|removed|replace|replaced|swap|swapped|gave up|give up|expensive|cheap|keep|kept)\b",
            question,
            re.IGNORECASE,
        ):
            latest_mid = notebook.meetings[-1]
            latest_letter = latest_mid[-1]
            fin_hits = tools.search(
                conn,
                ctx,
                "budget cost total twelve point five Euro cut remove replace",
                meeting_letter=latest_letter,
                k=6,
            )
            added.extend(notebook.add_turns(fin_hits))
            if fin_hits:
                _expand_regions(latest_mid, fin_hits, max_regions=2, before=4, after=12)

        retrieved.update(added)
        meetings = sorted(top_by_meeting.keys())
        notebook.add_call(
            f"seed_search(question, k={seed_k}+neighbors) -> {len(set(added))} turns ({','.join(meetings)})"
        )
        trace.append(
            {
                "phase": "seed_search",
                "k": seed_k,
                "n_returned": len(set(added)),
                "meetings": meetings,
                "turn_ids": sorted(set(added)),
            }
        )
    provenance: dict[str, Any] = {
        "run_id": run_id,
        "question_id": question_id,
        "team": ctx.team,
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
    }

    draft: dict | None = None
    last_tool_result = ""

    for step in range(1, max_steps + 1):
        is_final_step = step == max_steps
        directive = (
            "FINAL STEP: Stop calling tools and output your final JSON answer now."
            if is_final_step
            else ""
        )
        step_tools = None if is_final_step else active_tools
        step_schema = ANSWER_SCHEMA if is_final_step else None
        messages = _build_stateless_messages(
            question,
            notebook,
            last_tool_result,
            retrieved,
            step_label=f"main {step}/{max_steps}",
            directive=directive,
            tools_schemas=step_tools,
        )
        resp = engine.chat(
            messages,
            tools=step_tools,
            response_schema=step_schema,
            max_tokens=2048 if is_final_step else 1536,
            temperature=0.1,
            tier=tier,
        )
        _record_usage(provenance, resp)
        step_entry: dict[str, Any] = {
            "phase": "main",
            "step": step,
            "provider": resp.provider,
            "model": resp.model,
            "cached": resp.cached,
            "fallback_used": resp.fallback_used,
            "latency_sec": resp.latency_sec,
            "usage": {k: v for k, v in (resp.usage or {}).items() if k != "raw"},
        }
        if resp.tool_calls:
            last_tool_result, call_records = _execute_calls(
                conn, ctx, resp.tool_calls, notebook, retrieved, default_k
            )
            step_entry["tool_calls"] = call_records
            trace.append(step_entry)
            continue

        step_entry["final_text"] = resp.text
        trace.append(step_entry)
        parsed = parse_final(resp.text)
        if parsed is None:
            draft = {"text": "", "answer": "", "citations": [], "_raw": resp.text[:500]}
        else:
            draft = parsed
        break

    if draft is None:
        provenance["elapsed_sec"] = round(time.time() - started, 2)
        return {
            "ok": False,
            "error": ["max_steps_no_final"],
            "warnings": [],
            "draft": None,
            "retrieved": sorted(retrieved),
            "trace": trace,
            "provenance": provenance,
        }

    # ---- CODE-ENFORCED VALIDATE GATE ----
    _clean_unicode_quotes(conn, ctx, draft)
    if not draft.get("text") and "_raw" in draft:
        ok, errors, warnings = False, ["invalid_final_json"], []
    else:
        ok, errors, warnings = validate.validate_answer(
            conn, ctx, draft, retrieved, question=question
        )

    trace.append({"phase": "validate_initial", "ok": ok, "errors": list(errors), "warnings": list(warnings)})

    # SOFT single_meeting -> ONE extra retrieval round, then re-validate
    if ok and "single_meeting" in warnings:
        single_directive = (
            "Your draft verified cleanly, but all citations are from a single meeting. "
            "Every question needs evidence from at least TWO meetings of this team. "
            "Search another meeting of this team (using `search` with a different `meeting_letter` "
            "or a different query), and return an updated JSON answer (1-2 sentences, 1-3 citations) "
            "with verified quotes from >=2 meetings if relevant evidence exists."
        )
        resp = _tool_rounds(
            conn,
            engine,
            question,
            ctx,
            notebook,
            retrieved,
            active_tools,
            trace,
            phase="single_meeting",
            directive=single_directive,
            previous_draft=draft,
            max_rounds=2,
            default_k=default_k,
            tier=tier,
            provenance=provenance,
        )
        if resp is None:
            provenance["elapsed_sec"] = round(time.time() - started, 2)
            return {
                "ok": False,
                "error": ["single_meeting_tool_loop"],
                "warnings": warnings,
                "draft": draft,
                "retrieved": sorted(retrieved),
                "trace": trace,
                "provenance": provenance,
            }
        updated = parse_final(resp.text)
        if updated is None:
            ok, errors, warnings = False, ["invalid_final_json"], []
        else:
            draft = updated
            _clean_unicode_quotes(conn, ctx, draft)
            ok, errors, warnings = validate.validate_answer(
                conn, ctx, draft, retrieved, question=question
            )
        trace.append({"phase": "validate_after_single_meeting", "ok": ok, "errors": list(errors), "warnings": list(warnings)})

    # ONE repair attempt on any hard validation error
    if not ok:
        repair_directive = (
            "Fix these validation/citation errors in your previous draft:\n"
            + _format_error_feedback(errors, draft)
            + "\nYou may call a tool if you need a verified turn, then output ONLY the corrected JSON object."
        )
        resp = _tool_rounds(
            conn,
            engine,
            question,
            ctx,
            notebook,
            retrieved,
            active_tools,
            trace,
            phase="repair",
            directive=repair_directive,
            previous_draft=draft,
            max_rounds=2,
            default_k=default_k,
            tier=tier,
            provenance=provenance,
        )
        if resp is None:
            provenance["elapsed_sec"] = round(time.time() - started, 2)
            return {
                "ok": False,
                "error": ["repair_tool_loop"],
                "warnings": warnings,
                "draft": draft,
                "retrieved": sorted(retrieved),
                "trace": trace,
                "provenance": provenance,
            }
        repaired = parse_final(resp.text)
        if repaired is None:
            ok, errors, warnings = False, ["invalid_final_json"], []
        else:
            draft = repaired
            _clean_unicode_quotes(conn, ctx, draft)
            ok, errors, warnings = validate.validate_answer(
                conn, ctx, draft, retrieved, question=question
            )
        trace.append({"phase": "validate_after_repair", "ok": ok, "errors": list(errors), "warnings": list(warnings)})

    # Safe post-repair pruning: if the draft already has >=1 verified, retrieved
    # citations and valid length, drop any unverified extra citation rather than
    # rejecting the entire answer (never invents or alters any quote).
    if not ok and isinstance(draft, dict) and draft.get("citations"):
        kept: list[dict] = []
        for cite in draft["citations"]:
            if not isinstance(cite, dict):
                continue
            vr = validate.verify_quote(conn, ctx, cite.get("meeting_id"), cite.get("quote", ""))
            if vr.ok and vr.turn_id in retrieved:
                clean_cite = dict(cite)
                clean_cite["turn_id"] = vr.turn_id
                kept.append(clean_cite)
        if kept and len(kept) < len(draft["citations"]):
            pruned_draft = dict(draft)
            pruned_draft["citations"] = kept[:3]
            p_ok, p_errors, p_warnings = validate.validate_answer(
                conn, ctx, pruned_draft, retrieved, question=question
            )
            if p_ok:
                draft, ok, errors, warnings = pruned_draft, p_ok, p_errors, p_warnings
                trace.append(
                    {
                        "phase": "prune_unverified_extra_citations",
                        "ok": ok,
                        "kept": len(kept[:3]),
                        "warnings": list(warnings),
                    }
                )

    provenance["elapsed_sec"] = round(time.time() - started, 2)
    if not ok:
        return {
            "ok": False,
            "error": errors,
            "warnings": warnings,
            "draft": draft,
            "retrieved": sorted(retrieved),
            "trace": trace,
            "provenance": provenance,
        }

    _log_run_and_citations(conn, run_id, question_id, draft, provenance, verified=True)
    return {
        "ok": True,
        "error": [],
        "warnings": warnings,
        "draft": draft,
        "retrieved": sorted(retrieved),
        "trace": trace,
        "provenance": provenance,
    }


def _load_work_state(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("results"), dict):
                return data
        except Exception:
            pass
    return {"results": {}}


def _save_work_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_questions(
    question_ids: list[str] | None = None,
    *,
    limit: int | None = None,
    k: int = 8,
    max_steps: int = config.MAX_STEPS,
    use_ledger: bool = True,
    out_path: str | Path | None = None,
    resume: bool = True,
    force: bool = False,
    retry_failed: bool = True,
) -> dict:
    data = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    if question_ids:
        wanted = set(question_ids)
        questions = [q for q in questions if q["question_id"] in wanted]
    if limit is not None:
        questions = questions[:limit]

    save_path = Path(out_path) if out_path else (config.WORK_DIR / "answers_work.json")
    # Preserve existing verified results for other questions when running a subset with force=True.
    state = _load_work_state(save_path) if (resume or question_ids) else {"results": {}}

    conn = store.connect()
    engine = llm.LLM(conn=conn)
    passed: list[str] = []
    failed: list[str] = []

    try:
        for item in questions:
            qid = item["question_id"]
            team = item["team"]
            question_text = item["question"]

            existing = state["results"].get(qid)
            if resume and not force and existing and existing.get("ok"):
                passed.append(qid)
                continue

            ctx = tools.RunContext(team=team, as_of=None, mode="answer")
            try:
                res = run(
                    conn,
                    engine,
                    question_text,
                    ctx,
                    question_id=qid,
                    max_steps=max_steps,
                    default_k=k,
                    use_ledger=use_ledger,
                )

                # Failed-question path: retry once with wider k and more steps
                if not res["ok"] and retry_failed:
                    retry_k = min(16, max(k + 4, 12))
                    retry_steps = min(10, max_steps + 2)
                    res_retry = run(
                        conn,
                        engine,
                        question_text,
                        ctx,
                        question_id=qid,
                        max_steps=retry_steps,
                        default_k=retry_k,
                        use_ledger=use_ledger,
                    )
                    res_retry["retried"] = True
                    res_retry["initial_error"] = res.get("error")
                    res = res_retry
            except llm.LLMError as exc:
                res = {
                    "ok": False,
                    "error": [f"llm_error:{type(exc).__name__}"],
                    "warnings": [],
                    "draft": None,
                    "retrieved": [],
                    "trace": [{"phase": "exception", "error": str(exc)[:1000]}],
                    "provenance": {"calls": 0, "last_model": None},
                }
                if existing and existing.get("ok"):
                    # Preserve previously verified cached work when all providers
                    # are unavailable during a forced re-run; record the failure
                    # cleanly in the return summary without fabricating an answer.
                    preserved = dict(existing)
                    preserved["last_rerun_error"] = str(exc)[:500]
                    state["results"][qid] = preserved
                    _save_work_state(save_path, state)
                    failed.append(qid)
                    print(
                        f"{qid} ({team}) ok=False errors={res['error']} "
                        f"(preserved prior verified answer)",
                        flush=True,
                    )
                    continue

            state["results"][qid] = {
                "question_id": qid,
                "team": team,
                "question": question_text,
                **res,
            }
            _save_work_state(save_path, state)

            if res["ok"]:
                passed.append(qid)
            else:
                failed.append(qid)
            print(
                f"{qid} ({team}) ok={res['ok']} "
                f"errors={res['error']} warnings={res['warnings']} "
                f"calls={res['provenance']['calls']} "
                f"model={res['provenance']['last_model']}",
                flush=True,
            )
    finally:
        conn.close()

    return {
        "total": len(questions),
        "passed": len(passed),
        "failed_count": len(failed),
        "passed_ids": passed,
        "failed": failed,
        "out_path": str(save_path),
    }
