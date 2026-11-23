from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import config, store, tools, validate


class CitationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meeting_id: str
    quote: str


class AnswerOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    answer: str
    citations: list[CitationOut] = Field(min_length=1, max_length=3)


class SubmissionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[AnswerOut]
    brief: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reverify_question(
    conn: sqlite3.Connection,
    q_spec: dict,
    entry: dict | None,
) -> tuple[bool, list[str], list[str], list[dict]]:
    if not entry:
        return False, ["missing_result"], [], []
    if not entry.get("ok"):
        return False, list(entry.get("error") or ["agent_failed"]), list(entry.get("warnings") or []), []

    draft = entry.get("draft")
    if not isinstance(draft, dict):
        return False, ["missing_draft"], [], []

    ctx = tools.RunContext(team=q_spec["team"], as_of=None, mode="answer")
    retrieved = set(entry.get("retrieved") or [])
    answer_text = draft.get("answer") or draft.get("text") or ""
    citations = [
        {"meeting_id": c.get("meeting_id", ""), "quote": c.get("quote", "")}
        for c in (draft.get("citations") or [])
        if isinstance(c, dict)
    ]
    check_payload = {"text": answer_text, "citations": citations}
    ok, errors, warnings = validate.validate_answer(
        conn, ctx, check_payload, retrieved, question=q_spec["question"]
    )
    return ok, errors, warnings, check_payload["citations"]


def run(
    answers_path: str | Path | None = None,
    brief_path: str | Path | None = None,
    out_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    require_brief: bool = False,
) -> dict:
    questions_data = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = questions_data["questions"]

    ans_file = Path(answers_path) if answers_path else (config.WORK_DIR / "answers_work.json")
    br_file = Path(brief_path) if brief_path else (config.WORK_DIR / "brief_TS3012.json")
    out_file = Path(out_path) if out_path else (config.ROOT / "answers.json")
    prov_file = Path(provenance_path) if provenance_path else (config.ROOT / "answers_provenance.json")

    work_results: dict[str, dict] = {}
    if ans_file.exists():
        loaded = json.loads(ans_file.read_text(encoding="utf-8"))
        work_results = loaded.get("results", {}) if isinstance(loaded, dict) else {}

    own_conn = conn is None
    conn = conn or store.connect()
    try:
        verified_answers: list[AnswerOut] = []
        failures: list[dict[str, Any]] = []
        per_question_prov: list[dict[str, Any]] = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0,
            "billed_output_tokens": 0,
        }
        models_used: dict[str, int] = {}

        for q_spec in questions:
            qid = q_spec["question_id"]
            team = q_spec["team"]
            entry = work_results.get(qid)
            ok, errors, warnings, verified_cites = _reverify_question(conn, q_spec, entry)

            prov = (entry or {}).get("provenance") or {}
            usage = prov.get("usage") or {}
            for k in usage_totals:
                usage_totals[k] += int(usage.get(k, 0) or 0)
            last_model = prov.get("last_model")
            if last_model:
                models_used[last_model] = models_used.get(last_model, 0) + 1

            per_question_prov.append(
                {
                    "question_id": qid,
                    "team": team,
                    "ok": ok,
                    "errors": errors,
                    "warnings": warnings,
                    "provider": prov.get("last_provider"),
                    "model": last_model,
                    "providers": prov.get("providers", []),
                    "models": prov.get("models", []),
                    "fallback_used": bool(prov.get("fallback_used")),
                    "calls": prov.get("calls", 0),
                    "usage": usage,
                    "elapsed_sec": prov.get("elapsed_sec"),
                    "retried": bool((entry or {}).get("retried")),
                    "verified_turn_ids": [c.get("turn_id") for c in verified_cites if c.get("turn_id")],
                    "meetings_cited": sorted({c.get("meeting_id") for c in verified_cites if c.get("meeting_id")}),
                }
            )

            if ok:
                draft = entry["draft"]
                verified_answers.append(
                    AnswerOut(
                        question_id=qid,
                        answer=draft.get("answer") or draft.get("text") or "",
                        citations=[
                            CitationOut(meeting_id=c["meeting_id"], quote=c["quote"])
                            for c in verified_cites
                        ],
                    )
                )
            else:
                failures.append(
                    {
                        "question_id": qid,
                        "team": team,
                        "errors": errors,
                        "warnings": warnings,
                    }
                )

        # Optional brief handling (validated if present; required only if require_brief=True)
        brief_text = ""
        brief_prov: dict[str, Any] = {"status": "pending"}
        brief_ok = True
        brief_errors: list[str] = []

        if br_file.exists():
            brief_data = json.loads(br_file.read_text(encoding="utf-8"))
            brief_text = str(brief_data.get("brief") or brief_data.get("raw_text") or "")
            ctx_brief = tools.RunContext(team="TS3012", as_of="c", mode="brief")
            brief_ok, brief_errors = validate.validate_brief(
                conn,
                ctx_brief,
                {"raw_text": brief_text, "bullets": brief_data.get("bullets")},
            )
            brief_prov = {
                "status": "verified" if brief_ok else "failed",
                "errors": brief_errors,
                "provenance": brief_data.get("provenance", {}),
            }
        elif require_brief:
            brief_ok = False
            brief_errors = ["brief_missing"]
            brief_prov = {"status": "missing", "errors": brief_errors}

        provenance_doc = {
            "generated_at": _now_iso(),
            "total_questions": len(questions),
            "verified_count": len(verified_answers),
            "failed_count": len(failures),
            "all_25_verified": len(verified_answers) == len(questions) and len(failures) == 0,
            "models_used": models_used,
            "usage_totals": usage_totals,
            "manual_review_queue": failures,
            "brief": brief_prov,
            "questions": per_question_prov,
        }
        prov_file.parent.mkdir(parents=True, exist_ok=True)
        prov_file.write_text(
            json.dumps(provenance_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        can_write_answers = (
            len(verified_answers) == len(questions)
            and len(failures) == 0
            and brief_ok
        )
        wrote_answers = False
        if can_write_answers:
            submission = SubmissionOut(answers=verified_answers, brief=brief_text)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                json.dumps(submission.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            wrote_answers = True

        return {
            "ok": can_write_answers,
            "wrote_answers": wrote_answers,
            "out_path": str(out_file) if wrote_answers else None,
            "provenance_path": str(prov_file),
            "verified_count": len(verified_answers),
            "total_questions": len(questions),
            "failures": failures,
            "brief_status": brief_prov["status"],
            "brief_errors": brief_errors,
        }
    finally:
        if own_conn:
            conn.close()
