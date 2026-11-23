from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import agent, config, cost_estimate, llm, store, tools, validate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_required_facts(answer_text: str, required_facts: list[str] | None) -> dict[str, Any]:
    """Deterministic keyword/phrase check when user supplies required_facts.

    A fact string may list '|' separated equivalent surface forms (e.g.
    '14.6|fourteen point six') so spoken words and numerals score identically.
    """
    if not required_facts:
        return {"matched": None, "total": 0, "recall": None, "missing": []}
    norm_ans = validate.normalize(answer_text or "")
    matched = 0
    missing: list[str] = []
    for fact in required_facts:
        alternatives = [validate.normalize(alt) for alt in str(fact).split("|") if alt.strip()]
        if any(alt in norm_ans for alt in alternatives):
            matched += 1
        else:
            missing.append(str(fact))
    return {
        "matched": matched,
        "total": len(required_facts),
        "recall": round(matched / len(required_facts), 4),
        "missing": missing,
    }


def _summarize_custom_rows(
    rows: list[dict[str, Any]], *, use_ledger: bool
) -> dict[str, Any]:
    n = len(rows)
    valid_citations_total = sum(int(r.get("verified_citations") or 0) for r in rows)
    total_citations = sum(int(r.get("total_citations") or 0) for r in rows)
    passed_validation = sum(1 for r in rows if r.get("ok"))
    multi_meeting_passed = sum(
        1 for r in rows if len(r.get("meetings_cited") or []) >= 2
    )
    failed_llm = sum(
        1
        for r in rows
        if any(str(e).startswith("llm_error:") for e in (r.get("errors") or []))
    )
    gold_recalls = [
        float(r["gold_retrieval_recall"])
        for r in rows
        if r.get("gold_retrieval_recall") is not None
    ]
    gold_cite_recalls = [
        float(r["gold_citation_recall"])
        for r in rows
        if r.get("gold_citation_recall") is not None
    ]
    fact_recalls: list[float] = []
    auto_full_fact_match = 0
    auto_partial_fact_match = 0
    auto_no_fact_match = 0
    for r in rows:
        fc = r.get("fact_check") or {}
        if fc.get("recall") is not None:
            rec = float(fc["recall"])
            fact_recalls.append(rec)
            if r.get("ok") and rec == 1.0:
                auto_full_fact_match += 1
            elif r.get("ok") and rec > 0.0:
                auto_partial_fact_match += 1
            else:
                auto_no_fact_match += 1

    return {
        "n_questions": n,
        "use_ledger": use_ledger,
        "passed_validation": passed_validation,
        "failed_validation": n - passed_validation,
        "failed_llm_errors": failed_llm,
        "validation_pass_rate": round(passed_validation / n, 4) if n else 0.0,
        "multi_meeting_count": multi_meeting_passed,
        "multi_meeting_rate": round(multi_meeting_passed / n, 4) if n else 0.0,
        "verified_citations": valid_citations_total,
        "total_citations": total_citations,
        "citation_validity_rate": (
            round(valid_citations_total / total_citations, 4)
            if total_citations
            else 0.0
        ),
        "mean_gold_retrieval_recall": (
            round(sum(gold_recalls) / len(gold_recalls), 4)
            if gold_recalls
            else None
        ),
        "mean_gold_citation_recall": (
            round(sum(gold_cite_recalls) / len(gold_cite_recalls), 4)
            if gold_cite_recalls
            else None
        ),
        "mean_fact_recall": (
            round(sum(fact_recalls) / len(fact_recalls), 4)
            if fact_recalls
            else None
        ),
        "automated_fact_tiers": {
            "all_required_facts_matched_and_verified": auto_full_fact_match,
            "partial_required_facts_matched_and_verified": auto_partial_fact_match,
            "zero_facts_or_failed_validation": auto_no_fact_match,
            "note": "Deterministic required_facts substring recall only; NOT a substitute for human_rubric_score.",
        },
        "results": rows,
    }


def evaluate_custom_questions(
    conn: sqlite3.Connection,
    engine: llm.LLM,
    questions: list[dict],
    *,
    k: int = 8,
    max_steps: int = config.MAX_STEPS,
    use_ledger: bool = False,
    existing_rows: dict[str, dict[str, Any]] | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Run the agent on user-supplied custom questions with manual ground truth.

    Never invents ground truth; compares agent output against the user's
    expected_answer, gold_turn_ids, and optional required_facts, and logs
    citation validity + retrieval recall. Supports resuming already-verified
    rows via existing_rows and incremental checkpointing via on_progress.
    """
    rows: list[dict[str, Any]] = []
    prior = existing_rows or {}

    for item in questions:
        qid = str(item["question_id"])
        team = str(item["team"])
        q_text = str(item["question"])
        as_of = item.get("as_of")
        gold_ids = set(item.get("gold_turn_ids") or [])
        expected = item.get("expected_answer")
        req_facts = item.get("required_facts") or []

        prev = prior.get(qid)
        if prev and prev.get("ok") and prev.get("question") == q_text:
            rows.append(prev)
            if on_progress is not None:
                on_progress(_summarize_custom_rows(rows, use_ledger=use_ledger))
            continue

        ctx = tools.RunContext(team=team, as_of=as_of, mode="answer")
        try:
            res = agent.run(
                conn,
                engine,
                q_text,
                ctx,
                question_id=qid,
                max_steps=max_steps,
                default_k=k,
                use_ledger=use_ledger,
            )
        except llm.LLMError as exc:
            res = {
                "ok": False,
                "error": [f"llm_error:{type(exc).__name__}"],
                "warnings": [],
                "draft": None,
                "retrieved": [],
                "trace": [{"phase": "exception", "error": str(exc)[:800]}],
                "provenance": {"calls": 0, "last_model": None},
            }

        draft = res.get("draft") or {}
        ans_text = draft.get("answer") or draft.get("text") or ""
        cites = draft.get("citations") or []
        retrieved_set = set(res.get("retrieved") or [])
        cited_turn_ids = set()

        verified_count = 0
        meetings_cited = set()
        for c in cites:
            vr = validate.verify_quote(conn, ctx, c.get("meeting_id"), c.get("quote", ""))
            if vr.ok:
                verified_count += 1
                if vr.turn_id:
                    cited_turn_ids.add(vr.turn_id)
                if c.get("meeting_id"):
                    meetings_cited.add(c["meeting_id"])

        gold_ret_recall = (
            round(len(retrieved_set & gold_ids) / len(gold_ids), 4)
            if gold_ids
            else None
        )
        gold_cite_hit = (
            round(len(cited_turn_ids & gold_ids) / len(gold_ids), 4)
            if gold_ids
            else None
        )
        fact_score = score_required_facts(ans_text, req_facts)

        row = {
            "question_id": qid,
            "team": team,
            "category": item.get("category"),
            "question": q_text,
            "expected_answer": expected,
            "predicted_answer": ans_text,
            "ok": res["ok"],
            "errors": res.get("error", []),
            "warnings": res.get("warnings", []),
            "citations": cites,
            "verified_citations": verified_count,
            "total_citations": len(cites),
            "meetings_cited": sorted(meetings_cited),
            "gold_turn_ids": sorted(gold_ids),
            "gold_retrieval_recall": gold_ret_recall,
            "gold_citation_recall": gold_cite_hit,
            "fact_check": fact_score,
            "manual_rubric_score": None,  # Explicitly unpopulated: requires human evaluator sign-off
            "provenance": res.get("provenance", {}),
        }
        rows.append(row)
        print(
            f"EVAL {qid} ({team}) ok={row['ok']} errors={row['errors']} "
            f"warnings={row['warnings']} facts={fact_score.get('matched')}/{fact_score.get('total')} "
            f"model={(row['provenance'] or {}).get('last_model')}",
            flush=True,
        )
        if on_progress is not None:
            on_progress(_summarize_custom_rows(rows, use_ledger=use_ledger))

    return _summarize_custom_rows(rows, use_ledger=use_ledger)


def _targets_from_answers_work(
    work_path: Path, max_questions: int = 10
) -> list[dict[str, Any]]:
    """Build retrieval-only targets from verified citations in answers_work.json.

    Explicitly labeled as verified-citation proxy targets (NOT manual ground truth).
    """
    if not work_path.exists():
        return []
    data = json.loads(work_path.read_text(encoding="utf-8"))
    results = data.get("results") or {}
    targets: list[dict[str, Any]] = []
    for qid in sorted(results.keys()):
        entry = results[qid]
        if not entry.get("ok"):
            continue
        cites = (entry.get("draft") or {}).get("citations") or []
        turn_ids = sorted({c["turn_id"] for c in cites if c.get("turn_id")})
        if not turn_ids:
            continue
        targets.append(
            {
                "question_id": qid,
                "team": entry["team"],
                "question": entry["question"],
                "gold_turn_ids": turn_ids,
                "target_source": "verified_citation_proxy_NOT_manual_ground_truth",
            }
        )
        if len(targets) >= max_questions:
            break
    return targets


def run_no_team_retrieval_test(
    conn: sqlite3.Connection,
    items: list[dict],
    *,
    ks: tuple[int, ...] = (8, 16),
) -> dict[str, Any]:
    """Compare recall@k with team filter vs no-team-filter (ctx.mode='eval', team=None).

    Also verifies that answer/brief modes reject team=None.
    """
    # Enforce guardrail check inside eval harness
    for bad_mode in ("answer", "brief"):
        try:
            tools.RunContext(team=None, mode=bad_mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mode={bad_mode} failed to reject team=None")

    per_k_summary: dict[str, Any] = {}
    details: list[dict[str, Any]] = []

    for k in ks:
        team_recalls: list[float] = []
        noteam_recalls: list[float] = []
        cross_team_rates: list[float] = []
        k_rows: list[dict[str, Any]] = []

        for item in items:
            gold = set(item.get("gold_turn_ids") or [])
            if not gold:
                continue
            team = item["team"]
            query = item["question"]

            ctx_team = tools.RunContext(team=team, as_of=item.get("as_of"), mode="answer")
            ctx_eval = tools.RunContext(team=None, as_of=item.get("as_of"), mode="eval")

            t_start = time.time()
            hits_team = tools.search(conn, ctx_team, query, k=k)
            hits_eval = tools.search(conn, ctx_eval, query, k=k)
            elapsed = round(time.time() - t_start, 3)

            ids_team = {h["turn_id"] for h in hits_team}
            ids_eval = {h["turn_id"] for h in hits_eval}

            rec_team = len(ids_team & gold) / len(gold)
            rec_eval = len(ids_eval & gold) / len(gold)
            other_team_hits = sum(1 for h in hits_eval if h["team"] != team)
            other_rate = other_team_hits / len(hits_eval) if hits_eval else 0.0

            team_recalls.append(rec_team)
            noteam_recalls.append(rec_eval)
            cross_team_rates.append(other_rate)
            k_rows.append(
                {
                    "question_id": item["question_id"],
                    "team": team,
                    "k": k,
                    "gold_count": len(gold),
                    "team_filtered_hits": len(ids_team & gold),
                    "no_team_hits": len(ids_eval & gold),
                    "recall_team_filtered": round(rec_team, 4),
                    "recall_no_team": round(rec_eval, 4),
                    "recall_gap": round(rec_team - rec_eval, 4),
                    "cross_team_hit_rate": round(other_rate, 4),
                    "elapsed_sec": elapsed,
                }
            )

        n = len(team_recalls)
        mean_team = round(sum(team_recalls) / n, 4) if n else 0.0
        mean_noteam = round(sum(noteam_recalls) / n, 4) if n else 0.0
        mean_other = round(sum(cross_team_rates) / n, 4) if n else 0.0
        per_k_summary[f"k={k}"] = {
            "n_questions": n,
            "mean_recall_team_filtered": mean_team,
            "mean_recall_no_team": mean_noteam,
            "mean_recall_gap": round(mean_team - mean_noteam, 4),
            "mean_cross_team_contamination": mean_other,
        }
        details.extend(k_rows)

    return {
        "eval_mode_guard_verified": True,
        "target_sources": sorted({i.get("target_source", "user_manual_ground_truth") for i in items}),
        "summary": per_k_summary,
        "details": details,
    }


def run(
    *,
    questions_path: str | Path | None = None,
    k: int = 8,
    max_steps: int = config.MAX_STEPS,
    no_team_test: bool = False,
    out_path: str | Path | None = None,
    compare_ledger: bool = False,
) -> dict[str, Any]:
    q_file = Path(questions_path) if questions_path else (config.WORK_DIR / "eval_questions.json")
    report_path = Path(out_path) if out_path else (config.METRICS_DIR / "eval_report.json")

    conn = store.connect()
    try:
        report: dict[str, Any] = {
            "generated_at": _now_iso(),
            "questions_path": str(q_file),
            "custom_eval": None,
            "no_team_retrieval_test": None,
            "cost_time_estimate": cost_estimate.build_estimate(conn),
        }

        custom_items: list[dict] = []
        if q_file.exists():
            loaded = json.loads(q_file.read_text(encoding="utf-8"))
            custom_items = loaded.get("questions", loaded) if isinstance(loaded, dict) else loaded
            # Filter out placeholder/empty template rows without real questions
            custom_items = [
                q for q in custom_items
                if isinstance(q, dict)
                and q.get("question")
                and not str(q.get("question_id", "")).startswith("TEMPLATE")
            ]

        if custom_items and not no_team_test:
            existing_rows: dict[str, dict[str, Any]] = {}
            if report_path.exists():
                try:
                    prev_report = json.loads(report_path.read_text(encoding="utf-8"))
                    for row in ((prev_report.get("custom_eval") or {}).get("results") or []):
                        if isinstance(row, dict) and row.get("question_id"):
                            existing_rows[str(row["question_id"])] = row
                except Exception:
                    existing_rows = {}

            def _save_partial(partial_eval: dict[str, Any]) -> None:
                report["custom_eval"] = partial_eval
                report_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = report_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(report_path)

            engine = llm.LLM(conn=conn)
            report["custom_eval"] = evaluate_custom_questions(
                conn,
                engine,
                custom_items,
                k=k,
                max_steps=max_steps,
                use_ledger=False,
                existing_rows=existing_rows,
                on_progress=_save_partial,
            )
            if compare_ledger:
                report["custom_eval_with_ledger"] = evaluate_custom_questions(
                    conn, engine, custom_items, k=k, max_steps=max_steps, use_ledger=True
                )

        if no_team_test or custom_items:
            retrieval_items = [q for q in custom_items if q.get("gold_turn_ids")]
            if len(retrieval_items) < 5:
                retrieval_items = _targets_from_answers_work(
                    config.WORK_DIR / "answers_work.json", max_questions=10
                )
            if retrieval_items:
                ks = (k, max(k * 2, 16)) if k != 16 else (8, 16)
                report["no_team_retrieval_test"] = run_no_team_retrieval_test(
                    conn, retrieval_items, ks=ks
                )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "ok": True,
            "out_path": str(report_path),
            "custom_questions_evaluated": len(custom_items) if report["custom_eval"] else 0,
            "no_team_summary": (report["no_team_retrieval_test"] or {}).get("summary"),
            "cost_time_scales": list(report["cost_time_estimate"]["scales"].keys()),
        }
    finally:
        conn.close()
