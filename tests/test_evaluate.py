import json
import sqlite3

import pytest

from catalyst import cost_estimate, evaluate, llm, store, tools


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    meetings = [
        ("TS3006b", "TS3006", "b", 2, 1),
        ("TS3006d", "TS3006", "d", 4, 1),
        ("ES2003d", "ES2003", "d", 4, 1),
    ]
    conn.executemany("INSERT INTO meetings VALUES (?,?,?,?,?)", meetings)

    turns = [
        ("TS3006b#00010", "TS3006b", "TS3006", "b", 10, "00:02:00", "Speaker A",
         "We plan to include both an LCD screen and speech recognition.", 0, 0, 11),
        ("TS3006d#00101", "TS3006d", "TS3006", "d", 101, "00:10:00", "Speaker D",
         "Because if we lose the LCD screen we need many extra buttons.", 0, 0, 12),
        ("ES2003d#00012", "ES2003d", "ES2003", "d", 12, "00:03:00", "Speaker D",
         "We plan to include an LCD screen and rubber grip for cost.", 0, 0, 12),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)
    conn.executemany(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        [(t[7], t[0], t[1], t[2]) for t in turns],
    )
    return conn


def test_score_required_facts():
    s = evaluate.score_required_facts(
        "Team TS3006 dropped speech recognition and kept the LCD screen at 14.6 Euros.",
        ["speech recognition", "LCD screen", "14.6|fourteen point six", "Speaker B"],
    )
    assert s["matched"] == 3
    assert s["total"] == 4
    assert s["missing"] == ["Speaker B"]


def test_no_team_retrieval_test_measures_gap_and_enforces_guard(monkeypatch):
    conn = make_db()
    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    items = [
        {
            "question_id": "E01",
            "team": "TS3006",
            "question": "LCD screen speech recognition extra buttons",
            "gold_turn_ids": ["TS3006b#00010", "TS3006d#00101"],
        }
    ]
    res = evaluate.run_no_team_retrieval_test(conn, items, ks=(2, 8))
    assert res["eval_mode_guard_verified"] is True
    assert "k=2" in res["summary"]
    assert "k=8" in res["summary"]
    assert res["summary"]["k=8"]["mean_recall_team_filtered"] == 1.0
    # Cross-team turn ES2003d#00012 shows up in eval mode (team=None)
    assert res["summary"]["k=8"]["mean_cross_team_contamination"] > 0.0


def test_evaluate_custom_questions_with_mock_llm(monkeypatch):
    conn = make_db()
    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    class MockLLM:
        def chat(self, messages, **kwargs):
            return llm.LLMResult(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 kept the LCD screen to avoid needing many extra buttons.",
                        "citations": [
                            {"meeting_id": "TS3006b", "quote": "We plan to include both an LCD screen and speech recognition."},
                            {"meeting_id": "TS3006d", "quote": "Because if we lose the LCD screen we need many extra buttons."},
                        ],
                    }
                ),
                tool_calls=[],
                usage={"prompt_tokens": 100, "completion_tokens": 50, "thoughts_tokens": 0, "total_tokens": 150, "billed_output_tokens": 50},
                provider="mock",
                model="mock-eval",
            )

    custom = [
        {
            "question_id": "C01",
            "team": "TS3006",
            "question": "Why did TS3006 keep the LCD screen?",
            "expected_answer": "To avoid needing many extra buttons.",
            "gold_turn_ids": ["TS3006b#00010", "TS3006d#00101"],
            "required_facts": ["LCD screen", "extra buttons"],
        }
    ]
    report = evaluate.evaluate_custom_questions(conn, MockLLM(), custom, k=8, max_steps=2)
    assert report["validation_pass_rate"] == 1.0
    assert report["citation_validity_rate"] == 1.0
    assert report["multi_meeting_rate"] == 1.0
    assert report["mean_gold_retrieval_recall"] == 1.0
    assert report["mean_fact_recall"] == 1.0
    assert report["results"][0]["manual_rubric_score"] is None


def test_cost_estimate_includes_thoughts_and_verification_tags():
    conn = make_db()
    est = cost_estimate.build_estimate(conn)
    assert "138_meetings" in est["scales"]
    assert "1000_meetings" in est["scales"]
    assert "10000_meetings" in est["scales"]
    assert est["measured_baselines"]["embeddings_full_run"]["centres_per_sec"] == 2.34
    assert est["pricing"]["gemini-3.8-flash_through_2026"]["status"] == "VERIFIED"
    assert "NOT VERIFIED" in est["pricing"]["gemini-2.5-flash_paid"]["status"]
    assert "NOT VERIFIED" in est["pricing"]["groq_gpt_oss_120b_paid"]["status"]
    s1000 = est["scales"]["1000_meetings"]["ledger_plus_reconcile"]
    assert s1000["reconcile_thoughts_tokens_included"] > 0
    assert s1000["estimated_cost_usd"]["gemini_3.8_flash_2026_VERIFIED_price"] > 0


def test_evaluate_custom_questions_handles_all_providers_failed_without_fabrication(monkeypatch):
    conn = make_db()
    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    class FailingLLM:
        def chat(self, *args, **kwargs):
            raise llm.AllProvidersFailed("all down", errors=[{"provider": "gemini", "status": 429}])

    custom = [
        {
            "question_id": "C_FAIL",
            "team": "TS3006",
            "question": "Why did TS3006 keep the LCD screen?",
            "expected_answer": "To avoid needing many extra buttons.",
            "gold_turn_ids": ["TS3006d#00101"],
            "required_facts": ["LCD screen"],
        }
    ]
    report = evaluate.evaluate_custom_questions(conn, FailingLLM(), custom, k=8, max_steps=2)
    assert report["failed_llm_errors"] == 1
    assert report["validation_pass_rate"] == 0.0
    assert report["results"][0]["ok"] is False
    assert report["results"][0]["predicted_answer"] == ""
    assert "llm_error:AllProvidersFailed" in report["results"][0]["errors"]
    assert report["results"][0]["manual_rubric_score"] is None


def test_evaluate_custom_questions_resumes_verified_rows_and_checkpoints(monkeypatch):
    conn = make_db()
    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    calls = {"count": 0}

    class CountingFailLLM:
        def chat(self, *args, **kwargs):
            calls["count"] += 1
            raise llm.AllProvidersFailed("rate limited")

    custom = [
        {
            "question_id": "C1",
            "team": "TS3006",
            "question": "Why did TS3006 keep the LCD screen?",
            "expected_answer": "To avoid needing many extra buttons.",
            "gold_turn_ids": ["TS3006d#00101"],
            "required_facts": ["LCD screen"],
        },
        {
            "question_id": "C2",
            "team": "TS3006",
            "question": "What did TS3006 drop?",
            "expected_answer": "Speech recognition.",
            "gold_turn_ids": ["TS3006d#00101"],
            "required_facts": ["speech"],
        },
    ]
    prior = {
        "C1": {
            "question_id": "C1",
            "team": "TS3006",
            "question": "Why did TS3006 keep the LCD screen?",
            "ok": True,
            "predicted_answer": "They kept the LCD screen.",
            "verified_citations": 1,
            "total_citations": 1,
            "meetings_cited": ["TS3006b", "TS3006d"],
            "gold_retrieval_recall": 1.0,
            "gold_citation_recall": 1.0,
            "fact_check": {"matched": 1, "total": 1, "recall": 1.0, "missing": []},
            "errors": [],
            "manual_rubric_score": None,
        }
    }
    snapshots = []
    report = evaluate.evaluate_custom_questions(
        conn,
        CountingFailLLM(),
        custom,
        k=8,
        max_steps=1,
        existing_rows=prior,
        on_progress=lambda partial: snapshots.append(partial["n_questions"]),
    )
    # C1 was resumed without calling LLM; only C2 attempted LLM and failed cleanly
    assert calls["count"] == 1
    assert snapshots == [1, 2]
    assert report["passed_validation"] == 1
    assert report["failed_llm_errors"] == 1
    assert report["results"][0]["ok"] is True
    assert report["results"][1]["ok"] is False
