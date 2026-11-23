import json
import sqlite3

import pytest

from catalyst import agent, config, llm, store, tools


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3006b','TS3006','b',2,2)")
    conn.execute("INSERT INTO meetings VALUES ('TS3006d','TS3006','d',4,2)")

    turns = [
        (
            "TS3006b#00010",
            "TS3006b",
            "TS3006",
            "b",
            10,
            "00:02:00",
            "Speaker A",
            "We plan to include both an LCD screen and speech recognition in the remote.",
            0,
            0,
            14,
        ),
        (
            "TS3006b#00011",
            "TS3006b",
            "TS3006",
            "b",
            11,
            "00:02:10",
            "Speaker D",
            "Both features sound attractive to young buyers if the budget allows.",
            0,
            0,
            11,
        ),
        (
            "TS3006d#00100",
            "TS3006d",
            "TS3006",
            "d",
            100,
            "00:10:00",
            "Speaker D",
            "There are more people who like speech recognition than an LCD screen.",
            0,
            0,
            12,
        ),
        (
            "TS3006d#00101",
            "TS3006d",
            "TS3006",
            "d",
            101,
            "00:10:15",
            "Speaker A",
            "So because of cost we drop the LCD screen and keep speech recognition.",
            0,
            0,
            13,
        ),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)
    conn.executemany(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        [(t[7], t[0], t[1], t[2]) for t in turns],
    )
    return conn


class MockLLM:
    def __init__(self, responses: list[llm.LLMResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, **kwargs) -> llm.LLMResult:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("MockLLM ran out of scripted responses")
        return self.responses.pop(0)


def make_result(text: str = "", tool_calls: list[dict] | None = None) -> llm.LLMResult:
    return llm.LLMResult(
        text=text,
        tool_calls=tool_calls or [],
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "thoughts_tokens": 10,
            "total_tokens": 170,
            "billed_output_tokens": 50,
        },
        provider="mock",
        model="mock-1",
        fallback_used=False,
    )


def _patch_vec(monkeypatch, ids_by_call=None):
    calls = {"n": 0}

    def fake_vec(ctx, query, team, n, meeting_letter=None):
        idx = calls["n"]
        calls["n"] += 1
        if ids_by_call and idx < len(ids_by_call):
            return [{"turn_id": tid} for tid in ids_by_call[idx]]
        return []

    monkeypatch.setattr(tools, "_vector_search", fake_vec)


def test_agent_happy_path_stateless_and_verify_overrides_turn_id(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    mock = MockLLM(
        [
            make_result(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "search",
                        "arguments": {"query": "LCD screen speech recognition"},
                    }
                ]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": (
                            "Team TS3006 dropped the LCD screen and kept speech recognition "
                            "because Speaker D noted more people liked speech recognition."
                        ),
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "turn_id": "FAKE_LLM_ID_1",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "turn_id": "FAKE_LLM_ID_2",
                                "quote": "There are more people who like speech recognition than an LCD screen.",
                            },
                        ],
                    }
                )
            ),
        ]
    )

    res = agent.run(conn, mock, "Which feature was kept?", ctx, question_id="Q01")
    assert res["ok"] is True
    assert res["error"] == []
    assert res["warnings"] == []
    # Authoritative verify() turn_ids override the LLM's fake turn_ids
    cites = res["draft"]["citations"]
    assert cites[0]["turn_id"] == "TS3006b#00010"
    assert cites[1]["turn_id"] == "TS3006d#00100"
    # Stateless: every LLM call has exactly 2 messages (system + user), never growing history
    assert all(len(c["messages"]) == 2 for c in mock.calls)
    # Logged to answer_runs and citations_log
    assert conn.execute("SELECT COUNT(*) FROM answer_runs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM citations_log WHERE verified = 1").fetchone()[0] == 2


def test_agent_max_steps_draft_is_none_rejects(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    # Script max_steps=3 tool-calling responses that never produce a final answer
    mock = MockLLM(
        [
            make_result(tool_calls=[{"id": f"c{i}", "name": "search", "arguments": {"query": "LCD"}}])
            for i in range(3)
        ]
    )

    res = agent.run(conn, mock, "Which feature was kept?", ctx, max_steps=3)
    assert res["ok"] is False
    assert res["error"] == ["max_steps_no_final"]
    assert res["draft"] is None
    assert len(mock.calls) == 3


def test_agent_single_meeting_triggers_extra_retrieval_round(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    mock = MockLLM(
        [
            # Main step 1: search meeting d only
            make_result(
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "search",
                        "arguments": {"query": "speech recognition", "meeting_letter": "d"},
                    }
                ]
            ),
            # Main step 2: valid draft, but only from meeting d -> triggers single_meeting warning
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            }
                        ],
                    }
                )
            ),
            # single_meeting round 1: searches meeting b
            make_result(
                tool_calls=[
                    {
                        "id": "c2",
                        "name": "search",
                        "arguments": {"query": "LCD screen", "meeting_letter": "b"},
                    }
                ]
            ),
            # single_meeting round 2: final answer citing both b and d
            make_result(
                text=json.dumps(
                    {
                        "answer": (
                            "Team TS3006 initially planned both features in meeting b, then "
                            "dropped the LCD screen and kept speech recognition in meeting d."
                        ),
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            },
                        ],
                    }
                )
            ),
        ]
    )

    res = agent.run(conn, mock, "Which feature was kept?", ctx)
    assert res["ok"] is True
    assert res["warnings"] == []
    assert {c["meeting_id"] for c in res["draft"]["citations"]} == {"TS3006b", "TS3006d"}


def test_agent_single_meeting_persistent_tool_loop_rejects(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    mock = MockLLM(
        [
            make_result(
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"query": "speech", "meeting_letter": "d"}}]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            }
                        ],
                    }
                )
            ),
            # Keeps calling tools through both single_meeting rounds
            make_result(tool_calls=[{"id": "c2", "name": "search", "arguments": {"query": "LCD", "meeting_letter": "b"}}]),
            make_result(tool_calls=[{"id": "c3", "name": "search", "arguments": {"query": "cost", "meeting_letter": "b"}}]),
        ]
    )

    res = agent.run(conn, mock, "Which feature was kept?", ctx)
    assert res["ok"] is False
    assert res["error"] == ["single_meeting_tool_loop"]


def test_agent_repair_fixes_bad_citation_and_persistent_repair_rejects(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    # Case 1: Initial draft has a paraphrased quote -> repair fixes it!
    mock_ok = MockLLM(
        [
            make_result(
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"query": "LCD speech recognition"}}]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 kept speech recognition over the LCD screen.",
                        "citations": [
                            {"meeting_id": "TS3006d", "quote": "this is a paraphrased quote that fails verification"},
                        ],
                    }
                )
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            },
                        ],
                    }
                )
            ),
        ]
    )

    res_ok = agent.run(conn, mock_ok, "Which feature was kept?", ctx)
    assert res_ok["ok"] is True
    assert res_ok["error"] == []

    # Case 2: Repair keeps calling tools -> rejected with repair_tool_loop
    mock_loop = MockLLM(
        [
            make_result(
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"query": "LCD"}}]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 kept speech recognition.",
                        "citations": [{"meeting_id": "TS3006d", "quote": "too short"}],
                    }
                )
            ),
            make_result(tool_calls=[{"id": "c2", "name": "search", "arguments": {"query": "speech"}}]),
            make_result(tool_calls=[{"id": "c3", "name": "search", "arguments": {"query": "cost"}}]),
        ]
    )

    res_loop = agent.run(conn, mock_loop, "Which feature was kept?", ctx)
    assert res_loop["ok"] is False
    assert res_loop["error"] == ["repair_tool_loop"]


def test_agent_rejects_unretrieved_citation(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    # Model never calls search, tries to cite a real turn verbatim -> must fail citation_not_retrieved
    real_quote = "So because of cost we drop the LCD screen and keep speech recognition."
    bad_draft = json.dumps(
        {
            "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
            "citations": [{"meeting_id": "TS3006d", "quote": real_quote}],
        }
    )
    mock = MockLLM([make_result(text=bad_draft), make_result(text=bad_draft)])

    res = agent.run(conn, mock, "Which feature was kept?", ctx, seed_search=False)
    assert res["ok"] is False
    assert "citation_not_retrieved" in res["error"]


def test_seed_search_and_duplicate_call_guard(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    mock = MockLLM(
        [
            make_result(
                tool_calls=[
                    {"id": "c1", "name": "search", "arguments": {"query": "LCD", "meeting_letter": "b"}},
                    {"id": "c2", "name": "search", "arguments": {"query": "LCD", "meeting_letter": "b"}},
                ]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            },
                        ],
                    }
                )
            ),
        ]
    )

    res = agent.run(conn, mock, "LCD screen speech recognition", ctx, seed_search=True)
    assert res["ok"] is True
    assert res["trace"][0]["phase"] == "seed_search"
    assert res["trace"][1]["tool_calls"][1].get("skipped_duplicate") is True


def test_parse_final_strips_turn_suffix_and_synthesis_omits_tool_prompt(monkeypatch):
    parsed = agent.parse_final(
        json.dumps(
            {
                "answer": "Speaker D reported two to three Euro.",
                "citations": [{"meeting_id": "IS1008d#00076", "quote": "so if we add approximately two to three Euro per"}],
            }
        )
    )
    assert parsed is not None
    assert parsed["citations"][0]["meeting_id"] == "IS1008d"

    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")
    mock = MockLLM(
        [
            make_result(
                tool_calls=[{"id": "c1", "name": "search", "arguments": {"query": "LCD"}}]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            },
                        ],
                    }
                )
            ),
        ]
    )
    res = agent.run(conn, mock, "LCD screen speech recognition", ctx, max_steps=2, seed_search=True)
    assert res["ok"] is True
    # Final step (step 2 of max_steps=2) uses SYNTHESIS_SYSTEM_PROMPT and tools=None
    assert mock.calls[1]["tools"] is None
    assert "Prior tool calls" not in mock.calls[1]["messages"][1]["content"]
    assert "Do NOT call any tools" in mock.calls[1]["messages"][0]["content"]


def test_clean_unicode_quotes_fixes_nbhyphen_only_when_verified():
    conn = make_db()
    conn.execute("INSERT INTO meetings VALUES ('IS1002b','IS1002','b',2,1)")
    conn.execute(
        "INSERT INTO turns VALUES "
        "('IS1002b#00010','IS1002b','IS1002','b',10,'00:01:00','Speaker B',"
        "'Twenty five Euros is the selling price, we really have to innovate here.',0,0,13)"
    )
    ctx = tools.RunContext(team="IS1002", as_of=None, mode="answer")
    draft = {
        "text": "Price was twenty five Euros.",
        "citations": [
            {
                "meeting_id": "IS1002b",
                "quote": "Twenty\u2011five Euros is the selling price, we really have to innovate here.",
            },
            {
                "meeting_id": "IS1002b",
                "quote": "Paraphrased\u2011quote that does not exist anywhere in transcript",
            },
        ],
    }
    agent._clean_unicode_quotes(conn, ctx, draft)
    # Verified quote normalized to pure ASCII exact match
    assert draft["citations"][0]["quote"] == "Twenty five Euros is the selling price, we really have to innovate here."
    # Unverified quote is left untouched (so validator still rejects it!)
    assert "\u2011" in draft["citations"][1]["quote"]


def test_post_repair_prunes_unverified_extra_citation_when_verified_remain(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    bad_3rd = json.dumps(
        {
            "answer": "Team TS3006 dropped speech recognition and kept the LCD screen.",
            "citations": [
                {"meeting_id": "TS3006b", "quote": "We plan to include both an LCD screen and speech recognition in the remote."},
                {"meeting_id": "TS3006d", "quote": "So because of cost we drop the LCD screen and keep speech recognition."},
                {"meeting_id": "TS3006d", "quote": "slightly misquoted extra third citation not found verbatim"},
            ],
        }
    )
    mock = MockLLM(
        [
            make_result(tool_calls=[{"id": "c1", "name": "search", "arguments": {"query": "LCD speech"}}]),
            make_result(text=bad_3rd),
            # Repair returns same draft with 2 verified + 1 bad extra citation
            make_result(text=bad_3rd),
        ]
    )
    res = agent.run(conn, mock, "Which feature was kept?", ctx, max_steps=2, seed_search=True)
    assert res["ok"] is True
    assert len(res["draft"]["citations"]) == 2
    assert {c["meeting_id"] for c in res["draft"]["citations"]} == {"TS3006b", "TS3006d"}


def test_parallel_searches_do_not_starve_and_get_turns_prioritized(monkeypatch):
    conn = make_db()
    _patch_vec(monkeypatch)
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")

    mock = MockLLM(
        [
            # Parallel searches across b and d
            make_result(
                tool_calls=[
                    {"id": "c1", "name": "search", "arguments": {"query": "LCD", "meeting_letter": "b"}},
                    {"id": "c2", "name": "search", "arguments": {"query": "speech", "meeting_letter": "d"}},
                ]
            ),
            # Explicit get_turns around d#100
            make_result(
                tool_calls=[
                    {
                        "id": "c3",
                        "name": "get_turns",
                        "arguments": {"meeting_id": "TS3006d", "seq_start": 100, "seq_end": 101},
                    }
                ]
            ),
            make_result(
                text=json.dumps(
                    {
                        "answer": "Team TS3006 dropped the LCD screen and kept speech recognition.",
                        "citations": [
                            {
                                "meeting_id": "TS3006b",
                                "quote": "We plan to include both an LCD screen and speech recognition in the remote.",
                            },
                            {
                                "meeting_id": "TS3006d",
                                "quote": "So because of cost we drop the LCD screen and keep speech recognition.",
                            },
                        ],
                    }
                )
            ),
        ]
    )

    res = agent.run(conn, mock, "Which feature was kept?", ctx)
    assert res["ok"] is True
    assert "TS3006b#00010" in res["retrieved"]
    assert "TS3006d#00101" in res["retrieved"]
    # Step 3 prompt should contain the explicit get_turns turn TS3006d#00101
    assert "TS3006d#00101" in mock.calls[2]["messages"][1]["content"]


def test_extract_explicit_roles_only_self_identified_and_cleans_answer_unicode():
    conn = make_db()
    conn.execute("INSERT INTO meetings VALUES ('TS3006a','TS3006','a',1,2)")
    conn.execute(
        "INSERT INTO turns VALUES "
        "('TS3006a#00001','TS3006a','TS3006','a',1,'00:00:05','Speaker A',"
        "'Hello everyone, I am the Project Manager for this design.',0,0,10)"
    )
    conn.execute(
        "INSERT INTO turns VALUES "
        "('TS3006a#00002','TS3006a','TS3006','a',2,'00:00:10','Speaker C',"
        "'Okay, you are the Project Manager and I am the Industrial Designer.',0,0,12)"
    )
    ctx = tools.RunContext(team="TS3006", as_of=None, mode="answer")
    roles = agent._extract_explicit_roles(conn, ctx)
    assert roles["A"] == "Project Manager"
    assert roles["C"] == "Industrial Designer"

    draft = {
        "answer": "Team TS3006\u2019s scroll\u2011wheel was dropped.",
        "text": "Team TS3006\u2019s scroll\u2011wheel was dropped.",
        "citations": [],
    }
    agent._clean_unicode_quotes(conn, ctx, draft)
    assert draft["answer"] == "Team TS3006's scroll-wheel was dropped."
    assert draft["answer"].isascii()


def test_run_questions_force_preserves_verified_answer_on_all_providers_failed(tmp_path, monkeypatch):
    q_path = tmp_path / "questions.json"
    q_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question_id": "Q01",
                        "team": "TS3006",
                        "question": "Which feature was kept?",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "answers_work.json"
    prior_verified = {
        "results": {
            "Q01": {
                "question_id": "Q01",
                "team": "TS3006",
                "question": "Which feature was kept?",
                "ok": True,
                "draft": {
                    "text": "Team TS3006 kept the LCD screen.",
                    "citations": [
                        {"meeting_id": "TS3006d", "quote": "We kept the LCD screen for usability."}
                    ],
                },
            }
        }
    }
    out_path.write_text(json.dumps(prior_verified), encoding="utf-8")

    monkeypatch.setattr(agent.config, "QUESTIONS_PATH", q_path)
    monkeypatch.setattr(agent.store, "connect", lambda: make_db())

    class _AllFailLLM:
        def __init__(self, conn=None):
            pass

        def chat(self, *args, **kwargs):
            raise llm.AllProvidersFailed("all down", errors=[{"provider": "groq", "status": 429}])

    monkeypatch.setattr(agent.llm, "LLM", _AllFailLLM)
    _patch_vec(monkeypatch)

    summary = agent.run_questions(
        question_ids=["Q01"], out_path=out_path, force=True, retry_failed=False
    )
    assert summary["failed"] == ["Q01"]
    assert summary["passed"] == 0
    assert summary["passed_ids"] == []
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    # Prior verified answer is preserved rather than clobbered with draft=None
    assert saved["results"]["Q01"]["ok"] is True
    assert saved["results"]["Q01"]["draft"]["text"] == "Team TS3006 kept the LCD screen."
    assert "all down" in saved["results"]["Q01"]["last_rerun_error"]
