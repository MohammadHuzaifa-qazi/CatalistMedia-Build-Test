import json
import sqlite3

from catalyst import brief, config, llm, reconcile, store, tools, validate


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    meetings = [
        ("TS3012a", "TS3012", "a", 1, 2),
        ("TS3012b", "TS3012", "b", 2, 3),
        ("TS3012c", "TS3012", "c", 3, 2),
        ("TS3012d", "TS3012", "d", 4, 1),
    ]
    conn.executemany("INSERT INTO meetings VALUES (?,?,?,?,?)", meetings)

    turns = [
        ("TS3012a#00000", "TS3012a", "TS3012", "a", 0, "00:00:01", "Speaker A",
         "Okay good morning I'll be your Project Manager for this project.", 0, 0, 11),
        ("TS3012a#00001", "TS3012a", "TS3012", "a", 1, "00:00:10", "Speaker A",
         "The project I got was just for a standard TV remote control.", 0, 0, 12),
        ("TS3012b#00000", "TS3012b", "TS3012", "b", 0, "00:00:01", "Speaker C",
         "We decided to go for a universal remote control instead of TV only.", 0, 0, 13),
        ("TS3012b#00001", "TS3012b", "TS3012", "b", 1, "00:00:05", "Speaker B",
         "I will put some design ideas on paper and present them later.", 0, 0, 12),
        ("TS3012b#00002", "TS3012b", "TS3012", "b", 2, "00:00:10", "Speaker D",
         "I think the screen should be placed at the top of the remote.", 0, 0, 13),
        ("TS3012c#00000", "TS3012c", "TS3012", "c", 0, "00:00:01", "Speaker A",
         "Okay so we are definitely going for rubber for the case material.", 0, 0, 12),
        ("TS3012c#00001", "TS3012c", "TS3012", "c", 1, "00:00:05", "Speaker C",
         "I disagree and think the remote should be yellow for young people.", 0, 0, 12),
        ("TS3012d#00000", "TS3012d", "TS3012", "d", 0, "00:00:01", "Speaker A",
         "In the final meeting we dropped yellow and chose a black cover.", 0, 0, 12),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)

    decisions = [
        ("d1", "TS3012", "product_type", "product_type", "Standard TV remote", "TS3012a", "a",
         "TS3012a#00001", "A", "The project I got was just for a standard TV remote control.", "t", "t", "2026-01-01"),
        ("d2", "TS3012", "product_type", "product_type", "Universal remote control", "TS3012b", "b",
         "TS3012b#00000", "C", "We decided to go for a universal remote control instead of TV only.", "t", "t", "2026-01-01"),
        ("d3", "TS3012", "material", "material", "Rubber case material", "TS3012c", "c",
         "TS3012c#00000", "A", "Okay so we are definitely going for rubber for the case material.", "t", "t", "2026-01-01"),
        ("d4", "TS3012", "color", "color", "Black cover in meeting d", "TS3012d", "d",
         "TS3012d#00000", "A", "In the final meeting we dropped yellow and chose a black cover.", "t", "t", "2026-01-01"),
    ]
    conn.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", decisions)

    conn.execute(
        "INSERT INTO actions VALUES "
        "('a1','TS3012','B','Put design ideas on paper','TS3012b','b','TS3012b#00001',"
        "'I will put some design ideas on paper and present them later.','t','t','2026-01-01')"
    )
    conn.execute(
        "INSERT INTO disagreements VALUES "
        "('g1','TS3012','color','C','Remote should be yellow','TS3012c','c','TS3012c#00001',"
        "'I disagree and think the remote should be yellow for young people.','t','t','2026-01-01')"
    )

    reconcile.apply_proposals(
        conn,
        "TS3012",
        "asof_c",
        {
            "topics": [{"topic": "product_type", "topic_key": "product_type", "decision_ids": ["d1", "d2"]}],
            "supersedes": [{"newer_id": "d2", "older_id": "d1"}],
            "action_closures": [],
            "disagreement_resolutions": [],
        },
        None,
    )
    return conn


class MockBriefLLM:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs) -> llm.LLMResult:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        payload = self.payloads.pop(0)
        return llm.LLMResult(
            text=json.dumps(payload),
            tool_calls=[],
            usage={"prompt_tokens": 200, "completion_tokens": 80, "thoughts_tokens": 0, "total_tokens": 280, "billed_output_tokens": 80},
            provider="mock",
            model="mock-brief",
        )


def test_brief_four_section_calls_under_budget_zero_d_and_validates():
    conn = make_db()
    mock = MockBriefLLM(
        [
            # decided
            {
                "bullets": [
                    {
                        "summary": "The team selected rubber for the remote control case.",
                        "citations": [
                            {"meeting_id": "TS3012c", "quote": "Okay so we are definitely going for rubber for the case material."}
                        ],
                    }
                ]
            },
            # changed
            {
                "bullets": [
                    {
                        "summary": "The team shifted from a standard TV remote in meeting a to a universal remote control in meeting b.",
                        "citations": [
                            {"meeting_id": "TS3012a", "quote": "The project I got was just for a standard TV remote control."},
                            {"meeting_id": "TS3012b", "quote": "We decided to go for a universal remote control instead of TV only."},
                        ],
                    }
                ]
            },
            # open_actions
            {
                "bullets": [
                    {
                        "summary": "Speaker B has an open action to put design ideas on paper.",
                        "citations": [
                            {"meeting_id": "TS3012b", "quote": "I will put some design ideas on paper and present them later."}
                        ],
                    }
                ]
            },
            # open_disagreements
            {
                "bullets": [
                    {
                        "summary": "Speaker C prefers a yellow remote control for young users.",
                        "citations": [
                            {"meeting_id": "TS3012c", "quote": "I disagree and think the remote should be yellow for young people."}
                        ],
                    }
                ]
            },
        ]
    )

    res = brief.generate_brief(conn, mock, team="TS3012", as_of="c")
    assert res["ok"] is True, res["errors"]
    assert len(mock.calls) == 4
    # Every prompt stayed well under 6K tokens and NEVER contained TS3012d
    for call in mock.calls:
        combined = " ".join(m["content"] for m in call["messages"])
        assert len(combined) // 4 < config.PROMPT_BUDGET_TOKENS
        assert "TS3012d" not in combined
        assert "black cover" not in combined
    assert "TS3012d" not in res["brief"]
    assert '[TS3012c: "Okay so we are definitely going for rubber for the case material."]' in res["brief"]
    assert conn.execute("SELECT COUNT(*) FROM answer_runs WHERE kind = 'brief'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM citations_log WHERE question_id IS NULL AND verified = 1").fetchone()[0] >= 4


def test_brief_drops_predictive_and_d_bullets_and_repairs():
    conn = make_db()
    mock = MockBriefLLM(
        [
            # decided: first try has TS3012d + prediction -> dropped -> triggers 1 repair!
            {
                "bullets": [
                    {
                        "summary": "In the next meeting the team will decide on a black cover from TS3012d.",
                        "citations": [
                            {"meeting_id": "TS3012d", "quote": "In the final meeting we dropped yellow and chose a black cover."}
                        ],
                    }
                ]
            },
            # decided repair: valid!
            {
                "bullets": [
                    {
                        "summary": "The team selected rubber for the remote control case.",
                        "citations": [
                            {"meeting_id": "TS3012c", "quote": "Okay so we are definitely going for rubber for the case material."}
                        ],
                    }
                ]
            },
            # changed
            {
                "bullets": [
                    {
                        "summary": "Changed from a standard TV remote in meeting a to a universal remote in meeting b.",
                        "citations": [
                            {"meeting_id": "TS3012b", "quote": "We decided to go for a universal remote control instead of TV only."}
                        ],
                    }
                ]
            },
            # open_actions
            {
                "bullets": [
                    {
                        "summary": "Speaker B owns putting design ideas on paper.",
                        "citations": [
                            {"meeting_id": "TS3012b", "quote": "I will put some design ideas on paper and present them later."}
                        ],
                    }
                ]
            },
            # open_disagreements
            {
                "bullets": [
                    {
                        "summary": "Speaker C wants a yellow remote.",
                        "citations": [
                            {"meeting_id": "TS3012c", "quote": "I disagree and think the remote should be yellow for young people."}
                        ],
                    }
                ]
            },
        ]
    )

    res = brief.generate_brief(conn, mock, team="TS3012", as_of="c")
    assert res["ok"] is True, res["errors"]
    assert len(mock.calls) == 5  # 4 sections + 1 repair
    assert "TS3012d" not in res["brief"]
