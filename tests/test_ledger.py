import sqlite3

from catalyst import ledger, llm, store


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3012c','TS3012','c',3,3)")
    turns = [
        (
            "TS3012c#00000",
            "TS3012c",
            "TS3012",
            "c",
            0,
            "00:00:01",
            "Speaker A",
            "We decided to use the blue design for the final product",
            0,
            0,
            11,
        ),
        (
            "TS3012c#00001",
            "TS3012c",
            "TS3012",
            "c",
            1,
            "00:00:05",
            "Speaker B",
            "I think we should use the red design instead of blue",
            0,
            0,
            11,
        ),
        (
            "TS3012c#00002",
            "TS3012c",
            "TS3012",
            "c",
            2,
            "00:00:09",
            "Speaker C",
            "The final budget is twenty five Euro for the whole unit",
            0,
            0,
            11,
        ),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)
    conn.executemany(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        [(turn[7], turn[0], turn[1], turn[2]) for turn in turns],
    )
    return conn


def make_window(conn: sqlite3.Connection, turn_ids: list[str]) -> dict:
    rows = [
        dict(conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone())
        for turn_id in turn_ids
    ]
    return {
        "index": 0,
        "turns": rows,
        "turn_ids": set(turn_ids),
        "text": "\n".join(
            f'[{row["turn_id"]}] {row["speaker"]}: {row["text"]}' for row in rows
        ),
    }


def test_normalize_topic():
    assert ledger.normalize_topic("LCD screen / cost") == "lcd_screen_cost"
    assert ledger.normalize_topic("") == "general"


def test_build_windows_has_overlap():
    turns = [
        {
            "turn_id": f"M#{index:05d}",
            "speaker": "Speaker A",
            "text": "word " * 10,
        }
        for index in range(6)
    ]
    windows = ledger.build_windows(turns, max_tokens=40, overlap_tokens=20)
    assert len(windows) > 1
    assert windows[0]["turn_ids"] & windows[1]["turn_ids"]


def test_store_window_keeps_valid_and_drops_invalid():
    conn = make_db()
    meeting = dict(
        conn.execute(
            "SELECT * FROM meetings WHERE meeting_id = 'TS3012c'"
        ).fetchone()
    )
    window = make_window(conn, ["TS3012c#00000", "TS3012c#00001"])
    data = {
        "decisions": [
            {
                "topic": "design",
                "decision": "use blue",
                "turn_id": "TS3012c#00000",
                "quote": "We decided to use the blue design for the final product",
            },
            {
                "topic": "bad",
                "decision": "bad",
                "turn_id": "TS3012c#00001",
                "quote": "blue design",
            },
        ],
        "actions": [],
        "disagreements": [],
        "figures": [],
    }
    result = llm.LLMResult(
        text="",
        tool_calls=[],
        usage={},
        provider="test",
        model="test",
    )

    counts = ledger.store_window(conn, meeting, window, data, result)

    assert counts["decisions"] == 1
    assert counts["drops"] == 1
    assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ledger_drops").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM topic_key_map").fetchone()[0] == 2


def test_store_window_rejects_quote_outside_window():
    conn = make_db()
    meeting = dict(
        conn.execute(
            "SELECT * FROM meetings WHERE meeting_id = 'TS3012c'"
        ).fetchone()
    )
    window = make_window(conn, ["TS3012c#00000", "TS3012c#00001"])
    data = {
        "decisions": [
            {
                "topic": "budget",
                "decision": "budget is twenty five Euro",
                "turn_id": "TS3012c#00002",
                "quote": "The final budget is twenty five Euro for the whole unit",
            }
        ],
        "actions": [],
        "disagreements": [],
        "figures": [],
    }
    result = llm.LLMResult(
        text="",
        tool_calls=[],
        usage={},
        provider="test",
        model="test",
    )

    counts = ledger.store_window(conn, meeting, window, data, result)

    assert counts["decisions"] == 0
    assert counts["drops"] == 1
    reason = conn.execute("SELECT reason FROM ledger_drops").fetchone()[0]
    assert reason == "quote_not_in_window"


def test_question_teams_include_brief_team():
    teams = ledger._question_teams()
    assert len(teams) == 24
    assert "TS3012" in teams
