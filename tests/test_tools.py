import sqlite3

import pytest

from catalyst import store, tools


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3012c','TS3012','c',3,1)")
    conn.execute("INSERT INTO meetings VALUES ('TS3012d','TS3012','d',4,1)")

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
            "TS3012d#00000",
            "TS3012d",
            "TS3012",
            "d",
            0,
            "00:00:01",
            "Speaker B",
            "We changed the design to red for the final product",
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


def test_brief_scope_never_returns_d():
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")

    with pytest.raises(tools.ScopeError):
        tools.get_turns(conn, ctx, "TS3012d")

    hits = tools.search(conn, ctx, "changed the design to red", k=10)
    assert all(not hit["turn_id"].startswith("TS3012d") for hit in hits)

    with pytest.raises(tools.ScopeError):
        tools.search(conn, ctx, "red", meeting_letter="d")

    assert tools.ledger(conn, ctx, kind="all") == []


def test_eval_mode_allows_none_team():
    conn = make_db()
    ctx = tools.RunContext(team=None, as_of=None, mode="eval")

    hits = tools.search(conn, ctx, "blue design", k=10)
    assert any(hit["turn_id"] == "TS3012c#00000" for hit in hits)


def test_answer_and_brief_require_team():
    with pytest.raises(ValueError):
        tools.RunContext(team=None, mode="answer")
    with pytest.raises(ValueError):
        tools.RunContext(team=None, mode="brief")
