import sqlite3

import pytest

from catalyst import store, tools, validate


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3012c','TS3012','c',3,2)")
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
            "TS3012c#00001",
            "TS3012c",
            "TS3012",
            "c",
            1,
            "00:00:05",
            "Speaker B",
            "We will use the blue design for the final product",
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
            "Speaker C",
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


def test_sentence_counter_decimals():
    assert validate.count_sentences("The total is 2.3 Euro.") == 1
    assert validate.count_sentences("It costs 2.3 Euro. We agreed.") == 2


def test_sentence_counter_abbreviations():
    assert validate.count_sentences("Mr. Smith agreed. We will proceed.") == 2
    assert validate.count_sentences("Use e.g. this one.") == 1


def test_verify_quote_word_boundary_and_scope():
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")

    ok = validate.verify_quote(
        conn,
        ctx,
        "TS3012c",
        "We decided to use the blue design for the final product",
    )
    assert ok.ok and ok.turn_id == "TS3012c#00000"

    too_short = validate.verify_quote(conn, ctx, "TS3012c", "blue design")
    assert not too_short.ok and too_short.reason == "too_short"

    partial = validate.verify_quote(
        conn, ctx, "TS3012c", "decided to use the blue desig"
    )
    assert not partial.ok and partial.reason == "not_found"

    out_of_scope = validate.verify_quote(
        conn,
        ctx,
        "TS3012d",
        "We changed the design to red for the final product",
    )
    assert not out_of_scope.ok and out_of_scope.reason == "out_of_scope"


def test_validate_brief_rules():
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")

    good = {
        "raw_text": 'Decision [TS3012c: "We decided to use the blue design for the final product"]'
    }
    ok, errors = validate.validate_brief(conn, ctx, good)
    assert ok, errors

    uncited = {"raw_text": "Decision without citation"}
    ok, errors = validate.validate_brief(conn, ctx, uncited)
    assert not ok and "uncited_bullet" in errors

    d_ref = {
        "raw_text": 'Decision [TS3012d: "We changed the design to red for the final product"]'
    }
    ok, errors = validate.validate_brief(conn, ctx, d_ref)
    assert not ok and "asof_violation" in errors

    predictive = {"raw_text": "We will decide the final design next meeting."}
    ok, errors = validate.validate_brief(conn, ctx, predictive)
    assert not ok and "predicts_final_meeting" in errors

    quoted_commitment = {
        "raw_text": 'Commitment [TS3012c: "We will use the blue design for the final product"]'
    }
    ok, errors = validate.validate_brief(conn, ctx, quoted_commitment)
    assert ok, errors
