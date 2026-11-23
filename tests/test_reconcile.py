import sqlite3

import pytest

from catalyst import llm, reconcile, status, store, tools


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    meetings = [
        ("TS3012a", "TS3012", "a", 1, 3),
        ("TS3012b", "TS3012", "b", 2, 2),
        ("TS3012c", "TS3012", "c", 3, 2),
        ("TS3012d", "TS3012", "d", 4, 1),
    ]
    conn.executemany("INSERT INTO meetings VALUES (?,?,?,?,?)", meetings)

    turns = [
        (
            "TS3012a#00000",
            "TS3012a",
            "TS3012",
            "a",
            0,
            "00:00:01",
            "Speaker A",
            "We will use a monochrome screen for the remote control design",
            0,
            0,
            11,
        ),
        (
            "TS3012a#00001",
            "TS3012a",
            "TS3012",
            "a",
            1,
            "00:00:05",
            "Speaker B",
            "I will prepare the first prototype for the next review",
            0,
            0,
            10,
        ),
        (
            "TS3012a#00002",
            "TS3012a",
            "TS3012",
            "a",
            2,
            "00:00:09",
            "Speaker C",
            "I think the screen should stay monochrome for now",
            0,
            0,
            9,
        ),
        (
            "TS3012b#00000",
            "TS3012b",
            "TS3012",
            "b",
            0,
            "00:00:01",
            "Speaker A",
            "We will change the screen to color instead of monochrome",
            0,
            0,
            10,
        ),
        (
            "TS3012b#00001",
            "TS3012b",
            "TS3012",
            "b",
            1,
            "00:00:05",
            "Speaker B",
            "I finished the prototype and it is done now",
            0,
            0,
            9,
        ),
        (
            "TS3012c#00000",
            "TS3012c",
            "TS3012",
            "c",
            0,
            "00:00:01",
            "Speaker A",
            "We resolved the screen disagreement and agree on color",
            0,
            0,
            9,
        ),
        (
            "TS3012c#00001",
            "TS3012c",
            "TS3012",
            "c",
            1,
            "00:00:05",
            "Speaker C",
            "The final battery will be a standard AA cell",
            0,
            0,
            9,
        ),
        (
            "TS3012d#00000",
            "TS3012d",
            "TS3012",
            "d",
            0,
            "00:00:01",
            "Speaker A",
            "We will change the battery to a rechargeable pack next time",
            0,
            0,
            10,
        ),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)

    decisions = [
        (
            "d1",
            "TS3012",
            "screen",
            "screen",
            "Use a monochrome screen",
            "TS3012a",
            "a",
            "TS3012a#00000",
            "A",
            "We will use a monochrome screen for the remote control design",
            "test",
            "test",
            "2026-01-01",
        ),
        (
            "d2",
            "TS3012",
            "screen",
            "screen",
            "Change the screen to color",
            "TS3012b",
            "b",
            "TS3012b#00000",
            "A",
            "We will change the screen to color instead of monochrome",
            "test",
            "test",
            "2026-01-01",
        ),
        (
            "d3",
            "TS3012",
            "battery",
            "battery",
            "Use a standard AA battery",
            "TS3012c",
            "c",
            "TS3012c#00001",
            "C",
            "The final battery will be a standard AA cell",
            "test",
            "test",
            "2026-01-01",
        ),
        (
            "d4",
            "TS3012",
            "power",
            "power",
            "Use a rechargeable pack",
            "TS3012d",
            "d",
            "TS3012d#00000",
            "A",
            "We will change the battery to a rechargeable pack next time",
            "test",
            "test",
            "2026-01-01",
        ),
    ]
    conn.executemany("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", decisions)

    actions = [
        (
            "a1",
            "TS3012",
            "B",
            "Prepare the first prototype",
            "TS3012a",
            "a",
            "TS3012a#00001",
            "I will prepare the first prototype for the next review",
            "test",
            "test",
            "2026-01-01",
        )
    ]
    conn.executemany("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?)", actions)

    disagreements = [
        (
            "g1",
            "TS3012",
            "screen",
            "C",
            "The screen should stay monochrome",
            "TS3012a",
            "a",
            "TS3012a#00002",
            "I think the screen should stay monochrome for now",
            "test",
            "test",
            "2026-01-01",
        )
    ]
    conn.executemany(
        "INSERT INTO disagreements VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        disagreements,
    )
    return conn


def result() -> llm.LLMResult:
    return llm.LLMResult(
        text="",
        tool_calls=[],
        usage={},
        provider="test",
        model="test",
    )


def valid_proposals() -> dict:
    return {
        "topics": [
            {
                "topic": "screen",
                "topic_key": "screen",
                "decision_ids": ["d1", "d2"],
            },
            {
                "topic": "battery",
                "topic_key": "battery",
                "decision_ids": ["d3"],
            },
        ],
        "supersedes": [
            {
                "newer_id": "d2",
                "older_id": "d1",
                "quote": "We will change the screen to color instead of monochrome",
            }
        ],
        "action_closures": [
            {
                "action_id": "a1",
                "quote": "I finished the prototype and it is done now",
            }
        ],
        "disagreement_resolutions": [
            {
                "disagreement_id": "g1",
                "quote": "We resolved the screen disagreement and agree on color",
            }
        ],
    }


def test_scope_as_of():
    assert reconcile.scope_as_of("all") is None
    assert reconcile.scope_as_of("asof_c") == "c"
    with pytest.raises(ValueError):
        reconcile.scope_as_of("bad")


def test_records_for_scope_excludes_d_for_asof_c():
    conn = make_db()
    all_ids = {record["id"] for record in reconcile.records_for_scope(conn, "TS3012", "all")}
    asof_ids = {
        record["id"] for record in reconcile.records_for_scope(conn, "TS3012", "asof_c")
    }
    assert "d4" in all_ids
    assert "d4" not in asof_ids
    assert {"d1", "d2", "d3", "a1", "g1"} <= asof_ids


def test_apply_valid_proposals_and_statuses():
    conn = make_db()
    counts = reconcile.apply_proposals(
        conn, "TS3012", "all", valid_proposals(), result()
    )
    assert counts["topics"] == 4
    assert counts["supersedes"] == 1
    assert counts["action_closures"] == 1
    assert counts["disagreement_resolutions"] == 1
    assert counts["drops"] == 0

    ctx = tools.RunContext(team="TS3012", as_of=None, mode="answer")
    assert status.decision_status(conn, ctx, "d1", "TS3012", "a", "screen") == "superseded"
    assert status.decision_status(conn, ctx, "d2", "TS3012", "b", "screen") == "current"
    assert status.decision_status(conn, ctx, "d3", "TS3012", "c", "battery") == "current"
    assert status.action_status(conn, ctx, "a1", "a") == "done"
    assert status.disagreement_status(conn, ctx, "g1", "a") == "resolved"


def test_invalid_edges_are_dropped():
    conn = make_db()
    proposals = {
        "topics": [],
        "supersedes": [
            {
                "newer_id": "d1",
                "older_id": "d2",
                "quote": "We will change the screen to color instead of monochrome",
            },
            {
                "newer_id": "d2",
                "older_id": "d1",
                "quote": "color instead",
            },
            {
                "newer_id": "d2",
                "older_id": "d1",
                "quote": "We will use a monochrome screen for the remote control design",
            },
            {
                "newer_id": "nope",
                "older_id": "d1",
                "quote": "We will change the screen to color instead of monochrome",
            },
        ],
        "action_closures": [
            {
                "action_id": "a1",
                "quote": "I will prepare the first prototype for the next review",
            },
            {"action_id": "nope", "quote": "I finished the prototype and it is done now"},
        ],
        "disagreement_resolutions": [
            {
                "disagreement_id": "g1",
                "quote": "I think the screen should stay monochrome for now",
            },
            {"disagreement_id": "nope", "quote": "We resolved the screen disagreement and agree on color"},
        ],
    }
    counts = reconcile.apply_proposals(conn, "TS3012", "all", proposals, result())
    assert counts["supersedes"] == 0
    assert counts["action_closures"] == 0
    assert counts["disagreement_resolutions"] == 0
    assert counts["drops"] > 0
    assert conn.execute("SELECT COUNT(*) FROM decision_supersedes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM action_closed_by").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM disagreement_resolved_by").fetchone()[0] == 0


def test_supersede_uses_newer_quote_when_missing():
    conn = make_db()
    proposals = {
        "topics": [],
        "supersedes": [{"newer_id": "d2", "older_id": "d1"}],
        "action_closures": [],
        "disagreement_resolutions": [],
    }
    counts = reconcile.apply_proposals(conn, "TS3012", "all", proposals, result())
    assert counts["supersedes"] == 1


def test_edge_aliases_are_supported():
    conn = make_db()
    proposals = {
        "topics": [],
        "supersedes": [
            {
                "later_id": "d2",
                "earlier_id": "d1",
                "quote": "We will change the screen to color instead of monochrome",
            }
        ],
        "action_closures": [
            {
                "earlier_id": "a1",
                "quote": "I finished the prototype and it is done now",
            }
        ],
        "disagreement_resolutions": [
            {
                "earlier_id": "g1",
                "quote": "We resolved the screen disagreement and agree on color",
            }
        ],
    }
    counts = reconcile.apply_proposals(conn, "TS3012", "all", proposals, result())
    assert counts["supersedes"] == 1
    assert counts["action_closures"] == 1
    assert counts["disagreement_resolutions"] == 1


def test_groq_topic_variants_are_normalized():
    conn = make_db()
    proposals = {
        "topics": [
            ["d1", "d2"],
            {"topic": ["d3"]},
            {"name": "power", "records": ["d4", "a1", "g1"]},
        ],
        "supersedes": [],
        "action_closures": [],
        "disagreement_resolutions": [],
    }
    counts = reconcile.apply_proposals(conn, "TS3012", "all", proposals, result())
    assert counts["topics"] == 4
    rows = {
        row["decision_id"]: (row["topic_key"], row["topic"])
        for row in conn.execute(
            "SELECT decision_id, topic_key, topic FROM topic_key_map "
            "WHERE team = 'TS3012' AND scope = 'all'"
        )
    }
    assert rows["d1"] == ("screen", "screen")
    assert rows["d2"] == ("screen", "screen")
    assert rows["d3"] == ("battery", "battery")
    assert rows["d4"] == ("power", "power")


def test_delete_then_insert_is_scoped():
    conn = make_db()
    reconcile.apply_proposals(conn, "TS3012", "all", valid_proposals(), result())
    reconcile.apply_proposals(conn, "TS3012", "asof_c", {}, result())

    all_topics = conn.execute(
        "SELECT COUNT(*) FROM topic_key_map WHERE scope = 'all'"
    ).fetchone()[0]
    asof_topics = conn.execute(
        "SELECT COUNT(*) FROM topic_key_map WHERE scope = 'asof_c'"
    ).fetchone()[0]
    assert all_topics == 4
    assert asof_topics == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_supersedes WHERE scope = 'all'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_supersedes WHERE scope = 'asof_c'"
    ).fetchone()[0] == 0

    reconcile.apply_proposals(
        conn, "TS3012", "asof_c", valid_proposals(), result()
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_supersedes WHERE scope = 'all'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_supersedes WHERE scope = 'asof_c'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM action_closed_by WHERE scope = 'asof_c'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM disagreement_resolved_by WHERE scope = 'asof_c'"
    ).fetchone()[0] == 1


def test_decision_status_uses_reconciled_topic_key_map():
    conn = make_db()
    # d3 has raw topic_key='battery', d4 has raw topic_key='power'.
    # Reconcile clusters both into 'power_source' in topic_key_map.
    proposals = {
        "topics": [
            {
                "topic": "power source",
                "topic_key": "power_source",
                "decision_ids": ["d3", "d4"],
            }
        ],
        "supersedes": [],
        "action_closures": [],
        "disagreement_resolutions": [],
    }
    reconcile.apply_proposals(conn, "TS3012", "all", proposals, result())

    ctx_all = tools.RunContext(team="TS3012", as_of=None, mode="answer")
    assert status.decision_status(conn, ctx_all, "d3", "TS3012", "c", "battery") == "superseded"
    assert status.decision_status(conn, ctx_all, "d4", "TS3012", "d", "power") == "current"

    ledger_decs = {
        r["id"]: r for r in tools.ledger(conn, ctx_all, kind="decisions")
    }
    assert ledger_decs["d3"]["status"] == "superseded"
    assert ledger_decs["d4"]["status"] == "current"
    assert ledger_decs["d3"]["topic_key"] == "power_source"


def test_asof_c_reconcile_prompts_never_contain_d():
    conn = make_db()
    recs = reconcile.records_for_scope(conn, "TS3012", "asof_c")
    decs = [r for r in recs if r["kind"] == "decision"]
    acts = [r for r in recs if r["kind"] == "action"]
    diss = [r for r in recs if r["kind"] == "disagreement"]
    later = reconcile._later_evidence_lines(decs, [], "a")
    prompts = [
        reconcile._decision_prompt("TS3012", "asof_c", reconcile._decision_lines(decs)),
        reconcile._supersede_prompt("TS3012", "asof_c", reconcile._decision_lines(decs)),
        reconcile._action_prompt("TS3012", "asof_c", reconcile._action_lines(acts), later),
        reconcile._disagreement_prompt("TS3012", "asof_c", reconcile._disagreement_lines(diss), later),
    ]
    for p in prompts:
        assert "TS3012d" not in p
        assert "d4" not in p
        assert "rechargeable pack" not in p
