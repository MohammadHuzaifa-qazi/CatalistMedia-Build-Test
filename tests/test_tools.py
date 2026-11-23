import sqlite3

import pytest

from catalyst import config, store, tools


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


def test_brief_scope_never_returns_d(monkeypatch):
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")

    # Search tests exercise scoping, not the real Chroma index.
    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    with pytest.raises(tools.ScopeError):
        tools.get_turns(conn, ctx, "TS3012d")

    hits = tools.search(conn, ctx, "changed the design to red", k=10)
    assert all(not hit["turn_id"].startswith("TS3012d") for hit in hits)

    with pytest.raises(tools.ScopeError):
        tools.search(conn, ctx, "red", meeting_letter="d")

    assert tools.ledger(conn, ctx, kind="all") == []


def test_eval_mode_allows_none_team(monkeypatch):
    conn = make_db()
    ctx = tools.RunContext(team=None, as_of=None, mode="eval")

    monkeypatch.setattr(tools, "_vector_search", lambda *args, **kwargs: [])

    hits = tools.search(conn, ctx, "blue design", k=10)
    assert any(hit["turn_id"] == "TS3012c#00000" for hit in hits)


def test_vector_search_failure_propagates(monkeypatch):
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="answer")

    def boom(*args, **kwargs):
        raise tools.VectorSearchError("missing collection")

    monkeypatch.setattr(tools, "_vector_search", boom)

    with pytest.raises(tools.VectorSearchError):
        tools.search(conn, ctx, "blue design")


def test_vector_search_wraps_missing_dependency(monkeypatch):
    monkeypatch.setattr(tools, "_collection_cache", None)
    monkeypatch.setattr(tools, "_embedder_cache", None)

    def missing():
        raise ImportError("No module named 'chromadb'")

    monkeypatch.setattr(tools, "_collection", missing)

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="answer")
    with pytest.raises(tools.VectorSearchError) as excinfo:
        tools._vector_search(ctx, "blue", "TS3012", 8)

    message = str(excinfo.value)
    assert config.CHROMA_COLLECTION in message
    assert "ImportError" in message
    assert "python -m catalyst.pipeline embed" in message


def test_vector_search_empty_collection_is_loud(monkeypatch):
    class EmptyCollection:
        def count(self):
            return 0

    monkeypatch.setattr(tools, "_collection_cache", EmptyCollection())
    monkeypatch.setattr(tools, "_embedder_cache", None)

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="answer")
    with pytest.raises(tools.VectorSearchError) as excinfo:
        tools._vector_search(ctx, "blue", "TS3012", 8)

    assert "empty" in str(excinfo.value)
    assert "python -m catalyst.pipeline embed" in str(excinfo.value)


def test_vector_search_returns_ids(monkeypatch):
    class Vector:
        def tolist(self):
            return [0.1, 0.2]

    class FakeEmbedder:
        def embed(self, texts, batch_size):
            assert texts == ["blue"]
            return [Vector()]

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            assert kwargs["where"] == {
                "$and": [{"team": "TS3012"}, {"letter": {"$in": ["a", "b", "c"]}}]
            }
            return {"ids": [["TS3012c#00000"]]}

    monkeypatch.setattr(tools, "_collection_cache", FakeCollection())
    monkeypatch.setattr(tools, "_embedder_cache", FakeEmbedder())

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="answer")
    hits = tools._vector_search(ctx, "blue", "TS3012", 8)

    assert hits == [{"turn_id": "TS3012c#00000"}]


def test_build_chroma_where_filter_combinations():
    assert tools._build_chroma_where("TS3012", None, None) == {"team": "TS3012"}
    assert tools._build_chroma_where("TS3012", "c", None) == {
        "$and": [{"team": "TS3012"}, {"letter": {"$in": ["a", "b", "c"]}}]
    }
    assert tools._build_chroma_where("TS3012", "c", "b") == {
        "$and": [{"team": "TS3012"}, {"letter": "b"}]
    }
    assert tools._build_chroma_where("TS3012", "a", None) == {
        "$and": [{"team": "TS3012"}, {"letter": "a"}]
    }
    assert tools._build_chroma_where(None, None, None) is None
    assert tools._build_chroma_where(None, "c", None) == {
        "letter": {"$in": ["a", "b", "c"]}
    }


def test_brief_scope_adversarial_all_tools_never_return_d(monkeypatch):
    conn = make_db()
    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")

    # Add a decision in meeting d to verify ledger never returns it either.
    conn.execute(
        "INSERT INTO decisions VALUES "
        "('dec_d','TS3012','color','color','Change to red','TS3012d','d',"
        "'TS3012d#00000','B','We changed the design to red for the final product',"
        "'test','test','2026-01-01')"
    )

    seen_where = []

    class Vector:
        def tolist(self):
            return [0.1, 0.2]

    class FakeEmbedder:
        def embed(self, texts, batch_size):
            return [Vector()]

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            seen_where.append(kwargs["where"])
            # Filter simulated items using the passed where clause
            allowed = kwargs["where"]["$and"][1]["letter"]["$in"]
            ids = ["TS3012c#00000"] if "c" in allowed else []
            if "d" in allowed:
                ids.insert(0, "TS3012d#00000")
            return {"ids": [ids]}

    monkeypatch.setattr(tools, "_collection_cache", FakeCollection())
    monkeypatch.setattr(tools, "_embedder_cache", FakeEmbedder())

    # 1. search with query targeting d text and mentioning TS3012d
    hits = tools.execute_tool(
        conn, ctx, "search", {"query": "TS3012d changed the design to red", "k": 10}
    )
    assert seen_where == [
        {"$and": [{"team": "TS3012"}, {"letter": {"$in": ["a", "b", "c"]}}]}
    ]
    assert all(not h["turn_id"].startswith("TS3012d") for h in hits)

    # 2. search explicitly requesting letter d -> ScopeError
    with pytest.raises(tools.ScopeError):
        tools.execute_tool(conn, ctx, "search", {"query": "red", "meeting_letter": "d"})

    # 3. get_turns on TS3012d -> ScopeError
    with pytest.raises(tools.ScopeError):
        tools.execute_tool(conn, ctx, "get_turns", {"meeting_id": "TS3012d"})

    # 4. ledger never includes d
    recs = tools.execute_tool(conn, ctx, "ledger", {"kind": "all"})
    assert all(not r["turn_id"].startswith("TS3012d") for r in recs)

    # 5. verify on TS3012d fails with out_of_scope
    vr = tools.execute_tool(
        conn,
        ctx,
        "verify",
        {
            "meeting_id": "TS3012d",
            "quote": "We changed the design to red for the final product",
        },
    )
    assert not vr.ok and vr.reason == "out_of_scope"


def test_trim_turn_text_best_span_and_exact_verify():
    conn = make_db()
    prefix = " ".join(f"filler{i}" for i in range(220))
    needle = "We definitely chose titanium because it resists scratches nicely."
    suffix = " ".join(f"ending{i}" for i in range(180))
    long_text = f"{prefix} {needle} {suffix}"

    conn.execute(
        "INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "TS3012c#00099",
            "TS3012c",
            "TS3012",
            "c",
            99,
            "00:10:00",
            "Speaker A",
            long_text,
            0,
            0,
            len(long_text.split()),
        ),
    )

    trimmed, truncated = tools.trim_turn_text(
        long_text, query="titanium resists scratches", max_words=150
    )
    assert truncated is True
    assert len(trimmed.split()) == 150
    assert needle in trimmed
    assert trimmed in long_text

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")
    vr = tools.verify(conn, ctx, "TS3012c", needle)
    assert vr.ok and vr.turn_id == "TS3012c#00099"


def test_get_turns_caps_count_and_words():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)
    conn.execute("INSERT INTO meetings VALUES ('TS3012c','TS3012','c',3,40)")

    body_220 = " ".join(f"word{i}" for i in range(220))
    rows = [
        (
            f"TS3012c#{i:05d}",
            "TS3012c",
            "TS3012",
            "c",
            i,
            f"00:00:{i:02d}",
            "Speaker A",
            body_220,
            0,
            0,
            220,
        )
        for i in range(40)
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="brief")
    # Also verify execute_tool ignores attempts to raise max_turns/max_words_per_turn
    turns = tools.execute_tool(
        conn,
        ctx,
        "get_turns",
        {"meeting_id": "TS3012c", "max_turns": 999, "max_words_per_turn": 999},
    )
    assert 1 <= len(turns) <= config.GET_TURNS_MAX_TURNS
    assert all(t["truncated"] is True for t in turns)
    assert all(t["shown_words"] <= config.MAX_DISPLAY_TURN_WORDS for t in turns)
    assert sum(t["shown_words"] for t in turns) <= config.GET_TURNS_MAX_WORDS
    assert all(t["turn_id"].startswith("TS3012c#") for t in turns)

    # Also verify execute_tool accepts explicit None for optional params (as emitted by Groq)
    turns_none = tools.execute_tool(
        conn,
        ctx,
        "get_turns",
        {"meeting_id": "TS3012c", "seq_start": 0, "seq_end": 1, "speaker": None, "query": None},
    )
    assert len(turns_none) == 2

    # Invalid meeting_letter (e.g. passing team ID) raises ValueError instead of silent 0 hits
    with pytest.raises(ValueError):
        tools.execute_tool(
            conn,
            ctx,
            "search",
            {"query": "word0", "meeting_letter": "TS3012"},
        )


def test_answer_and_brief_require_team():
    with pytest.raises(ValueError):
        tools.RunContext(team=None, mode="answer")
    with pytest.raises(ValueError):
        tools.RunContext(team=None, mode="brief")


def test_vector_search_failure_increments_counter_and_logs(monkeypatch, caplog):
    tools.reset_vector_search_failures()
    monkeypatch.setattr(tools, "_collection_cache", None)
    monkeypatch.setattr(tools, "_embedder_cache", None)

    def broken():
        raise RuntimeError("chroma index corrupted")

    monkeypatch.setattr(tools, "_collection", broken)

    ctx = tools.RunContext(team="TS3012", as_of="c", mode="answer")
    with caplog.at_level("ERROR", logger="catalyst.tools"):
        with pytest.raises(tools.VectorSearchError):
            tools._vector_search(ctx, "blue", "TS3012", 8)

    assert tools.VECTOR_SEARCH_FAILURES == 1
    assert any("vector search failed" in rec.message for rec in caplog.records)

    class EmptyCollection:
        def count(self):
            return 0

    monkeypatch.setattr(tools, "_collection_cache", EmptyCollection())
    with caplog.at_level("ERROR", logger="catalyst.tools"):
        with pytest.raises(tools.VectorSearchError):
            tools._vector_search(ctx, "blue", "TS3012", 8)

    assert tools.VECTOR_SEARCH_FAILURES == 2
    assert any("is empty" in rec.message for rec in caplog.records)
