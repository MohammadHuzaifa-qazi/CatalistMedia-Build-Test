import sqlite3

from catalyst import store


def test_sanitize_removes_fts_operators():
    query = store.sanitize_fts_query('hello "world" OR (x) -y')
    assert query == '"hello" "world" "OR" "x" "y"'


def test_fts_known_phrase():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    text = "Okay good morning This is our first team meeting"
    conn.execute("INSERT INTO meetings VALUES ('TS3012a','TS3012','a',1,1)")
    conn.execute(
        "INSERT INTO turns VALUES "
        "('TS3012a#00000','TS3012a','TS3012','a',0,'00:00:02','Speaker A',?,0,0,12)",
        (text,),
    )
    conn.execute(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        (text, "TS3012a#00000", "TS3012a", "TS3012"),
    )

    rows = store.fts_search(
        conn, "good morning This is our first", team="TS3012", phrase=True
    )
    assert any(row["turn_id"] == "TS3012a#00000" for row in rows)


def test_fts_as_of_pushed_before_limit():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3012c','TS3012','c',3,1)")
    conn.execute("INSERT INTO meetings VALUES ('TS3012d','TS3012','d',4,1)")
    turns = [
        ("TS3012c#00000", "TS3012c", "TS3012", "c", 0, "00:00:01", "Speaker A", "rubber case design", 0, 0, 3),
        ("TS3012d#00000", "TS3012d", "TS3012", "d", 0, "00:00:01", "Speaker B", "rubber rubber rubber case", 0, 0, 4),
    ]
    conn.executemany("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)", turns)
    conn.executemany(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        [(t[7], t[0], t[1], t[2]) for t in turns],
    )

    # Even with limit=1, TS3012d must NOT occupy the slot when as_of='c'.
    rows = store.fts_search(conn, "rubber", team="TS3012", as_of="c", limit=1)
    assert [r["turn_id"] for r in rows] == ["TS3012c#00000"]


def test_fts_multiword_or_fallback_and_acronym():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    conn.execute("INSERT INTO meetings VALUES ('TS3006d','TS3006','d',4,1)")
    text = "There are more people who like speech recognition than an L_C_D_ screen"
    conn.execute(
        "INSERT INTO turns VALUES "
        "('TS3006d#00100','TS3006d','TS3006','d',100,'00:05:00','Speaker D',?,0,0,13)",
        (text,),
    )
    conn.execute(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) VALUES (?,?,?,?)",
        (text, "TS3006d#00100", "TS3006d", "TS3006"),
    )

    rows = store.fts_search(conn, "LCD screen speech recognition cost", team="TS3006", limit=5)
    assert any(r["turn_id"] == "TS3006d#00100" for r in rows)
