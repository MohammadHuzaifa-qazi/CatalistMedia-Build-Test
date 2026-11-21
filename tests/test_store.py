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
