from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from . import config

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE meetings (
  meeting_id      TEXT PRIMARY KEY,
  team            TEXT NOT NULL,
  letter          TEXT NOT NULL,
  meeting_number  INTEGER,
  n_turns         INTEGER
);

CREATE TABLE turns (
  turn_id     TEXT PRIMARY KEY,
  meeting_id  TEXT NOT NULL REFERENCES meetings(meeting_id),
  team        TEXT NOT NULL,
  letter      TEXT NOT NULL,
  seq         INTEGER NOT NULL,
  timestamp   TEXT,
  speaker     TEXT NOT NULL,
  text        TEXT NOT NULL,
  char_start  INTEGER,
  char_end    INTEGER,
  n_words     INTEGER NOT NULL
);
CREATE INDEX idx_turns_meeting ON turns(meeting_id, seq);
CREATE INDEX idx_turns_team    ON turns(team, letter, seq);

CREATE VIRTUAL TABLE turns_fts USING fts5(
  text,
  turn_id    UNINDEXED,
  meeting_id UNINDEXED,
  team       UNINDEXED,
  tokenize = 'porter'
);

CREATE TABLE decisions (
  decision_id    TEXT PRIMARY KEY,
  team           TEXT NOT NULL,
  topic          TEXT NOT NULL,
  topic_key      TEXT NOT NULL,
  decision       TEXT NOT NULL,
  meeting_id     TEXT NOT NULL,
  meeting_letter TEXT NOT NULL,
  turn_id        TEXT NOT NULL,
  speaker        TEXT,
  quote          TEXT NOT NULL,
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  extracted_at   TEXT NOT NULL
);
CREATE INDEX idx_dec_team_topic ON decisions(team, topic_key, meeting_letter);

CREATE TABLE topic_key_map (
  team        TEXT NOT NULL,
  scope       TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
  topic_key   TEXT NOT NULL,
  topic       TEXT NOT NULL,
  PRIMARY KEY (team, scope, decision_id)
);

CREATE TABLE decision_supersedes (
  newer_id            TEXT NOT NULL REFERENCES decisions(decision_id),
  older_id            TEXT NOT NULL REFERENCES decisions(decision_id),
  superseding_letter  TEXT NOT NULL,
  scope               TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider            TEXT,
  model               TEXT,
  created_at          TEXT NOT NULL,
  PRIMARY KEY (newer_id, older_id, scope)
);

CREATE TABLE actions (
  action_id      TEXT PRIMARY KEY,
  team           TEXT NOT NULL,
  owner          TEXT,
  task           TEXT NOT NULL,
  meeting_id     TEXT NOT NULL,
  meeting_letter TEXT NOT NULL,
  turn_id        TEXT NOT NULL,
  quote          TEXT NOT NULL,
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  extracted_at   TEXT NOT NULL
);
CREATE INDEX idx_act_team_letter ON actions(team, meeting_letter);

CREATE TABLE action_closed_by (
  action_id      TEXT NOT NULL REFERENCES actions(action_id),
  closer_turn_id TEXT NOT NULL,
  closing_letter TEXT NOT NULL,
  scope          TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider       TEXT,
  model          TEXT,
  created_at     TEXT NOT NULL,
  PRIMARY KEY (action_id, closer_turn_id, scope)
);

CREATE TABLE disagreements (
  disagreement_id TEXT PRIMARY KEY,
  team            TEXT NOT NULL,
  topic           TEXT NOT NULL,
  who             TEXT,
  position        TEXT,
  meeting_id      TEXT NOT NULL,
  meeting_letter  TEXT NOT NULL,
  turn_id         TEXT NOT NULL,
  quote           TEXT NOT NULL,
  provider        TEXT NOT NULL,
  model           TEXT NOT NULL,
  extracted_at    TEXT NOT NULL
);
CREATE INDEX idx_dis_team_letter ON disagreements(team, meeting_letter);

CREATE TABLE disagreement_resolved_by (
  disagreement_id  TEXT NOT NULL REFERENCES disagreements(disagreement_id),
  resolver_turn_id TEXT NOT NULL,
  resolving_letter TEXT NOT NULL,
  scope            TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider         TEXT,
  model            TEXT,
  created_at       TEXT NOT NULL,
  PRIMARY KEY (disagreement_id, resolver_turn_id, scope)
);

CREATE TABLE figures (
  figure_id      TEXT PRIMARY KEY,
  team           TEXT NOT NULL,
  meeting_id     TEXT NOT NULL,
  meeting_letter TEXT NOT NULL,
  value          TEXT NOT NULL,
  unit           TEXT,
  component      TEXT,
  role           TEXT NOT NULL CHECK(role IN ('total','limit','unit_price','conversion')),
  turn_id        TEXT NOT NULL,
  quote          TEXT NOT NULL,
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  extracted_at   TEXT NOT NULL
);
CREATE INDEX idx_fig_team_letter ON figures(team, meeting_letter);

CREATE TABLE citations_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      TEXT,
  question_id TEXT,
  meeting_id  TEXT NOT NULL,
  turn_id     TEXT,
  quote       TEXT NOT NULL,
  verified    INTEGER NOT NULL,
  method      TEXT
);

CREATE TABLE llm_cache (
  cache_key     TEXT PRIMARY KEY,
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  prompt_hash   TEXT NOT NULL,
  response_json TEXT NOT NULL,
  usage_json    TEXT,
  created_at    TEXT NOT NULL
);

CREATE TABLE answer_runs (
  run_id        TEXT PRIMARY KEY,
  question_id   TEXT,
  kind          TEXT NOT NULL CHECK(kind IN ('question','brief')),
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  fallback_used INTEGER NOT NULL,
  usage_json    TEXT,
  created_at    TEXT NOT NULL
);

CREATE TABLE ledger_drops (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  team        TEXT,
  meeting_id  TEXT,
  kind        TEXT NOT NULL,
  reason      TEXT NOT NULL,
  raw_json    TEXT,
  created_at  TEXT NOT NULL
);
"""

TABLES = [
    "ledger_drops",
    "answer_runs",
    "llm_cache",
    "citations_log",
    "figures",
    "disagreement_resolved_by",
    "disagreements",
    "action_closed_by",
    "actions",
    "decision_supersedes",
    "topic_key_map",
    "decisions",
    "turns_fts",
    "turns",
    "meetings",
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    config.ensure_work_dir()
    conn = sqlite3.connect(str(path or config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection, reset: bool = False) -> None:
    if reset:
        for table in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.executescript(SCHEMA)
    conn.commit()


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def store_from_jsonl(conn: sqlite3.Connection, reset: bool = True) -> dict:
    init_db(conn, reset=reset)

    meetings = list(load_jsonl(config.MEETINGS_JSONL))
    turns = list(load_jsonl(config.TURNS_JSONL))

    conn.executemany(
        "INSERT INTO meetings (meeting_id, team, letter, meeting_number, n_turns) "
        "VALUES (:meeting_id, :team, :letter, :meeting_number, :n_turns)",
        meetings,
    )
    conn.executemany(
        "INSERT INTO turns (turn_id, meeting_id, team, letter, seq, timestamp, speaker, "
        "text, char_start, char_end, n_words) VALUES (:turn_id, :meeting_id, :team, "
        ":letter, :seq, :timestamp, :speaker, :text, :char_start, :char_end, :n_words)",
        turns,
    )
    conn.executemany(
        "INSERT INTO turns_fts (text, turn_id, meeting_id, team) "
        "VALUES (:text, :turn_id, :meeting_id, :team)",
        turns,
    )
    conn.commit()

    n_meetings = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
    n_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    n_fts = conn.execute("SELECT COUNT(*) FROM turns_fts").fetchone()[0]
    return {"meetings": n_meetings, "turns": n_turns, "fts_rows": n_fts}


def _tokens(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", query)


def sanitize_fts_query(query: str) -> str | None:
    tokens = _tokens(query)
    if not tokens:
        return None
    return " ".join(f'"{token}"' for token in tokens)


def sanitize_phrase(query: str) -> str | None:
    tokens = _tokens(query)
    if not tokens:
        return None
    return '"' + " ".join(tokens) + '"'


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    team: str | None = None,
    limit: int = 20,
    phrase: bool = False,
) -> list[dict]:
    match = sanitize_phrase(query) if phrase else sanitize_fts_query(query)
    if not match:
        return []

    sql = "SELECT turn_id, meeting_id, team, rank FROM turns_fts WHERE turns_fts MATCH ?"
    params: list = [match]
    if team:
        sql += " AND team = ?"
        params.append(team)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return [dict(row) for row in conn.execute(sql, params)]


def known_phrase_check(conn: sqlite3.Connection, turn: dict) -> dict:
    phrase = " ".join(turn["text"].split()[:6])
    rows = fts_search(conn, phrase, team=turn["team"], limit=20, phrase=True)
    return {
        "phrase": phrase,
        "expected_turn": turn["turn_id"],
        "returned": [r["turn_id"] for r in rows],
        "found": any(r["turn_id"] == turn["turn_id"] for r in rows),
    }
