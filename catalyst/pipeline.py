from __future__ import annotations

import argparse
import json
import sys

from . import config, ledger, parse, store


def cmd_parse(_args: argparse.Namespace) -> int:
    result = parse.ingest()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_checks_pass"] else 1


def cmd_store(args: argparse.Namespace) -> int:
    if not config.TURNS_JSONL.exists() or not config.MEETINGS_JSONL.exists():
        print("turns.jsonl/meetings.jsonl missing; run `parse` first", file=sys.stderr)
        return 1

    conn = store.connect()
    try:
        result = store.store_from_jsonl(conn, reset=not args.no_reset)
        turns = list(store.load_jsonl(config.TURNS_JSONL))
        probe = next(t for t in turns if t["n_words"] >= 10)
        result["known_phrase_check"] = store.known_phrase_check(conn, probe)
    finally:
        conn.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["known_phrase_check"]["found"] else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    parsed = parse.ingest()
    print(json.dumps({"parse": parsed}, indent=2, ensure_ascii=False))
    if not parsed["all_checks_pass"]:
        return 1
    return cmd_store(args)


def cmd_ledger(args: argparse.Namespace) -> int:
    teams = [team.strip() for team in args.teams.split(",") if team.strip()] or None
    result = ledger.run(
        teams=teams,
        limit=args.limit,
        max_windows=args.max_windows,
        reset=args.reset,
        all_teams=args.all,
        window_tokens=args.window_tokens,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catalyst")
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse transcripts -> turns.jsonl")
    p_parse.set_defaults(func=cmd_parse)

    p_store = sub.add_parser("store", help="load turns.jsonl -> SQLite + FTS5")
    p_store.add_argument("--no-reset", action="store_true", help="do not drop existing tables")
    p_store.set_defaults(func=cmd_store)

    p_ingest = sub.add_parser("ingest", help="parse + store")
    p_ingest.add_argument("--no-reset", action="store_true", help="do not drop existing tables")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ledger = sub.add_parser("ledger", help="extract windowed ledger")
    p_ledger.add_argument("--teams", default="")
    p_ledger.add_argument("--all", action="store_true")
    p_ledger.add_argument("--limit", type=int)
    p_ledger.add_argument("--max-windows", type=int)
    p_ledger.add_argument("--reset", action="store_true")
    p_ledger.add_argument("--window-tokens", type=int, default=2500)
    p_ledger.set_defaults(func=cmd_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
