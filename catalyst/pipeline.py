from __future__ import annotations

import argparse
import json
import sys

from . import config, ledger, parse, reconcile, store


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


def cmd_reconcile(args: argparse.Namespace) -> int:
    teams = [team.strip() for team in args.teams.split(",") if team.strip()] or None
    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()] or None
    result = reconcile.run(
        teams=teams,
        scopes=scopes,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


def cmd_embed(args: argparse.Namespace) -> int:
    # Imported lazily: chromadb/fastembed are only needed for this command.
    from . import embed

    embed.run(limit=args.limit, reset=args.reset, batch_size=args.batch)
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    try:
        from . import agent
    except ImportError:
        print(
            "answer command not implemented yet (requires catalyst.agent)",
            file=sys.stderr,
        )
        return 2
    questions = (
        [q.strip() for q in args.question.split(",") if q.strip()]
        if args.question
        else None
    )
    result = agent.run_questions(
        question_ids=questions,
        limit=args.limit,
        k=args.k,
        max_steps=args.max_steps,
        use_ledger=not args.no_ledger,
        out_path=args.out,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result.get("failed") else 1


def cmd_brief(args: argparse.Namespace) -> int:
    try:
        from . import brief
    except ImportError:
        print(
            "brief command not implemented yet (requires catalyst.brief)",
            file=sys.stderr,
        )
        return 2
    result = brief.run(
        team=args.team,
        as_of=args.as_of,
        out_path=args.out,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_assemble(args: argparse.Namespace) -> int:
    try:
        from . import assemble
    except ImportError:
        print(
            "assemble command not implemented yet (requires catalyst.assemble)",
            file=sys.stderr,
        )
        return 2
    result = assemble.run(
        answers_path=args.answers,
        brief_path=args.brief,
        out_path=args.out,
        provenance_path=args.provenance_out,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        from . import evaluate
    except ImportError:
        print(
            "eval command not implemented yet (requires catalyst.evaluate)",
            file=sys.stderr,
        )
        return 2
    result = evaluate.run(
        questions_path=args.questions,
        k=args.k,
        max_steps=args.max_steps,
        no_team_test=args.no_team_test,
        out_path=args.out,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok", True) else 1


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

    p_reconcile = sub.add_parser("reconcile", help="reconcile topics and edges")
    p_reconcile.add_argument("--teams", default="")
    p_reconcile.add_argument("--scopes", default="")
    p_reconcile.add_argument("--limit", type=int)
    p_reconcile.set_defaults(func=cmd_reconcile)

    p_embed = sub.add_parser("embed", help="embed turns -> Chroma (resumable)")
    p_embed.add_argument("--limit", type=int)
    p_embed.add_argument("--reset", action="store_true")
    p_embed.add_argument("--batch", type=int)
    p_embed.set_defaults(func=cmd_embed)

    p_answer = sub.add_parser("answer", help="answer questions via retrieval-first agent")
    p_answer.add_argument("--question", default="", help="question id(s), comma-separated")
    p_answer.add_argument("--limit", type=int)
    p_answer.add_argument("--k", type=int, default=8)
    p_answer.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p_answer.add_argument("--no-ledger", action="store_true")
    p_answer.add_argument("--force", action="store_true", help="re-run even if already verified in answers_work.json")
    p_answer.add_argument("--out", default=str(config.WORK_DIR / "answers_work.json"))
    p_answer.set_defaults(func=cmd_answer)

    p_brief = sub.add_parser("brief", help="generate pre-meeting brief (as_of='c')")
    p_brief.add_argument("--team", default="TS3012")
    p_brief.add_argument("--as-of", dest="as_of", default="c")
    p_brief.add_argument("--out", default=str(config.WORK_DIR / "brief_TS3012.json"))
    p_brief.set_defaults(func=cmd_brief)

    p_assemble = sub.add_parser("assemble", help="validate and assemble answers.json")
    p_assemble.add_argument("--answers", default=str(config.WORK_DIR / "answers_work.json"))
    p_assemble.add_argument("--brief", default=str(config.WORK_DIR / "brief_TS3012.json"))
    p_assemble.add_argument("--out", default=str(config.ROOT / "answers.json"))
    p_assemble.add_argument(
        "--provenance-out",
        dest="provenance_out",
        default=str(config.ROOT / "answers_provenance.json"),
    )
    p_assemble.set_defaults(func=cmd_assemble)

    p_eval = sub.add_parser("eval", help="run evaluation harness on custom questions")
    p_eval.add_argument("--questions", default=str(config.WORK_DIR / "eval_questions.json"))
    p_eval.add_argument("--k", type=int, default=8)
    p_eval.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    p_eval.add_argument("--no-team-test", dest="no_team_test", action="store_true")
    p_eval.add_argument("--out", default=str(config.METRICS_DIR / "eval_report.json"))
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
