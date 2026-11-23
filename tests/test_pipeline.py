from catalyst import pipeline


def test_cli_parser_has_embed_command():
    parser = pipeline.build_parser()
    args = parser.parse_args(
        ["embed", "--limit", "2", "--batch", "4", "--reset"]
    )

    assert args.command == "embed"
    assert args.limit == 2
    assert args.batch == 4
    assert args.reset is True
    assert args.func is pipeline.cmd_embed


def test_cli_parser_embed_defaults():
    parser = pipeline.build_parser()
    args = parser.parse_args(["embed"])

    assert args.limit is None
    assert args.batch is None
    assert args.reset is False


def test_cli_parser_has_answer_brief_assemble_eval_commands():
    parser = pipeline.build_parser()

    ans = parser.parse_args(["answer", "--question", "Q01", "--k", "10", "--max-steps", "6"])
    assert ans.command == "answer"
    assert ans.question == "Q01"
    assert ans.k == 10
    assert ans.max_steps == 6
    assert ans.func is pipeline.cmd_answer

    br = parser.parse_args(["brief", "--team", "TS3012", "--as-of", "c"])
    assert br.command == "brief"
    assert br.team == "TS3012"
    assert br.as_of == "c"
    assert br.func is pipeline.cmd_brief

    asm = parser.parse_args(["assemble", "--out", "out.json"])
    assert asm.command == "assemble"
    assert asm.out == "out.json"
    assert asm.func is pipeline.cmd_assemble

    ev = parser.parse_args(["eval", "--no-team-test", "--k", "16"])
    assert ev.command == "eval"
    assert ev.no_team_test is True
    assert ev.k == 16
    assert ev.func is pipeline.cmd_eval


def test_all_cli_subcommands_are_wired():
    parser = pipeline.build_parser()
    for cmd in ("parse", "store", "ingest", "ledger", "reconcile", "embed", "answer", "brief", "assemble", "eval"):
        args = parser.parse_args([cmd])
        assert args.command == cmd
        assert callable(args.func)
