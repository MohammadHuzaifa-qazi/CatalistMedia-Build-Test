from catalyst import config, parse


def test_single_file_offsets():
    path = config.TRANSCRIPTS_DIR / "TS3012a.txt"
    meeting, turns = parse.parse_transcript(path)
    raw = path.open("r", encoding="utf-8", newline="").read()

    assert meeting["meeting_id"] == "TS3012a"
    assert meeting["team"] == "TS3012"
    assert meeting["letter"] == "a"
    assert meeting["n_turns"] == len(turns)
    for turn in turns:
        assert raw[turn.char_start:turn.char_end] == turn.text
        assert turn.turn_id.startswith("TS3012a#")


def test_corpus_checkpoint():
    result = parse.ingest()
    assert result["checks"]["files_138"]
    assert result["checks"]["turns_48148"]
    assert result["checks"]["words_676745"]
    assert result["checks"]["citable_23821"]
    assert result["checks"]["min_turns_51"]
    assert result["checks"]["max_turns_1034"]
    assert result["checks"]["zero_spoken_digits"]
