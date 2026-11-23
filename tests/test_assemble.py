import json
import sqlite3

import pytest
from pydantic import ValidationError

from catalyst import assemble, config, store


def test_submission_pydantic_forbids_extra_keys():
    # Valid minimal schema passes
    sub = assemble.SubmissionOut(
        answers=[
            assemble.AnswerOut(
                question_id="Q01",
                answer="Team TS3006 kept the LCD screen.",
                citations=[
                    assemble.CitationOut(
                        meeting_id="TS3006d",
                        quote="Well yeah but we dropped the speech recognition.",
                    )
                ],
            )
        ],
        brief="Brief text",
    )
    assert sub.answers[0].question_id == "Q01"

    # Extra key on CitationOut (e.g. turn_id) is forbidden
    with pytest.raises(ValidationError):
        assemble.CitationOut(
            meeting_id="TS3006d",
            quote="Well yeah but we dropped the speech recognition.",
            turn_id="TS3006d#00422",
        )

    # Extra key on AnswerOut (e.g. model / provenance) is forbidden
    with pytest.raises(ValidationError):
        assemble.AnswerOut(
            question_id="Q01",
            answer="Test.",
            citations=[{"meeting_id": "TS3006d", "quote": "a b c d e f"}],
            model="gemini-2.5-flash",
        )

    # Extra key on SubmissionOut is forbidden
    with pytest.raises(ValidationError):
        assemble.SubmissionOut(answers=[], brief="", provenance={})


def test_assemble_refuses_to_write_answers_if_not_25_of_25(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_db(conn)

    work_file = tmp_path / "answers_work.json"
    work_file.write_text(json.dumps({"results": {}}), encoding="utf-8")
    out_file = tmp_path / "answers.json"
    prov_file = tmp_path / "answers_provenance.json"

    res = assemble.run(
        answers_path=work_file,
        brief_path=tmp_path / "missing_brief.json",
        out_path=out_file,
        provenance_path=prov_file,
        conn=conn,
    )
    assert res["ok"] is False
    assert res["wrote_answers"] is False
    assert not out_file.exists()
    assert prov_file.exists()
    assert len(res["failures"]) == 25
