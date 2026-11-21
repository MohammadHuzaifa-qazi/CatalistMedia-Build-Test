from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config

TURN_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}) (Speaker [A-Z])$", re.MULTILINE)
TEAM_RE = re.compile(config.TEAM_RE)


@dataclass
class Turn:
    turn_id: str
    meeting_id: str
    team: str
    letter: str
    seq: int
    timestamp: str
    speaker: str
    text: str
    char_start: int
    char_end: int
    n_words: int


def parse_transcript(path: Path) -> tuple[dict, list[Turn]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()

    stem = path.stem
    m = TEAM_RE.match(stem)
    if not m:
        raise ValueError(f"bad meeting filename: {path.name}")
    team, letter = m.group(1), m.group(2)

    headers = list(TURN_RE.finditer(raw))
    if not headers:
        raise ValueError(f"no turns in {path.name}")

    turns: list[Turn] = []
    for i, header in enumerate(headers):
        seg_start = header.end()
        seg_end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        segment = raw[seg_start:seg_end]
        text = segment.strip()

        if text:
            lead = len(segment) - len(segment.lstrip())
            trail = len(segment) - len(segment.rstrip())
            char_start = seg_start + lead
            char_end = seg_end - trail
        else:
            char_start = char_end = seg_start

        if raw[char_start:char_end] != text:
            raise AssertionError(f"offset mismatch in {path.name} turn {i}")

        turns.append(
            Turn(
                turn_id=f"{stem}#{i:05d}",
                meeting_id=stem,
                team=team,
                letter=letter,
                seq=i,
                timestamp=header.group(1),
                speaker=header.group(2),
                text=text,
                char_start=char_start,
                char_end=char_end,
                n_words=len(text.split()),
            )
        )

    meeting = {
        "meeting_id": stem,
        "team": team,
        "letter": letter,
        "meeting_number": "abcd".index(letter) + 1,
        "n_turns": len(turns),
    }
    return meeting, turns


def ingest() -> dict:
    config.ensure_work_dir()
    files = sorted(config.TRANSCRIPTS_DIR.glob("*.txt"))
    meetings: list[dict] = []
    turns: list[Turn] = []

    for path in files:
        meeting, file_turns = parse_transcript(path)
        meetings.append(meeting)
        turns.extend(file_turns)

    with config.TURNS_JSONL.open("w", encoding="utf-8") as f:
        for turn in turns:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    with config.MEETINGS_JSONL.open("w", encoding="utf-8") as f:
        for meeting in meetings:
            f.write(json.dumps(meeting, ensure_ascii=False) + "\n")

    return checkpoint(turns, meetings, files)


def checkpoint(turns: list[Turn], meetings: list[dict], files: list[Path]) -> dict:
    per_file: dict[str, int] = {}
    for turn in turns:
        per_file[turn.meeting_id] = per_file.get(turn.meeting_id, 0) + 1

    words = sum(t.n_words for t in turns)
    citable = sum(1 for t in turns if t.n_words >= config.MIN_QUOTE_WORDS)
    digits = sum(1 for t in turns if re.search(r"\d", t.text))
    min_turns = min(per_file.values())
    max_turns = max(per_file.values())

    checks = {
        "files_138": len(files) == 138,
        "meetings_138": len(meetings) == 138,
        "turns_48148": len(turns) == 48148,
        "words_676745": words == 676745,
        "citable_23821": citable == 23821,
        "min_turns_51": min_turns == 51,
        "max_turns_1034": max_turns == 1034,
        "zero_spoken_digits": digits == 0,
    }
    return {
        "files": len(files),
        "meetings": len(meetings),
        "turns": len(turns),
        "words": words,
        "citable_ge6": citable,
        "min_turns_per_file": min_turns,
        "max_turns_per_file": max_turns,
        "spoken_turns_with_digits": digits,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
