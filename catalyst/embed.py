from __future__ import annotations

import argparse
import json
import time

import chromadb
from fastembed import TextEmbedding

from . import config, store

_embedder: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(
            model_name=config.EMBED_MODEL, threads=config.EMBED_THREADS
        )
    return _embedder


def load_turns() -> list[dict]:
    return list(store.load_jsonl(config.TURNS_JSONL))


def build_windows(turns: list[dict]) -> list[dict]:
    by_meeting: dict[str, list[dict]] = {}
    for turn in turns:
        by_meeting.setdefault(turn["meeting_id"], []).append(turn)

    windows: list[dict] = []
    for meeting_turns in by_meeting.values():
        meeting_turns.sort(key=lambda t: t["seq"])
        for i, centre in enumerate(meeting_turns):
            if centre["n_words"] < config.MIN_QUOTE_WORDS:
                continue
            lo = max(0, i - config.WINDOW_BEFORE)
            hi = min(len(meeting_turns), i + config.WINDOW_AFTER + 1)
            document = "\n".join(
                f'{t["speaker"]}: {t["text"]}' for t in meeting_turns[lo:hi]
            )
            windows.append(
                {
                    "turn_id": centre["turn_id"],
                    "document": document,
                    "metadata": {
                        "meeting_id": centre["meeting_id"],
                        "team": centre["team"],
                        "letter": centre["letter"],
                        "seq": centre["seq"],
                        "speaker": centre["speaker"],
                        "timestamp": centre["timestamp"],
                        "n_words": centre["n_words"],
                    },
                }
            )
    return windows


def load_progress() -> set[str]:
    if config.EMBED_PROGRESS.exists():
        return set(json.loads(config.EMBED_PROGRESS.read_text(encoding="utf-8")))
    return set()


def save_progress(done: set[str]) -> None:
    config.EMBED_PROGRESS.write_text(
        json.dumps(sorted(done)), encoding="utf-8"
    )


def run(limit: int | None = None, reset: bool = False, batch_size: int | None = None) -> dict:
    config.ensure_work_dir()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
        if config.EMBED_PROGRESS.exists():
            config.EMBED_PROGRESS.unlink()

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    done = load_progress()
    windows = build_windows(load_turns())
    if limit:
        windows = windows[:limit]
    todo = [w for w in windows if w["turn_id"] not in done]
    print(
        f"windows={len(windows)} todo={len(todo)} already_done={len(done)}",
        flush=True,
    )

    embedder = get_embedder()
    batch = batch_size or config.EMBED_BATCH
    started = time.time()

    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        texts = [w["document"] for w in chunk]
        vectors = [v.tolist() for v in embedder.embed(texts, batch_size=batch)]
        collection.upsert(
            ids=[w["turn_id"] for w in chunk],
            embeddings=vectors,
            documents=texts,
            metadatas=[w["metadata"] for w in chunk],
        )
        done.update(w["turn_id"] for w in chunk)

        if (start // batch) % 10 == 0:
            save_progress(done)
            elapsed = time.time() - started
            rate = (start + len(chunk)) / elapsed if elapsed else 0
            print(
                f"embedded={start + len(chunk)}/{len(todo)} rate={rate:.2f}/s",
                flush=True,
            )

    save_progress(done)
    elapsed = time.time() - started
    result = {
        "collection_count": collection.count(),
        "embedded_this_run": len(todo),
        "total_done": len(done),
        "elapsed_sec": round(elapsed, 1),
        "centres_per_sec": round(len(todo) / elapsed, 2) if elapsed else None,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catalyst.embed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args(argv)
    run(limit=args.limit, reset=args.reset, batch_size=args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
