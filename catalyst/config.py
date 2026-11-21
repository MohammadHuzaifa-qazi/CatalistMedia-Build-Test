from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Catalist_AI_Intern_Build_Test_Data (1)"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
QUESTIONS_PATH = DATA_DIR / "questions.json"

WORK_DIR = ROOT / "work"
DB_PATH = WORK_DIR / "catalyst.db"
CHROMA_DIR = WORK_DIR / "chroma"
TURNS_JSONL = WORK_DIR / "turns.jsonl"
MEETINGS_JSONL = WORK_DIR / "meetings.jsonl"
EMBED_PROGRESS = WORK_DIR / "embed_progress.json"
METRICS_DIR = WORK_DIR / "metrics"

PROMPT_BUDGET_TOKENS = 6000
EVIDENCE_BUDGET_TOKENS = 4000
MAX_STEPS = 8
MIN_QUOTE_WORDS = 6

WINDOW_BEFORE = 2
WINDOW_AFTER = 2
EMBED_BATCH = 32
EMBED_THREADS = 8
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION = "turns_embeddings"

GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
GROQ_BULK_MODEL = "openai/gpt-oss-20b"
GROQ_FINAL_MODEL = "openai/gpt-oss-120b"

GEMINI_38_THINKING_BULK = "LOW"
GEMINI_38_THINKING_FINAL = "HIGH"
GEMINI_25_BUDGET_BULK = 0
GEMINI_25_BUDGET_FINAL = 1024
GROQ_BULK_REASONING = "low"
GROQ_FINAL_REASONING = "medium"

LLM_MAX_TOKENS_BULK = 1536
LLM_MAX_TOKENS_FINAL = 2048
LLM_TIMEOUT_BULK = 12.0
LLM_TIMEOUT_FINAL = 30.0
CIRCUIT_BREAKER_FAILURES = 3
CIRCUIT_BREAKER_SECONDS = 600
RATE_STATE_PATH = WORK_DIR / "rate_state.json"
LLM_METRICS_PATH = METRICS_DIR / "llm_calls.jsonl"

RATE_LIMITS = {
    "gemini": {
        "rpm": 10,
        "tpm": 250_000,
        "rpd": 250,
        "tpd": 1_000_000,
        "verified": False,
    },
    "groq": {
        "rpm": 30,
        "tpm": 8_000,
        "rpd": 1_000,
        "tpd": 200_000,
        "verified": True,
    },
}

SCOPES = ("all", "asof_c")
TEAM_RE = r"^([A-Z]{2}\d{4})([a-d])$"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def ensure_work_dir() -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR
