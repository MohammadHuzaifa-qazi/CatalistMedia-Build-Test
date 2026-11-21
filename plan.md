# PLAN — Catalist Agentic AI Internship Build Test

> Goal: a working, evidence-based Agentic AI meeting-memory system that satisfies Catalist Media's exact build-test requirements and can be explained/demoed confidently.

**Deadline:** 23 Nov 2026 (today: 21 Nov 2026) — ~2 days.
**Principle:** judgment over polish. Simple, explainable, scalable.

## Dataset facts (verified by inspection — v2, corrected)

- 138 transcripts, 35 teams. ES: 15, IS: 10, TS: 10.
- 48,148 turns; avg 348.9/file; min 51 (IS1004a) / max 1034 (TS3005d).
- 33 teams have 4 meetings (a–d); 2 have 3:
  - `IS1002` -> b, c, d (a missing)
  - `IS1005` -> a, b, c (d missing)
- 676,745 spoken words (matches PDF ~676k); ~826,709 incl. headers/labels.
- File = 6-line header + turns (`HH:MM:SS Speaker X` then text); 0 turn-header near-misses.
- **0 digits in spoken text** — all digits are timestamps/headers; prices are spoken as words
  ("twenty five Euro", "two point three Euro"). 441 turns contain currency words.
- Turn length: median **5** words, p90 **34**, max **736**; **23,821 (49.5%)** turns have >=6 words
  (only these can host a valid quote) -> index turns, not only chunks.
- Speaker letters A–D stable per team (tested; no genuine change found). Role mapping is **best-effort**
  (explicit regex: 26/35 teams; role attached only when clear).
- `questions.json`: 25 questions (each names exactly one team; none needs >1 team) + `brief_task` team `TS3012`.
  Q09 (`IS1002`) is the only question on a 3-meeting team (b,c,d).
- Citation rules (from `instructions`): 1–3 citations, quote **>=6 words**, exact, **within one speaker turn**,
  **case+spacing ignored**, punctuation kept, paraphrases fail.
- Answer length: PDF "a sentence or two" vs JSON "one to three" -> **use 1–2 sentences** (satisfies both).
- Brief for TS3012 must **exclude `TS3012d`** (before final meeting); code-enforced `as_of` filter.

## Phase 1 corrections (v2)

See `PHASE1_DATASET_PROFILE.md` §3–§12. Key fixes: A1 (0 spoken digits / spoken number words),
B1 citation rules, B2 turn-length stats, B3 brief excludes `d`, B4 1–2 sentences,
B5 question classification, B6 letters stable (tested), B7 role regex 26/35 (not 28/35).

## Design consequences (from verified facts)

- Index **turns** with stable IDs; ~50.5% of turns are <6 words and cannot be cited.
- Deterministic quote validator (case/space-insensitive, >=6 words, one turn).
- Numbers: LLM extracts spoken values + component mapping; **code** verifies quotes + arithmetic.
- Brief: hard `as_of` filter excluding `TS3012d`.
- Answers: 1–2 sentences, 1–3 citations, speaker letter (+role only if explicit).
- Roles: per-team best-effort from first available meeting; handle missing `a`.

## Stack (free)

Python 3.13.2 + custom parser + SQLite(FTS5) + Chroma + fastembed (`bge-small-en-v1.5`, local ONNX) + Gemini 3.8 Flash primary / Groq `gpt-oss` fallback (free) + plain tool-calling loop + Pydantic + Python (quote check/math) + CLI.

## Phases

### Phase 1 — Dataset Inspection  [DONE — findings above]
- Inspect files, format, schema.
- Identify challenges (messy speech, missing meetings, indirect refs, supersession, numbers).
- Confirm dataset facts.

### Phase 2 — Architecture & Technology Decisions  [DONE — v2.3, closed]
- See `PHASE2_DESIGN.md` v2.3 (v2 amendments + v2.2 LLM lock + PATCH v2.3).
- Locked: Python 3.13.2 + SQLite FTS5 + Chroma + fastembed + Gemini 3.8 Flash primary / Groq `gpt-oss` fallback + plain tool-calling loop.
- Retrieval-first, then windowed ledger + scoped reconcile; model-vs-code split documented.

### Phase 3 — Core Implementation  [IN PROGRESS]
Deadline ~2 days (21–23 Nov). Critical path: **1–3 → embeddings (background) → 7 → ledger (background) → 4–6, 8**. Tests are written with each step.

**STEP 0 — Live provider probe (before Steps 7–9)**
- One small call per provider (Gemini, Groq); record RPM/RPD/TPM and 429 behaviour.
- Test which Gemini `thinking_level` values are accepted.
- Mark each result **VERIFIED / NOT VERIFIED**; use results to choose ledger scope (`--teams`).
- **Result (21 Nov):** Groq **VERIFIED** (`gpt-oss-20b`/`120b`; headers 1000 req/day, 8000 TPM; 429 not triggered). Gemini key valid; `gemini-3.8-flash` **UNAVAILABLE** (503/timeout); `gemini-2.5-flash` works; `thinkingLevel` unsupported on 2.5; `thinkingBudget=0` works; free-tier limits **NOT VERIFIED**. Saved to `work/step0_probe.json`.

1. **Scaffold + config**: package (`parse/store/embed/llm/tools/validate/status/agent/ledger/reconcile/brief/pipeline`), `requirements.txt`, `.env.example`, `.gitignore`; constants (prompt budget <6K, MAX_STEPS=8, batch 32, threads 8, scopes `all`/`asof_c`).
2. **Parser -> `turns.jsonl`**: 6-line header + `HH:MM:SS Speaker X`; `turn_id = {meeting}#{seq:05d}`; char offsets + `n_words`. **Check:** 48,148 turns; 676,745 words; **23,821 turns >=6 words**; **min/max turns per file 51/1034**; 0 spoken digits; **`text == file[char_start:char_end]` for every turn**.
3. **SQLite + FTS5**: full DDL — meetings, turns, FTS5, decisions, `topic_key_map`, `decision_supersedes`, actions, `action_closed_by`, disagreements, `disagreement_resolved_by`, citations_log, llm_cache, answer_runs, `ledger_drops`; **no stored status columns**; FTS sanitizer. **Check:** a known phrase returns its known turn.
4. **Embeddings + Chroma (background, resume-able)**: 5-turn context windows for >=6-word centres; fastembed (batch 32, threads 8); `turn_id` = centre; **store embedded ids and skip already-done** on restart. **Check:** 23,821 vectors; sample semantic hit; ~5.56 centres/s.
5. **Tools + scope binding**: `search` (hybrid RRF), `get_turns`, `ledger`, `verify`; `ctx={team,as_of,mode}`; executor injects; `team=None` only in `mode='eval'`; out-of-scope raises `ScopeError`. **Check:** TS3012d never returned under `{TS3012, as_of='c', mode='brief'}`.
6. **Validator + status derivation**: word-boundary single-turn verify; accepted `turn_id` = `verify()` result; validate (length, 1–3 citations, team, as_of, retrieved, soft `single_meeting`); derive decision/action/disagreement status from edges, ignoring letters > `as_of`. Sentence counter must handle **decimals ("2.3 Euro") and abbreviations** (tests). **STOP for review here** (before live LLM calls other than Step 0).
7. **LLM wrapper + cache + rate control**: Gemini first, Groq fallback on 429/413/5xx/timeout; **RPM/TPM-aware rate limiter**, **daily budget counter**, exponential backoff, **resume-from-checkpoint**; cache `sha256(provider|model|prompt)`; log `thoughts_token_count`; return `{text, tool_calls, usage, provider, model}` + provenance.
   - **DONE (21 Nov):** `catalyst/llm.py` (Gemini REST + Groq SDK), SQLite cache, RPM/TPM/RPD/TPD limiter, circuit breaker, metrics, prompt budget; `tests/test_llm.py`; **20 tests pass**; live fallback `3.8 timeout → 2.5 ok`; cache hit 0.026s.
8. **Agent loop (retrieval-first)**: **stateless prompt per step** = question + evidence notebook (selected turns, <=4K tokens) + last tool result; never append raw history; same prompt shape for Gemini/Groq (stable cache keys). Tool results: turns >150 words -> best-matching ~150-word span + "copy quotes only from displayed text"; **all shown turn_ids go into retrieved**. `single_meeting` -> one extra retrieval round; repair handles `tool_calls` (bounded) or rejects. **Check:** 25 Qs, validator passes, `answers.json` v0.
9. **Windowed ledger (scope parameter)**: `--teams` default = **24 teams used by the 25 questions + TS3012**; extend to all 138 only if measured limits allow. Index always covers all 138. 4 kinds: decisions, actions, disagreements, **figures** (`value, unit, component, total|limit|unit_price|conversion, turn_id, quote`), same windows. Quote from nearest citable turn; `verify()` failures dropped + counted. **Check:** sample ledger, all quotes verify, drops reported.
10. **Reconcile (scoped)**: one pass per `(team, scope)`; `all` = a–d, `asof_c` = a–c only; TS3012 gets both; LLM proposes topic clusters + supersede/closure/resolution edges; code validates ids/letter order/quotes/scope; delete-then-insert per `(team, scope)`. **Check:** `asof_c` never loads `d`; edges only valid letter order.
11. **Brief**: `ctx={TS3012, as_of='c', mode='brief'}`; 4 section-by-section calls (decided / changed / open actions+owner / open disagreements) within <6K each; fixed `[TS3012b: "exact quote"]`; `validate_brief()` requires >=1 verified citation per bullet, rejects uncited bullets and any `TS3012d`; **predictive-phrase filter only for forward-looking statements about the final meeting, not quoted commitments**; "Who" questions require the speaker letter (prompt rule + soft check). **Check:** valid brief, zero `d` references.
12. **Failed-question path + assemble**: after repair fails, list the question, retry with changed `k`/steps, else queue for manual review; **never write an unverified citation**; final assemble **hard-fails unless 25/25 answers have verified citations**.
13. **Metrics + CLI + handoff**: metrics JSON per stage (wall-clock, tokens incl. Gemini thought tokens, calls); ledger per-meeting tokens/time on a 5–10 meeting sample for the 1,000-meeting estimate; CLI (`ingest`, `embed`, `ledger`, `reconcile`, `answer`, `brief`, `eval`); `answers.json` Pydantic-valid with provider/model/fallback provenance; pytest green; then Phase 4/5.

### Phase 4 — Required Outputs
- Generate all 25 answers (1–2 sentences, 1–3 verified citations).
- Generate Team TS3012 pre-meeting brief (decisions, changes, open actions + owners, open disagreements; fixed `[meeting: "quote"]` citations; `validate_brief()` passes).
- Produce valid `answers.json`.

### Phase 5 — Evaluation
- Write >=10 custom questions; establish ground truth by reading transcripts.
- Run agent; compare to ground truth; report correct/incorrect.
- Identify 3 genuine failures with reasons.
- Estimate 1,000-meeting ingestion cost + time (measured where possible, assumptions stated; include `thoughts_token_count`).
- Report eval-only no-team recall gap and `ledger_drops` counts.

### Phase 6 — Submission Preparation
- README (how to run).
- ~1-page technical write-up (architecture, model-vs-code, tools + breakages, production needs).
- Demo instructions / talking points (3–5 min).
- Verify all deliverables exist.

## Deliverables checklist

- [ ] Source code + README
- [ ] `answers.json` (25 answers + brief) generated by agent
- [ ] ~1-page write-up
- [ ] 10+ custom questions + ground truth + scores
- [ ] 3 honest failures
- [ ] 1,000-meeting cost/time estimate
- [ ] Demo (recording or live-ready)
- [ ] Reply to email with links before deadline

## Open decisions (to confirm before coding)

1. Vector store: Chroma (decided in Phase 2; LanceDB rejected for now).
2. LLM: **LOCKED (v2.2)** — **Gemini Flash primary** (`gemini-3.8-flash`, free tier) + **Groq fallback** (`gpt-oss-20b` bulk, `gpt-oss-120b` final, free tier). One wrapper, prompt budget <6K, cache + provider/model provenance. Need `GEMINI_API_KEY` + `GROQ_API_KEY` in `.env`. **Approved fallback (21 Nov):** `3.8 → 2.5-flash → Groq`; 3.8 currently 503/timeout.
3. Order: retrieval-first end-to-end on the 25 questions, then add the ledger and compare.
4. Letter authoritative; role attached only when explicit in that meeting; letter stability across meetings is an ASSUMPTION (not proven).
