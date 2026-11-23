# ARCHITECTURE — Catalyst Agentic AI Meeting Memory

> Grounded, retrieval-first, turn-indexed meeting memory system built for the 138-meeting AMI corpus and architected to migrate cleanly to 10,000+ meetings.

---

## 1. One-Line Summary

Parse 138 AMI transcripts into offset-preserving speaker turns, index them in SQLite (metadata + FTS5) and Chroma (`fastembed` `BAAI/bge-small-en-v1.5` over citable 5-turn context windows), maintain an edge-only temporal memory ledger for cross-meeting evolution, and run a plain Python stateless tool-calling agent + deterministic citation validator to produce `answers.json`, `answers_provenance.json`, and the `as_of='c'` Team `TS3012` brief.

---

## 2. High-Level Data Flow

```text
Catalist_AI_Intern_Build_Test_Data (1)/transcripts/*.txt  (138 files, 35 teams)
      |
      v
[1] PARSER (catalyst/parse.py)
      |  6-line header + "HH:MM:SS Speaker X" turns + exact char offsets
      v
work/meetings.jsonl (138) + work/turns.jsonl (48,148 turns; 0 offset mismatches)
      |
      +---> [2] SQLITE + FTS5 (catalyst/store.py -> work/catalyst.db)
      |           - 15 tables: turns, turns_fts (porter), ledger & edge tables,
      |             llm_cache, answer_runs, citations_log, ledger_drops
      |
      +---> [3] LOCAL VECTOR INDEX (catalyst/embed.py -> work/chroma)
      |           - fastembed BAAI/bge-small-en-v1.5 (384-dim ONNX CPU)
      |           - 23,821 citable (>=6-word) centres, 5-turn context windows
      |           - Resumable via work/embed_progress.json
      |
      +---> [4] TEMPORAL LEDGER & SCOPED RECONCILIATION
      |           (catalyst/ledger.py, catalyst/reconcile.py, catalyst/status.py)
      |           - Windowed extraction (decisions, actions, disagreements, figures)
      |           - Scoped edge tables: scope='asof_c' (a-c only) vs scope='all' (a-d)
      |           - Query-time status derivation (never stored as mutable status)
      |
      v
[5] STATELESS TOOL-CALLING AGENT (catalyst/agent.py + catalyst/tools.py)
      |  code-bound RunContext(team, as_of, mode) -- hidden from LLM schemas
      |  tools: search, get_turns, ledger, verify
      |  zero-LLM seed_search + neighbor expansion + EvidenceNotebook (<6K prompt)
      |  loop: retrieve -> inspect -> forced tool-free JSON synthesis
      |
      v
[6] DETERMINISTIC VALIDATION & ASSEMBLY
      |  (catalyst/validate.py, catalyst/brief.py, catalyst/assemble.py)
      |  word-boundary single-turn quote check, retrieved-set check,
      |  1-2 sentences, 1-3 citations, as_of='c' & anti-prediction brief checks,
      |  Pydantic extra="forbid" refusal gate
      v
answers.json (25 answers + TS3012 brief)  +  answers_provenance.json
```

---

## 3. System Layers (Exact Implementation)

### 3.1 Ingestion & Deterministic Parsing (`catalyst/parse.py`)
- Reads all 138 transcripts under `Catalist_AI_Intern_Build_Test_Data (1)/transcripts/*.txt`.
- Parses the 6-line header (`Meeting`, `Team`, `Meeting number`, `Speakers`, `Times`, `Source`) and each `HH:MM:SS Speaker X` dialogue turn.
- Preserves exact character offsets (`char_start`, `char_end`), word counts (`n_words`), sequential `seq`, and stable `turn_id` (`<meeting_id>#<seq:05d>`, e.g., `TS3006d#00101`).
- Verified against the full corpus: **138 meetings**, **48,148 turns**, **0 character-offset mismatches** across all 48,148 turns.

### 3.2 Hybrid Memory Store (`catalyst/store.py`, `catalyst/embed.py`)
- **SQLite (`work/catalyst.db`, WAL mode, 15 tables):**
  - Core corpus: `meetings`, `turns` (indexed by `(meeting_id, seq)` and `(team, letter, seq)`), and virtual table `turns_fts` (`fts5`, `porter` tokenizer) over all 48,148 turns.
  - FTS5 query handling (`store.fts_search`): sanitizes FTS operators (`" * ( ) : ^ -`), pushes `team`, `as_of` (`turns.letter <= ?`), and `meeting_letter` into SQL `WHERE` before `LIMIT`, runs strict AND first, and falls back to stopword-filtered OR with AMI underscore-acronym expansion (`LCD -> "LCD" OR "L_C_D_"`, `TV -> "TV" OR "T_V_"`, `RSI -> "RSI" OR "R_S_I_"`).
  - Structured memory & edge tables: `decisions`, `topic_key_map`, `decision_supersedes`, `actions`, `action_closed_by`, `disagreements`, `disagreement_resolved_by`, `figures`, `ledger_drops`.
  - Operational & audit tables: `llm_cache`, `answer_runs`, `citations_log`.
- **Chroma 1.5.9 (`work/chroma`, collection `turns_embeddings`, cosine HNSW):**
  - Local ONNX embeddings via `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, `$0` API cost).
  - Indexes only the **23,821 citable turns** (`n_words >= 6`, 49.5% of turns; 50.5% of turns are `<6` words and cannot host a valid 6-word citation). All 48,148 turns remain searchable via FTS5 and `get_turns`.
  - Each vector embeds a **5-turn context window** (`2` turns before + centre + `2` after with speaker labels) keyed by the centre `turn_id` and scalar metadata `{meeting_id, team, letter, seq, speaker, timestamp, n_words}`.
  - Resumable checkpointing via `work/embed_progress.json`.

### 3.3 Code-Scoped Tools & Hybrid Retrieval (`catalyst/tools.py`)
Every tool takes a Python-constructed `RunContext(team, as_of, mode)` (`mode in {'answer', 'brief', 'eval'}`). **`team`, `as_of`, and `mode` are never exposed in the JSON tool schemas passed to the LLM**; `team=None` is permitted strictly in `mode='eval'` (no-team retrieval benchmark) and raises `ValueError` otherwise.

| Tool | LLM Parameters | Code Enforcement |
|---|---|---|
| `search` | `query`, `meeting_letter?` (`a\|b\|c\|d`), `k?` (`1..20`) | Pushes `team` and `as_of` into SQLite `WHERE` and Chroma `$and`/`$in` metadata filters before candidate pooling; fuses FTS5 + Chroma via Reciprocal Rank Fusion `1 / (60 + rank + 1)`; trims turns `>150` words to a contiguous query-centered verbatim span (`trim_turn_text`) with zero text mutation; raises loud `VectorSearchError` (never silently degrades) if Chroma is missing/empty. |
| `get_turns` | `meeting_id`, `seq_start?`, `seq_end?`, `speaker?`, `query?` | Raises `ScopeError` if `meeting_id` is outside `ctx.team` or `letter > ctx.as_of`; caps output at 25 turns / 1,800 words (`<=150` words/turn verbatim). |
| `ledger` | `topic?`, `kind?` (`all\|decisions\|actions\|disagreements\|figures`) | Queries records `meeting_letter <= ctx.as_of` and computes `status` at query time via `catalyst/status.py`. |
| `verify` | `meeting_id`, `quote` | Deterministic word-boundary, case/space-insensitive single-turn check (`validate.verify_quote`). |

### 3.4 Memory Over Time & Temporal Scoping (`catalyst/ledger.py`, `catalyst/reconcile.py`, `catalyst/status.py`)
- **Windowed extraction (`ledger.py`):** Splits each meeting into `~2,500`-token windows (`300`-token overlap) labeled with `[turn_id] Speaker X: ...`. Every extracted decision, action, disagreement, and spoken figure must carry a `>=6`-word verbatim quote verified against the window's turns; failed quotes are dropped and logged in `ledger_drops`.
- **Scoped reconciliation (`reconcile.py`):** Runs per `(team, scope)` where `scope in {'asof_c', 'all'}`. Under `asof_c`, only meetings `a–c` are loaded into the reconciliation prompts (`TS3012d` is never read). Reconciliation writes topic clusters (`topic_key_map`) and explicit evidence-backed edges (`decision_supersedes`, `action_closed_by`, `disagreement_resolved_by`), rejecting any edge with invalid IDs, non-monotonic meeting letters (`newer <= older`), or unverified later-meeting quotes.
- **Query-time status derivation (`status.py`):** Mutable statuses are never stored on `decisions`, `actions`, or `disagreements`. At query time, `decision_status` (`current` vs `superseded`), `action_status` (`open` vs `done`), and `disagreement_status` (`open` vs `resolved`) inspect only edges matching `scope` and with target letter `<= ctx.as_of`. This guarantees `0` leakage from `TS3012d` into `as_of='c'` queries or briefs.

### 3.5 Stateless Retrieval-First Agent Loop (`catalyst/agent.py`)
Instead of a third-party graph framework or growing chat transcripts, Catalyst runs a plain Python stateless loop where **every LLM call receives exactly 2 messages** (`system` + bounded `user` prompt strictly `<6,000` estimated tokens):
1. **Zero-LLM-cost seeding (`seed_search`):**
   - Extracts self-stated speaker roles deterministically from early `a`/`b` turns (`_extract_explicit_roles`, best-effort hints).
   - Runs hybrid `search(question)` plus per-meeting coverage searches across `a`, `b`, `c`, `d` so early proposals and late-meeting decisions both enter the `EvidenceNotebook`.
   - Expands neighboring turns (`seq-5..seq+10`) around up to 2 non-overlapping top clusters in `c` and `d` (1 in `b`, and 1 in `a` when relevant), propagating half the anchor score so neighbor turns survive budget selection.
   - When a question mentions costing/budget/prices/cuts/replacements/dropped features, seeds the latest meeting's costing section (`seq-4..seq+12`) at zero LLM cost because AMI totals are spoken as words.
2. **Tool-calling & synthesis rounds (`max_steps` bounded, default CLI `6`, budget-tuned `3–4`):**
   - Calls `LLM.chat()` with `active_tools` (`search`, `get_turns`, `verify`, plus `ledger` when enabled), blocking duplicate calls via `notebook.seen_calls`.
   - On the final step, switches to `SYNTHESIS_SYSTEM_PROMPT` with `tools=None` and `response_schema=ANSWER_SCHEMA` so models cannot loop endlessly on tool calls.
3. **Deterministic validation, `single_meeting` round & bounded repair:**
   - Normalizes Unicode punctuation to ASCII (`_clean_unicode_quotes`) when verbatim-safe, then runs `validate.validate_answer(conn, ctx, draft, retrieved, question)`.
   - If `ok` with warning `single_meeting`, runs 1 extra bounded retrieval/synthesis cycle (`max_rounds=2`) seeking corroborating evidence from another meeting.
   - If validation fails, runs 1 bounded repair cycle (`max_rounds=2`) with exact error diagnostics. If still invalid, rejects the draft (`ok=False`) and **never fabricates** an answer.

### 3.6 Pre-Final-Meeting Brief (`catalyst/brief.py`)
- Hard-bound to `RunContext(team='TS3012', as_of='c', mode='brief')`.
- Runs 4 separate `<6K`-token section calls (`decided`, `changed`, `open_actions`, `open_disagreements`) grounded strictly in `a–c` ledger records and verified `TS3012a` speaker roles (`A=PM`, `B=UI`, `C=Industrial`, `D=Marketing`), with `max_tokens=2560` on dense sections so Gemini thinking tokens do not truncate JSON.
- Formats `[TS3012x: "quote"]` inline citations in Python, runs `validate_brief()` (requires `>=1` verified citation per bullet, rejects uncited bullets, rejects any `TS3012d` reference, rejects predictive phrasing about meeting `d`), drops/repairs invalid bullets, and preserves any previously verified `brief_<team>.json` if a subsequent re-run encounters a provider outage.

### 3.7 Deterministic Validation & Strict Assembly (`catalyst/validate.py`, `catalyst/assemble.py`)
- `verify_quote()`: checks meeting exists, belongs to `ctx.team`, respects `ctx.as_of`, has `>=6` words, and matches a **single speaker turn** on word boundaries (`(?<!\w)...(?!\w)`) after case/whitespace normalization. The `turn_id` returned by `verify_quote()` overrides any LLM-supplied `turn_id`.
- `validate_answer()`: enforces `1–2` sentences (`count_sentences` protects decimals and abbreviations), `1–3` citations, every citation verified and present in the run's `retrieved` set, and warns on `single_meeting` or `who_missing_speaker`.
- `assemble.py`: re-verifies all 25 answers and the `TS3012` brief against `work/catalyst.db`, validates against Pydantic models with `extra="forbid"`, writes `answers.json` **only when 25/25 pass**, and writes full metadata/usage/provenance separately to `answers_provenance.json`.

### 3.8 Multi-Provider LLM Reliability & Security (`catalyst/llm.py`, `catalyst/config.py`)
- **Provider chain (`final` tier, 7 candidates):** `gemini-3.8-flash` (1 attempt, circuit-broken after 3 consecutive server/timeout failures) -> `gemini-2.5-flash` (`thinkingBudget=1024`) -> Groq `openai/gpt-oss-120b` (`reasoning_effort=medium`) -> Groq `openai/gpt-oss-20b` (`reasoning_effort=low`) -> `gemini-3.5-flash-lite` -> Groq `qwen/qwen3.8-27b` (`max_tokens<=750` under its 1,000 OTPM cap) -> `gemini-flash-lite-latest`. `bulk` tier skips `120b` and uses `thinkingBudget=0` on `2.5-flash`.
- **Rate limits & retry discipline:** Persistent per-model `RateLimiter` (`work/rate_state.json`) tracking sliding-window RPM/TPM and UTC-daily RPD/TPD, with per-model `MODEL_RATE_LIMITS` overrides and a 15% BPE cushion for Groq's 8K TPM ceiling. Respects HTTP `Retry-After` headers and Groq body retry delays (including compound daily delays such as `15m23.4s`, up to `3600s` on daily 429s); opens per-model cooldowns on 429 and circuit breakers on repeated 5xx/timeouts; refunds reserved tokens on failed calls and refunds daily request slots (`refund_request()`) on server 5xx/timeout errors; caches all successful responses in SQLite `llm_cache`; preserves prior verified answers/briefs during failed re-runs.
- **Groq recovery & clean failures:** Handles Groq `json_validate_failed` / `tool_use_failed` by lowering reasoning effort / expanding completion headroom within TPM and retrying once before falling back; converts tool schemas to nullable types for Groq compatibility. When every candidate is exhausted, raises a structured `AllProvidersFailed` listing each provider's status and retry delay — never fabricating an answer.
- **Credential security:** Gemini requests pass the API key via HTTP header `x-goog-api-key` (never in URL query parameters), and `_redact_secrets()` scrubs API key patterns from all exception strings and `work/metrics/llm_calls.jsonl`.

---

## 4. Model-vs-Code Responsibility Split

| Task | Owner | Rationale |
|---|---|---|
| Transcript parsing, character offsets, turn IDs, SQLite/FTS5 & Chroma indexing | **Code** | Deterministic, reproducible, zero hallucination |
| `team` and `as_of` scoping, FTS5/Chroma pushdown, `<=150`-word verbatim trimming | **Code** | Hard isolation boundary (cannot be bypassed by prompts) |
| Self-stated role hints (`a`/`b`), neighbor & costing window seeding, RRF ranking | **Code** | High-recall zero-LLM-cost grounding |
| Natural-language comprehension, cross-meeting synthesis, spoken-number extraction | **Model** | Requires semantic reasoning over conversational speech |
| Ledger record extraction & cross-meeting edge proposals (`supersedes`/`closed`/`resolved`) | **Model** | Requires interpreting implicit discourse & revisions |
| Edge validation (IDs, monotonic letter order, verbatim quotes) & query-time status | **Code** | Prevents silent hallucinations and temporal leakage |
| Quote verification, `turn_id` authority, length/count/retrieved checks, anti-prediction gate, Pydantic `extra="forbid"`, cost arithmetic | **Code** | Zero tolerance for paraphrased quotes or schema drift |

---

## 5. Scalability Analysis (1,000 and 10,000 Meetings)

### 5.1 Can the current prototype scale directly to 10,000 meetings?
**Architecturally yes (stateless `<6K`-token prompts, turn-level indexing, incremental ingestion, per-team partitioning); operationally it requires targeted infrastructure upgrades at 10,000 meetings:**

| Component | Behavior at 138 Meetings (Verified) | Bottleneck at 10,000 Meetings (~3.49M turns, ~1.73M vectors) | Production Upgrade |
|---|---|---|---|
| **Prompt size / Agent loop** | `<6K` tokens/call (team-scoped) | **Scales cleanly** (`O(1)` prompt size per query; never loads corpus) | Keep stateless `EvidenceNotebook`; pin a single paid model tier for consistent quality |
| **Local CPU Embeddings (`fastembed`)** | **2.34 centres/s** (`2.82 h` for `23,757` windows) | **Primary bottleneck:** ~**204.9 h (~8.5 days)** single-machine CPU (linear extrapolation) | GPU ONNX (`10–50x` faster) or hosted batch embeddings + `turn_id`-keyed persistent cache |
| **Vector Store (Chroma single-node)** | `23,821` vectors (`34.9 MB` raw float32) | ~`1.73M` vectors (`2.53 GB` raw float32 + HNSW RAM overhead); single-node latency/memory degrade | Migrate to **Qdrant** or **PostgreSQL (`pgvector`)** partitioned by `team`/`org_id` with pushed-down metadata filters |
| **Keyword Store (SQLite FTS5)** | `48,148` rows (sub-10ms reads) | ~`3.49M` rows: reads remain fast, but single-writer ingestion serializes | PostgreSQL `tsvector` + GIN index (or sharded FTS5) with batched async worker ingestion |
| **Ledger & Reconciliation (LLM)** | `4` `TS3012` meetings (`20` windows, `183.8 s`); `2` reconcile scopes (`145.1 s`) | Free tiers cannot run `~30,100` windows (`~179 h` serial); `thoughts_tokens` are `55.6%` of billed reconcile output | Async worker queue + paid Gemini/OpenAI Batch API (~50% discount); precompute & cache once per `(team, scope)` |
| **No-team-filter retrieval** | Tested empirically: cross-team contamination collapses recall when `team=None` | Severe at `2,536` teams if queries are unscoped | Require org/team routing (metadata filter or lightweight team-classifier router) before dense/BM25 retrieval |

### 5.2 Cost & Time Estimates (`catalyst/cost_estimate.py` — Source Status Tagged)

All figures clearly separate **VERIFIED measured baselines**, **VERIFIED published prices**, **ASSUMPTION reference prices**, and **LINEAR EXTRAPOLATIONS** (`work/metrics/cost_time_estimate.json`):

| Scale | Est. Turns / Citable Centres | CPU Embeddings @ `2.34/s` | Full Ledger + 1-Scope Reconcile Tokens (Prompt / Billed Output incl. Thoughts) | Serial LLM Time | Est. Cost (`gemini-3.8-flash` 2026 **VERIFIED** \$0.75/\$3.75) | Est. Cost (`gemini-3.8-flash` 2027 **VERIFIED** \$1.50/\$7.50) | Est. Cost (`gemini-2.5-flash` **ASSUMPTION** \$0.30/\$2.50) | Est. Cost (`groq 120b` **ASSUMPTION** \$0.15/\$0.75) |
|---|---|---|---|---|---|---|---|---|
| **138 meetings** (`35` teams) | `48,148` / `23,821` (**VERIFIED**) | `2.83 h` (`2.82 h` **VERIFIED**) | `2.91M` / `0.77M` (extrap. from `TS3012`) | `2.47 h` | \$5.08 | \$10.15 | \$2.80 | \$1.02 |
| **1,000 meetings** (`~254` teams) | `348,899` / `172,616` (*LINEAR EXTRAP.*) | `20.49 h` (*LINEAR EXTRAP.*) | `21.06M` / `5.60M` (*LINEAR EXTRAP.*) | `17.88 h` | **\$36.80** | **\$73.61** | **\$20.32** | **\$7.36** |
| **10,000 meetings** (`~2,536` teams) | `3.49M` / `1.73M` (*LINEAR EXTRAP.*) | `204.91 h` / `8.54 d` (*LINEAR EXTRAP.*) | `210.53M` / `56.00M` (*LINEAR EXTRAP.*) | `178.76 h` | **\$367.90** | **\$735.79** | **\$203.16** | **\$73.58** |

*Notes:* (1) `TS3012` averages `5.0` extraction windows/meeting vs `3.01` corpus average, making the token and cost extrapolations conservative upper bounds for AMI-length meetings. (2) Gemini `thoughts_tokens` are explicitly included in `billed_output_tokens` (`55.6%` of measured reconcile output). (3) Query-time QA costs are additive and scale with user query volume rather than corpus size (see `work/metrics/cost_time_estimate.json` for measured per-question averages by model).

---

## 6. Actual Technology Stack

| Layer | Implemented Choice |
|---|---|
| Language & Runtime | Python `3.12.2` |
| Parser & Validator | Pure Python (`re`, `sqlite3`, `json`, `hashlib`) |
| Keyword & Relational Store | SQLite (WAL) + FTS5 (`porter` tokenizer) |
| Vector Store | Chroma (`chromadb==1.5.9`, persistent cosine HNSW) |
| Embeddings | `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim, local ONNX CPU) |
| LLMs | Primary: Google Gemini (`gemini-3.8-flash` -> `gemini-2.5-flash`); Fallback: Groq (`openai/gpt-oss-120b` -> `openai/gpt-oss-20b`) via `requests` |
| Orchestration | Plain Python stateless tool-calling loop (`catalyst/agent.py`, `catalyst/pipeline.py` CLI) |
| Schema Enforcement | `pydantic` (`extra="forbid"`) |
| Testing & Evaluation | `pytest`, custom multi-meeting evaluation harness (`catalyst/evaluate.py`), cost estimator (`catalyst/cost_estimate.py`) |
| Demo Interface | Reproducible CLI commands (`DEMO_SCRIPT.md`, 3–5 minutes) |
