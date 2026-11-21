# PHASE 2 — DESIGN (v2.3, locked — Phase 2 closed)

> Applies all v2/v2.2 amendments + **PATCH v2.3**. Changes vs v2.2 and conflicts are listed in §13.
> Scope: decisions, DB DDL, tool signatures, agent loop, validator, supersession, reconcile, brief, model-vs-code, scale, eval, risks.
> No ADR log, no ARCHITECTURE.md rewrite. Verified Phase 1 v2 facts drive every decision.
> **Phase 2 is closed after this patch; no further redesign.**

---

## 1. Decisions

### 1.1 Locked stack

| Area | Decision | Why / rejected alternative |
|---|---|---|
| Language | Python 3.13.2 | **VERIFIED** working with fastembed + chromadb on this machine |
| Parser | custom regex (Phase 3) | deterministic, no LLM |
| Keyword store | SQLite **FTS5** | built-in, exact-term recall, BM25 |
| Vector store | **Chroma** (persistent local) | **VERIFIED**: chromadb 1.5.9 installs + runs (add + query) on Python 3.13.2; LanceDB rejected for a 1-day build |
| Embeddings | **fastembed** `BAAI/bge-small-en-v1.5` (ONNX, 384-dim) | **VERIFIED**: fastembed 0.7.4 + onnxruntime 1.23.2 run; 384-dim vectors produced |
| Orchestration | **plain tool-calling loop** (no LangGraph) | fewer moving parts, easier to defend |
| Validation | **code-enforced gate** before output | citations must be exact, never LLM-judged |
| Demo | **CLI / notebook only** | no Streamlit |
| Output | `answers.json` (Pydantic-validated) | matches required schema |

### 1.2 LLM stack (v2.2 lock)

**PRIMARY = Gemini Flash (free tier). FALLBACK = Groq `gpt-oss-*` (free tier). Test build only.**

| Role | Primary | Fallback |
|---|---|---|
| Bulk / ledger extraction | `gemini-3.8-flash` | Groq `openai/gpt-oss-20b` |
| Final answers (25 Qs) + TS3012 brief | `gemini-3.8-flash` | Groq `openai/gpt-oss-120b` |
| Reconcile (team-level, scoped) | `gemini-3.8-flash` | Groq `openai/gpt-oss-120b` |

- **Exact Gemini model name: VERIFIED** = `gemini-3.8-flash` (official Google model + pricing pages, Nov 2026).
- **Context window: VERIFIED** = **1,048,576 tokens**; max output **65,536**.
- **Paid pricing: VERIFIED** (through 2026-12-31): **$0.75 / 1M input, $3.75 / 1M output**; from 2027-01-01: $1.50 / $7.50.
- **Gemini free-tier RPM/RPD/TPM: NOT VERIFIED** — Google no longer publishes per-model free limits; they appear only in AI Studio per project/model. Record as VERIFIED only if read from official docs; otherwise mark NOT VERIFIED.
- **Groq free tier: VERIFIED** — `gpt-oss-120b` / `gpt-oss-20b` = **30 RPM, 1,000 RPD, 8K TPM, 200K TPD**. Llama was removed from the free tier on 16 Aug 2026 — never use Llama.

### 1.3 One LLM client wrapper (mandatory)

```
LLMClient.chat(messages, tools) -> {text, tool_calls, usage, provider, model}
```

- Try **Gemini first**: 2–3 retries with exponential backoff on 429 / 5xx / timeout.
- Fall back to **Groq only** on 429 / 413 / 5xx / timeout (not on content errors).
- Return the actual `provider` + `model` used with every result.
- **Prompt budget: every request < 6K tokens total** (system + history + tool results). Trim tool results and history so the Groq fallback never hits its 8K-TPM/413 wall.
- **Cache every response**, keyed by `sha256(provider | model | prompt)`, in `llm_cache`. Re-runs are free and deterministic.
- **Provenance:** store `provider` + `model` on every ledger record, answer, and brief. Final answers/brief prefer a single model; if a fallback was used, set `fallback_used=1` and flag that question.
- **Keys:** `.env` only (`GEMINI_API_KEY`, `GROQ_API_KEY`); `.gitignore`; never printed or logged.
- Token accounting from provider `usage`, never tiktoken for these models:
  - Gemini: `usage_metadata.prompt_token_count`, `candidates_token_count`, `total_token_count`, **`thoughts_token_count`** (`usageMetadata.thoughtsTokenCount`). **Log it and bill it as output** in cost estimates.
  - Groq: `usage.prompt_tokens`, `completion_tokens`, `total_tokens`.
- **Thinking level (v2.3):** test which `thinking_level` values the API accepts; a blog reports `"minimal"` errors. Record the result as **VERIFIED / NOT VERIFIED** (currently **NOT VERIFIED**).

### 1.4 Feasibility (v2 amendment 1)

**Measured corpus size (VERIFIED):**

| Measure | 138 meetings | >8K meetings |
|---|---|---|
| chars/4, full file | **1,117,522 (~1.1M)** | **72 / 138** |
| tiktoken o200k, full file | ~1.22M | 79 / 138 |
| tiktoken o200k, dialogue + speaker labels | **971,665** | **57 / 138** |

- A **single request per meeting does not fit Groq free (8K TPM)**: 72/138 exceed 8K on chars/4 (57/138 on the dialogue form). Sending whole meetings or a growing agent history triggers **HTTP 413**.
- **Windowed ledger:** ~3K-token windows with 300-token overlap = **~416 windows** for 138 meetings (avg **3.01 / meeting**). Each window is one LLM call; every window includes `turn_id`s.
- **Redesign that makes it fit:** windowed ledger + trimmed tool results + history summarization + **prompt budget < 6K** + response cache. Retrieval returns turn-level hits, not whole transcripts.

**Free vs paid (whole project):**

| Path | Verdict |
|---|---|
| Groq free, full 138-meeting ledger | **Not feasible in one day**: ~1.25M ledger input tokens vs 200K TPD ≈ **~6+ days**; also 1,000 RPD ceiling. |
| Gemini free, full ledger | **Conditional / NOT VERIFIED**: limits unpublished. If AI Studio allows ~1.3M input + ~0.4M output in a day, it fits; otherwise split across days. Must be checked in AI Studio. |
| Free, retrieval-first only (25 Qs + brief) | **Works**: ~300K input total; likely fits Gemini free daily, borderline for Groq free (split across 2 days / cache). |
| Free, ledger on question teams only | **Works over 1–2 days**: far fewer windows than all 138 meetings. |
| **Paid Gemini 3.8 Flash, full ledger + answers + brief** | **One run**, ~1.7M input + ~0.36M output ≈ **~$2.6 lower bound** (thinking tokens billed as output; through 2026-12-31); ~$5.25 after 2027-01-01; Batch API ≈ 50% off. **ESTIMATE, not measured (no API keys).** |
| Paid Groq fallback (ASSUMPTION prices) | `gpt-oss-120b` ≈ $0.15/$0.75 → ~$0.6; `gpt-oss-20b` ≈ $0.10/$0.50 → ~$0.4. Mark ASSUMPTION. |

- **$2.6 is a lower bound:** it excludes `thoughts_token_count`, which is billed as output. Actual cost is higher.
- **Honesty:** token counts above are measured; provider-tokenizer counts, `thoughts_token_count`, and actual billed cost are **NOT VERIFIED** (no keys). Gemini model name, context window, and paid price are **VERIFIED** from official pages.

### 1.5 Build order

1. **Retrieval-first end-to-end** on the 25 questions with the citation gate (no ledger).
2. **Then** the windowed ledger on all 138 meetings (cached).
3. **Compare** scores with vs without ledger; keep the better result per question.

### 1.6 Role handling

- **Speaker letter is authoritative.**
- **Role is attached only when explicitly stated in that meeting.**
- **Letter stability across meetings is an ASSUMPTION (not proven).** Regex found roles in 26/35 teams; 9 none: `ES2004, ES2005, ES2013, IS1001, TS3004, TS3006, TS3008, TS3010, TS3011`.

### 1.7 Number rule (v2.3)

- **Prefer a total actually stated in the transcript.** If the transcript states the total, quote it; do not recompute.
- **Arithmetic in code only when no total is stated.** Then the LLM extracts components, code verifies quotes + computes; never trust LLM arithmetic.
- Never invent a number; every number must trace to a verified quote or to code arithmetic over verified quotes.

---

## 2. DB DDL (SQLite + Chroma)

```sql
PRAGMA journal_mode = WAL;

CREATE TABLE meetings (
  meeting_id      TEXT PRIMARY KEY,
  team            TEXT NOT NULL,
  letter          TEXT NOT NULL,              -- 'a'|'b'|'c'|'d'
  meeting_number  INTEGER,
  n_turns         INTEGER
);

CREATE TABLE turns (
  turn_id     TEXT PRIMARY KEY,               -- e.g. 'TS3012a#00042'
  meeting_id  TEXT NOT NULL REFERENCES meetings(meeting_id),
  team        TEXT NOT NULL,
  letter      TEXT NOT NULL,
  seq         INTEGER NOT NULL,               -- 0-based order within meeting
  timestamp   TEXT,
  speaker     TEXT NOT NULL,
  text        TEXT NOT NULL,                  -- EXACT original turn text
  char_start  INTEGER,
  char_end    INTEGER,
  n_words     INTEGER NOT NULL
);
CREATE INDEX idx_turns_meeting ON turns(meeting_id, seq);
CREATE INDEX idx_turns_team    ON turns(team, letter, seq);

CREATE VIRTUAL TABLE turns_fts USING fts5(
  text,
  turn_id    UNINDEXED,
  meeting_id UNINDEXED,
  team       UNINDEXED,
  tokenize = 'porter'
);

-- Raw decisions only; status is DERIVED at query time (v2 amendment 3).
CREATE TABLE decisions (
  decision_id    TEXT PRIMARY KEY,
  team           TEXT NOT NULL,
  topic          TEXT NOT NULL,
  topic_key      TEXT NOT NULL,               -- default cluster key (scope-aware map below)
  decision       TEXT NOT NULL,
  meeting_id     TEXT NOT NULL,
  meeting_letter TEXT NOT NULL,
  turn_id        TEXT NOT NULL,
  speaker        TEXT,
  quote          TEXT NOT NULL,               -- exact, >=6 words, passes validator
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  extracted_at   TEXT NOT NULL
);
CREATE INDEX idx_dec_team_topic ON decisions(team, topic_key, meeting_letter);

-- Scope-aware topic_key mapping (v2.3): one row per decision per scope.
CREATE TABLE topic_key_map (
  team        TEXT NOT NULL,
  scope       TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
  topic_key   TEXT NOT NULL,
  topic       TEXT NOT NULL,
  PRIMARY KEY (team, scope, decision_id)
);

-- Supersession is stored ONLY as edges; no status column.
CREATE TABLE decision_supersedes (
  newer_id            TEXT NOT NULL REFERENCES decisions(decision_id),
  older_id            TEXT NOT NULL REFERENCES decisions(decision_id),
  superseding_letter  TEXT NOT NULL,          -- letter of newer decision
  scope               TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider            TEXT,
  model               TEXT,
  created_at          TEXT NOT NULL,
  PRIMARY KEY (newer_id, older_id, scope)
);

CREATE TABLE actions (
  action_id      TEXT PRIMARY KEY,
  team           TEXT NOT NULL,
  owner          TEXT,                        -- speaker letter (name only if explicit)
  task           TEXT NOT NULL,
  meeting_id     TEXT NOT NULL,
  meeting_letter TEXT NOT NULL,
  turn_id        TEXT NOT NULL,
  quote          TEXT NOT NULL,
  provider       TEXT NOT NULL,
  model          TEXT NOT NULL,
  extracted_at   TEXT NOT NULL
);
CREATE INDEX idx_act_team_letter ON actions(team, meeting_letter);

-- Closure is an edge; status derived at query time (v2.3).
CREATE TABLE action_closed_by (
  action_id      TEXT NOT NULL REFERENCES actions(action_id),
  closer_turn_id TEXT NOT NULL,
  closing_letter TEXT NOT NULL,
  scope          TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider       TEXT,
  model          TEXT,
  created_at     TEXT NOT NULL,
  PRIMARY KEY (action_id, closer_turn_id, scope)
);

CREATE TABLE disagreements (
  disagreement_id TEXT PRIMARY KEY,
  team            TEXT NOT NULL,
  topic           TEXT NOT NULL,
  who             TEXT,
  position        TEXT,
  meeting_id      TEXT NOT NULL,
  meeting_letter  TEXT NOT NULL,
  turn_id         TEXT NOT NULL,
  quote           TEXT NOT NULL,
  provider        TEXT NOT NULL,
  model           TEXT NOT NULL,
  extracted_at    TEXT NOT NULL
);
CREATE INDEX idx_dis_team_letter ON disagreements(team, meeting_letter);

-- Resolution is an edge; status derived at query time (v2.3).
CREATE TABLE disagreement_resolved_by (
  disagreement_id  TEXT NOT NULL REFERENCES disagreements(disagreement_id),
  resolver_turn_id TEXT NOT NULL,
  resolving_letter TEXT NOT NULL,
  scope            TEXT NOT NULL CHECK(scope IN ('all','asof_c')),
  provider         TEXT,
  model            TEXT,
  created_at       TEXT NOT NULL,
  PRIMARY KEY (disagreement_id, resolver_turn_id, scope)
);

CREATE TABLE citations_log (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id      TEXT,
  question_id TEXT,
  meeting_id  TEXT NOT NULL,
  turn_id     TEXT,
  quote       TEXT NOT NULL,
  verified    INTEGER NOT NULL,               -- 0/1
  method      TEXT                            -- 'word_boundary_single_turn_match'
);

CREATE TABLE llm_cache (
  cache_key     TEXT PRIMARY KEY,             -- sha256(provider|model|prompt)
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  prompt_hash   TEXT NOT NULL,
  response_json TEXT NOT NULL,
  usage_json    TEXT,
  created_at    TEXT NOT NULL
);

CREATE TABLE answer_runs (
  run_id        TEXT PRIMARY KEY,
  question_id   TEXT,                         -- NULL for brief
  kind          TEXT NOT NULL CHECK(kind IN ('question','brief')),
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  fallback_used INTEGER NOT NULL,
  usage_json    TEXT,                         -- includes thoughts_token_count for Gemini
  created_at    TEXT NOT NULL
);

-- Ledger records whose quote fails verify() are dropped and counted (v2.3).
CREATE TABLE ledger_drops (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  team        TEXT,
  meeting_id  TEXT,
  kind        TEXT NOT NULL,                  -- 'decision'|'action'|'disagreement'
  reason      TEXT NOT NULL,                  -- 'too_short'|'not_found'|'out_of_scope'
  raw_json    TEXT,
  created_at  TEXT NOT NULL
);
```

**Chroma collection `turns_embeddings`** (name is 3–512 chars `[a-zA-Z0-9._-]`):
- `ids` = centre `turn_id`
- `embeddings` = 384-dim (`bge-small-en-v1.5`)
- `documents` = **context-window text** (2 turns before + centre + 2 after, with speaker labels)
- `metadatas` = `{meeting_id, team, letter, seq, speaker, timestamp, n_words}` (scalar values only)

**Vector indexing rule (v2 amendment 5):**
- Embed **context windows**, but `turn_id` = **centre turn**.
- Embed **only centres with >=6 words** (citable). All turns — including <6-word turns — stay in **FTS5**.
- **Sanitize FTS5 queries:** strip/escape FTS operators (`" * ( ) : ^ -`), wrap user terms in quotes, pass as bound parameters (prevents syntax errors/injection).

---

## 3. Tool signatures (plain function-calling)

**`team` and `as_of` are code-bound — they are NOT in the schemas the LLM sees.** The executor injects them from the run context (v2 amendment 2).

```python
# Executor context: ctx = {team, as_of, mode}
# mode in {'answer','brief','eval'}; ctx.team=None allowed ONLY when mode='eval'.

def search(query: str, meeting_letter: str | None = None, k: int = 8) -> list[TurnHit]:
    """Hybrid FTS5(BM25) + Chroma vector, reciprocal-rank fusion.
    Executor injects ctx.team filter (none in eval mode); optional meeting_letter <= ctx.as_of.
    Returns turn-level hits only (never big chunks)."""

def get_turns(meeting_id: str, seq_start: int | None = None, seq_end: int | None = None,
              speaker: str | None = None) -> list[Turn]:
    """Fetch exact turns for context. Exact text preserved.
    Raises ScopeError if meeting.letter > ctx.as_of, or if ctx.team is set and
    meeting.team != ctx.team. In eval mode (ctx.team=None) no team filter applies."""

def ledger(topic: str | None = None, kind: str = "all") -> list[LedgerRecord]:
    """Query decisions/actions/disagreements for ctx.team, letters <= ctx.as_of.
    Status derived at query time from edges; never stored."""

def verify(meeting_id: str, quote: str) -> VerifyResult:
    """Deterministic. Word-boundary, case/space-insensitive match inside ONE turn.
    Raises ScopeError if meeting.letter > ctx.as_of, or if ctx.team is set and
    meeting.team != ctx.team. In eval mode no team filter applies."""
```

**Dataclasses**
```python
TurnHit      = {turn_id, meeting_id, team, letter, seq, speaker, timestamp, text, score, n_words}
Turn         = {turn_id, meeting_id, team, letter, seq, timestamp, speaker, text, n_words}
LedgerRecord = {kind, id, topic, text, status, meeting_id, meeting_letter, turn_id, speaker, quote, scope}
VerifyResult = {ok: bool, turn_id: str | None, reason: str}  # ok|too_short|not_found|bad_meeting|out_of_scope
```

**JSON schemas passed to the LLM:** `search{query, meeting_letter?, k}`, `get_turns{meeting_id, seq_start?, seq_end?, speaker?}`, `ledger{topic?, kind}`, `verify{meeting_id, quote}`. No `team`/`as_of`/`mode` parameters.

**Eval-only mode (v2.3):** `ctx.team=None` is allowed **only** when `ctx.mode='eval'` (no-team retrieval test). `answer`/`brief` runs raise `ValueError` if `team` is None.

**Scope unit test (required):** for the TS3012 brief (`team=TS3012, as_of='c', mode='brief'`), call every tool with adversarial args (including `meeting_id='TS3012d'`, `meeting_letter='d'`, queries mentioning `d`); assert **no returned `turn_id` ever starts with `TS3012d`**.

---

## 4. Agent loop (plain tool-calling)

```
def tool_rounds(messages, ctx, retrieved, max_rounds):
    for _ in range(max_rounds):
        resp = LLM.chat(messages, tools=TOOLS, tool_choice="auto")
        if not resp.tool_calls:
            return resp
        for call in resp.tool_calls:
            result = execute(call, ctx)          # executor injects team/as_of
            messages.append(tool_result(call, result))
            retrieved |= turn_ids(result)
    return None                                  # still calling tools -> reject

run(question, ctx):
  if ctx.mode != 'eval' and ctx.team is None:
      raise ValueError("team required outside eval mode")
  messages  = [system_prompt, user_question]
  retrieved = set()                              # turn_ids actually returned this run
  draft     = None

  for _ in range(MAX_STEPS=8):
      resp = LLM.chat(messages, tools=TOOLS, tool_choice="auto")
      if resp.tool_calls:
          for call in resp.tool_calls:
              result = execute(call, ctx)
              messages.append(tool_result(call, result))
              retrieved |= turn_ids(result)
          continue
      draft = parse_final(resp)                  # answer + citations[]
      break

  if draft is None:                              # MAX_STEPS hit with no final
      return {"error": ["max_steps_no_final"], "draft": None}

  # ---- CODE-ENFORCED VALIDATE GATE ----
  ok, errors, warnings = validate(draft, ctx, retrieved)

  # SOFT single_meeting -> ONE extra retrieval round, then re-validate
  if ok and "single_meeting" in warnings:
      messages.append({"role":"user","content":"Add evidence from another meeting if it exists."})
      resp = tool_rounds(messages, ctx, retrieved, max_rounds=2)
      if resp is None:
          return {"error": ["single_meeting_tool_loop"], "draft": draft}
      draft = parse_final(resp)
      ok, errors, warnings = validate(draft, ctx, retrieved)

  if not ok:                                     # ONE repair attempt
      messages.append({"role":"user","content":"Fix these citation errors: "+errors})
      resp = tool_rounds(messages, ctx, retrieved, max_rounds=2)
      if resp is None:
          return {"error": ["repair_tool_loop"], "draft": draft}
      draft = parse_final(resp)
      ok, errors, warnings = validate(draft, ctx, retrieved)

  if not ok:
      return {"error": errors, "draft": draft}   # reject, never fabricate
  return draft
```

Rules:
- The model chooses tools; **code decides** when output is acceptable.
- `team`/`as_of`/`mode` come only from `ctx`; the LLM cannot widen scope.
- Brief path: `ctx = {team:'TS3012', as_of:'c', mode:'brief'}`; `get_turns`/`ledger`/`verify` hard-exclude `d`.
- **Eval-only:** `ctx.team=None` only when `mode='eval'`; never in answer/brief.
- **`single_meeting`** is wired into `run()`: one extra retrieval round (bounded), then re-validate.
- **Repair handles `tool_calls`:** bounded tool rounds; if it is still calling tools → reject (`repair_tool_loop`).
- Termination: final draft, or `MAX_STEPS` → reject (never undefined).
- One repair loop max; on repeated failure return the draft with `error`.

**Reconcile is NOT in this loop** — it is a separate scoped pass (§7).

---

## 5. Validator algorithm (deterministic)

```
normalize(s):
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    # punctuation preserved (spec ignores only case and spacing)

verify(meeting_id, quote, ctx):
    if meeting outside ctx.team / ctx.as_of: return FAIL("out_of_scope")
    q = normalize(quote)
    if len(q.split()) < 6: return FAIL("too_short")
    for turn in turns_of(meeting_id):
        if re.search(r'(?<!\w)' + re.escape(q) + r'(?!\w)', normalize(turn.text)):
            return OK(turn.turn_id)             # word-boundary, single turn
    return FAIL("not_found")

validate(answer, ctx, retrieved):
    errors, warnings = [], []
    if sentence_count(answer.text) not in (1,2): errors.append("length")
    if not (1 <= len(answer.citations) <= 3):    errors.append("citation_count")
    for c in answer.citations:
        if c.meeting_id not in team_meetings(ctx.team): errors.append("wrong_team")
        if ctx.as_of and letter(c.meeting_id) > ctx.as_of: errors.append("as_of_violation")
        r = verify(c.meeting_id, c.quote, ctx)
        if not r.ok:
            errors.append(f"citation:{r.reason}")
        else:
            c.turn_id = r.turn_id               # authoritative: verify() wins, not the LLM
            if c.turn_id not in retrieved: errors.append("citation_not_retrieved")
    # SOFT check (v2 amendment 6): not a hard fail
    if len({letter(c.meeting_id) for c in answer.citations}) < 2:
        warnings.append("single_meeting")       # run() does ONE extra retrieval round
    return (len(errors)==0, errors, warnings)
```

Guarantees:
- Quote is **>=6 words**, **case/space-insensitive**, **word-boundary exact**, **inside one turn**.
- Punctuation preserved: matching is exact after case/space normalization.
- **The accepted `turn_id` is the one returned by `verify()`**, never the LLM's supplied id.
- Cited `turn_id` must have been **retrieved in this run** (no post-hoc fabrication).
- Multi-meeting is a **soft** signal: one extra retrieval round, then proceed with a warning.
- Accepted citations logged with `verified=1`; `run_id` links to `answer_runs`.

---

## 6. Status derivation (edge-only, query time)

**Storage:** only edges; **no status columns** (v2 amendment 3 + v2.3). All edges are scope-aware and ignored when their letter > `ctx.as_of`.

### 6.1 Decision status

```
decision_status(d, ctx):
    scope = 'asof_c' if ctx.as_of == 'c' else 'all'
    edges = decision_supersedes where scope=scope and superseding_letter <= ctx.as_of
    if exists edge (newer -> d):                 return "superseded"
    peers = decisions joined topic_key_map (team, scope) with letter <= ctx.as_of
    top   = max(peers, key=letter)
    if d.id == top.id:                           return "current"
    if tie at top letter and conflicting text:   return "disputed"
    return "superseded"
```

### 6.2 Action / disagreement status (v2.3)

```
action_status(a, ctx):
    scope = 'asof_c' if ctx.as_of == 'c' else 'all'
    if exists edge in action_closed_by where action_id=a.id, scope=scope,
       closing_letter <= ctx.as_of:              return "done"
    if a.meeting_letter <= ctx.as_of:            return "open"
    return "out_of_scope"

disagreement_status(d, ctx):
    scope = 'asof_c' if ctx.as_of == 'c' else 'all'
    if exists edge in disagreement_resolved_by where disagreement_id=d.id, scope=scope,
       resolving_letter <= ctx.as_of:            return "resolved"
    if d.meeting_letter <= ctx.as_of:            return "open"
    return "out_of_scope"
```

- `unknown` is reserved for records that fail validation — they are **dropped and counted** (`ledger_drops`), not stored.
- Edges with letter > `ctx.as_of` are ignored.

### 6.3 Scopes (v2.3)

- `scope='all'`: reconcile over all records (letters a–d).
- `scope='asof_c'`: **separate** reconcile over records with letter <= 'c' only; the LLM never sees `d`. Required for the TS3012 brief.
- Topic keys and all edges carry `scope`; reconcile is **delete-then-insert per `(team, scope)`**.
- TS3012 gets **both** scopes.

### 6.4 Rules

1. Edges are created only by the **reconcile** step (§7) from explicit decision/change/closure evidence.
2. **Latest mention is NOT automatically final**; a bare mention never creates an edge.
3. Chronological order is `a < b < c < d`; an edge is valid only if the newer letter > the older/closing letter.
4. Conflicting explicit decisions with no resolution → `disputed`; never guess a winner.
5. Every record carries `turn_id` + exact `quote` that passes the validator; otherwise dropped and counted.

---

## 7. Reconcile (team-level, scoped)

One pass **per (team, scope)** (not per question), over that scope's records only.

```
reconcile(team, scope):
    as_of   = 'c' if scope == 'asof_c' else None
    records = ledger(team, as_of)               # asof_c: letters a-c ONLY; d never loaded
    chunks  = pack(records, budget=<4K tokens)  # each chunk includes turn_ids
    proposals = []
    for chunk in chunks:
        proposals += LLM.chat(RECONCILE_PROMPT, chunk)   # Gemini -> Groq fallback
    # proposals: topic clusters, supersede edges, action closures, disagreement resolutions
    for p in proposals:
        if not schema_valid(p): drop
        for edge in p.edges:
            if not (ids_exist and newer.letter > older.letter): drop
            if not verify(edge.newer.quote):   drop
        for c in p.action_closures:
            if not (ids_exist and c.closing_letter <= as_of): drop
        for r in p.disagreement_resolutions:
            if not (ids_exist and r.resolving_letter <= as_of): drop
    delete-then-insert per (team, scope): topic_key_map, decision_supersedes,
                                          action_closed_by, disagreement_resolved_by
```

- **Scopes:** `all` = letters a–d; `asof_c` = letters a–c only (TS3012 brief). The `asof_c` LLM **never sees `d`**.
- The LLM **judges** supersession, closures, and topic clustering; **trigger phrases are hints only** ("we decided", "change to", "instead", "final", "done").
- **Code enforces:** schema, id existence, quote verification, strict letter order, `as_of` bound, `scope` consistency.
- Idempotent: delete-then-insert per `(team, scope)`; cached.

---

## 8. Brief generation (TS3012, v2.3)

**Scope:** `ctx = {team:'TS3012', as_of:'c', mode:'brief'}`; uses the `scope='asof_c'` reconcile outputs. `d` is never loaded.

**Generation is section-by-section**, each a separate LLM call with only that section's records and its own **<6K prompt budget**:
1. **Decided** — current decisions (`status=current`).
2. **Changed** — superseded decisions, old -> new with letters.
3. **Open actions** — `status=open`, with owner (speaker letter; role only if explicit).
4. **Open disagreements** — `status=open`, who + position.

**Fixed inline citation format (exact):**
```
[TS3012b: "exact quote"]
```
- Meeting id + colon + double-quoted exact quote, inside square brackets.
- Every bullet must carry **>=1** such citation.

**`validate_brief(brief, ctx)`:**
```
CITE = r'\[([A-Z]{2}\d{4}[a-d]): "([^"]+)"\]'
PREDICTIVE = r'\b(next meeting|expected to|likely|going to|will (?:be|decide|discuss|finali[sz]e))\b'

validate_brief(brief, ctx):
    errors = []
    for bullet in brief.bullets:
        refs = CITE.findall(bullet.text)
        if not refs:
            errors.append("uncited_bullet"); continue
        verified = 0
        for meeting_id, quote in refs:
            if meeting_id.startswith("TS3012d"):
                errors.append("asof_violation"); continue
            r = verify(meeting_id, quote, ctx)
            if r.ok:
                verified += 1; bullet.turn_ids.append(r.turn_id)
            else:
                errors.append(f"citation:{r.reason}")
        if verified < 1:
            errors.append("no_verified_citation")
    if re.search(r'\bTS3012d\b', brief.raw_text):
        errors.append("asof_violation")
    prose = CITE.sub('', brief.raw_text)          # ignore quotes for prediction check
    if re.search(PREDICTIVE, prose, re.I):
        errors.append("predicts_final_meeting")
    return (len(errors)==0, errors)
```

- Reject **any** `TS3012d` reference anywhere in the brief.
- Reject **uncited bullets**.
- **No prediction of the final meeting:** the prompt forbids it; the validator rejects predictive phrases (heuristic, stated as such). The brief is pre-meeting, not a forecast.
- Every accepted citation is logged; `provider`/`model`/`fallback_used` recorded in `answer_runs`.

---

## 9. Model-vs-code table

| Task | Model | Code | Why |
|---|---|---|---|
| File read / parse / IDs | | ✅ | deterministic |
| Turn chunking + metadata | | ✅ | deterministic |
| Context-window build | | ✅ | deterministic |
| FTS5 + vector retrieval | | ✅ | deterministic ranking |
| Embeddings | | ✅ (fastembed) | local, reproducible |
| Understanding conversation | ✅ | | language |
| Indirect reference resolution | ✅ | | semantic |
| Decision / action / disagreement extraction | ✅ | | interpretation |
| Topic clustering + supersession/closure proposal | ✅ | | reasoning over history |
| Reconcile edge/status validation | | ✅ | rules |
| Status derivation (decision/action/disagreement) | | ✅ | edges + `as_of` |
| Draft answer | ✅ | | generation |
| Brief section assembly | ✅ | | language |
| **Brief citation parse + verify** | | ✅ | exact, per bullet |
| **Exact quote verification** | | ✅ | must be exact; no hallucination |
| **Citation count / team / as_of / retrieved gate** | | ✅ | rules |
| **Arithmetic / unit normalization** | | ✅ | correctness |
| Number: prefer stated total; code arithmetic only if none | ✅ (extract) | ✅ (compute) | correctness |
| Token/time measurement | | ✅ | from `usage` |
| JSON schema validation | | ✅ (Pydantic) | rules |

---

## 10. Scale plan (10,000 meetings)

**Measured base (Phase 1 v2):** 138 meetings, 48,148 turns, **23,821 citable (>=6 words) = 172.6/meeting**.

**Measured embedding throughput (this machine, CPU, 8 threads, ONNX):**
- Single-turn text: **9.95 turns/sec** (batch 32; 8.29 @64, 6.67 @128).
- **5-turn context window (avg 593 chars): 5.56 centres/sec** ← used below (matches §2 indexing).

| Scale | Turns | Citable centres | Raw vectors (384×4 B) | Embed time @5.56 centres/s |
|---|---|---|---|---|
| 138 (now) | 48,148 | 23,821 | 36.6 MB | **~71 min** |
| 1,000 | ~348,899 | ~172,616 | ~265 MB | **~8.6 h** |
| **10,000** | **~3.49M** | **~1.73M** | **~2.65 GB** | **~86 h (~3.6 days)** |

(1,000 / 10,000 are linear extrapolations from the 138-meeting measurement, not measured.)

### What breaks first

1. **Embedding compute (CPU)** — ~86 h for 1.7M context windows. Fix: GPU ONNX / hosted embeddings / batch + cache.
2. **LLM ledger rate limits** — ~416 windows for 138 meetings; ~3.0K for 1,000; ~30K for 10,000, plus reconcile (per team+scope) and brief (4 calls). Free Groq 200K TPD cannot finish 138 in a day (~6+ days). Fix: paid Gemini, caching, question-team subset.
3. **Chroma single-node** — 1.7M vectors + HNSW overhead; memory/latency degrade before storage.
4. **SQLite FTS5 at ~3.5M rows** — reads fine; single-writer ingestion serializes; needs batched writes.
5. **Citation validation** — cheap (indexed, word-boundary search) if it never loads whole meetings.

### Migration path
- **Vectors:** Chroma -> **Qdrant** or **pgvector**; partition by `team`; batch upserts; keep `turn_id` as join key.
- **Keyword:** Postgres `tsvector`/GIN (or sharded FTS5).
- **Embeddings:** GPU ONNX / hosted; persistent cache keyed by `turn_id`.
- **Ingestion:** queue + workers, idempotent upserts, resume-from-checkpoint.
- **Ledger/reconcile:** precompute once per `(team, scope)`, cache; never recompute on query.
- **Serving:** read replicas; push metadata filters into the vector DB.

---

## 11. Eval plan

1. **Custom questions:** >=10, manually verified ground truth + gold `turn_id`s.
2. **Scoring:** answer correctness (manual rubric), citation validity (validator), citation support.
3. **NO-TEAM-FILTER retrieval test (>=5 Qs):** `ctx.mode='eval', ctx.team=None`; `search(query, k=8/16)` without team filter; **recall@k** vs team-filtered; report the gap. Answer/brief runs must reject `team=None`.
4. **With vs without ledger:** score retrieval-first baseline, then ledger-augmented; per-question deltas.
5. **Unit tests:** (a) TS3012d never returned by any tool under brief scope; (b) word-boundary quote match rejects partial words; (c) status derivation ignores edges > `as_of`; (d) reconcile drops invalid/letter-order edges; (e) `draft is None` on MAX_STEPS; (f) repair with persistent `tool_calls` rejects; (g) `validate_brief` rejects uncited bullets, `TS3012d`, and predictive phrases; (h) `verify()`-returned `turn_id` overrides the LLM's.
6. **Failure analysis:** 3 genuine failures with category (retrieval miss, wrong supersession, indirect reference, number error, unsupported claim).
7. **Cost/time for 1,000 meetings:** tokens from `usage` (incl. `thoughts_token_count`) x pricing; embedding time from **5.56 centres/s**; mark assumptions/unverified.
8. **Ledger drops:** report count by reason from `ledger_drops`; a high drop rate is a quality signal.

---

## 12. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Gemini free limits unpublished | rate-limit surprises | check AI Studio; mark NOT VERIFIED; retry/backoff; fallback Groq |
| Groq free TPD (200K/day) too small for full ledger | ledger stalls | windowing + cache + question-team subset + paid Gemini |
| Gemini thinking tokens billed as output | cost understated | log `thoughts_token_count`; $2.6 is a lower bound |
| `thinking_level` values rejected by API | failed calls | test live; record VERIFIED/NOT VERIFIED; default to a safe value |
| Gemini->Groq fallback changes answer style | inconsistent output | single-model preference per answer; `fallback_used` flag; cache |
| Prompt >8K on fallback → 413 | failed call | hard prompt budget <6K; trim tool results/history |
| CPU embedding ~5.56 centres/s | 10k = ~86 h | GPU / hosted / cache / citable-only |
| Chroma single-node at 1.7M vectors | latency/memory | migrate to Qdrant/pgvector |
| Scope leakage into `asof_c` reconcile | brief cites `d` | separate scoped reconcile; delete-then-insert per (team, scope); validator rejects `TS3012d` |
| Brief predicts final meeting | violates task | prompt forbids; validator rejects predictive phrases |
| Letter stability is an ASSUMPTION | role errors | attach role only when explicit in that meeting |
| Supersession ambiguity | wrong "current" | edge-only + explicit-change evidence; `disputed`/`unknown` |
| Ledger quote not from centre turn | missed evidence | quote from nearest citable turn in window; windows include turn_ids |
| Spoken number parsing | wrong costs | prefer stated total; code arithmetic only if none; verify quotes |
| FTS5 operator injection / syntax errors | failed search | sanitize + quote + bound params |
| LLM tool-call errors | bad loops | MAX_STEPS + validate gate + bounded repair |

---

## 13. Changes vs v2.2 (PATCH v2.3) + conflicts

1. **Action/disagreement status removed** from storage; added `action_closed_by` / `disagreement_resolved_by` edge tables; `open`/`done`/`resolved` derived at query time, ignoring edges with letter > `ctx.as_of`.
2. **Scoped reconcile:** added `scope` (`all` | `asof_c`) to `topic_key_map` and all edge tables; TS3012 gets a **separate `asof_c` reconcile over letters a–c only** (LLM never sees `d`); delete-then-insert per `(team, scope)`.
3. **New §8 Brief generation:** section-by-section (decided / changed / open actions+owner / open disagreements) within <6K per call; fixed `[meeting: "quote"]` format; `validate_brief()` requires >=1 verified citation per bullet, rejects uncited bullets, any `TS3012d`, and predictive phrases; no prediction of the final meeting.
4. **Token accounting:** Gemini `thoughts_token_count` logged and billed as output; **$2.6 is now a lower bound**; `thinking_level` accepted values = **NOT VERIFIED**.
5. **Ledger quote rule:** a supporting quote may come from the **nearest citable (>=6-word) turn in the window**; windows include `turn_id`s; records whose quote fails `verify()` are **dropped and counted** in `ledger_drops`.
6. **Validator:** accepted `turn_id` comes from `verify()`, not the LLM; `single_meeting` wired into `run()` as one extra retrieval round; repair step handles `tool_calls` with bounded rounds or rejects.
7. **Eval-only mode:** `ctx.team=None` allowed **only** when `mode='eval'`; answer/brief raise otherwise.
8. **Number rule re-added:** prefer a total actually stated in the transcript; code arithmetic only when none is stated.
9. **DDL:** removed `status` from `actions`/`disagreements`; added `topic_key_map`, `action_closed_by`, `disagreement_resolved_by`, `ledger_drops`; `decision_supersedes` now carries `scope`.

**Conflicts with raw files:** none. The 72/138 >8K figure is exact on chars/4 full-file; on dialogue tiktoken it is 57/138 (measure clarified). No new conflict with `questions.json`, transcripts, or the PDF.

---

## 14. Open / NOT VERIFIED

- Gemini free-tier RPM/RPD/TPM (AI Studio only).
- `thinking_level` accepted values (test against the live API; a blog reports `"minimal"` errors).
- Actual billed cost, `thoughts_token_count`, and provider-tokenizer counts (no keys); **$2.6 is a lower bound**.
- Letter stability across meetings (assumption).
- 1,000/10,000 embed times are linear extrapolations.
