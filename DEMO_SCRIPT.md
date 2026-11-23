# Catalyst — 3–5 Minute Live Demo Script

> **Goal:** Demonstrate the end-to-end grounded meeting-memory system — verified indices, live hybrid search, hard code-bound `as_of='c'` scope enforcement, deterministic citation & Pydantic verification, scoped `TS3012` pre-meeting brief, 12-question manual-gold evaluation, and honest failure/scaling analysis — all offline-safe and reproducible in 3–5 minutes without blocking on free-tier LLM rate limits.

---

## Segment 1 (0:00–0:45) — Problem, Data Facts & Test Suite

**Talking points:**
- `138` AMI transcripts, `35` teams, `48,148` speaker turns, `676,745` spoken words — too large to stuff into prompts, with production targeting `10,000` meetings.
- Two verified Phase 1 findings shaped the design:
  1. **`50.5%` of turns have `<6` words** (median `5`) and cannot host a 6-word citation -> we index turn-level records in SQLite + FTS5 (`48,148` rows) and embed only the `23,821` citable (`>=6`-word) 5-turn context windows in Chroma (`BAAI/bge-small-en-v1.5`).
  2. **`0` digits in spoken text** -> all prices and totals are spoken words (`"fifteen point five"`, `"two to three Euro"`), so retrieval combines FTS5 OR + AMI acronym expansion (`LCD -> L_C_D_`) with semantic search and latest-meeting costing seeding.

**Command to run (~4–6 s, 0 LLM calls):**
```powershell
python -m pytest -q
```
*(Expected: `94 passed` — covers parsing offsets, FTS5, Chroma `as_of` pushdown, loud vector errors, multi-provider rate limits / OTPM / 5xx request refund / secret redaction, scoped reconciliation, stateless agent loop, brief validation, strict Pydantic `extra="forbid"`, and evaluation/cost estimation.)*

---

## Segment 2 (0:45–1:45) — Live Hybrid Search & Hard `as_of='c'` Scope Enforcement

**Talking points:**
- Show real hybrid RRF retrieval (SQLite FTS5 + Chroma `fastembed`).
- Prove hard code-bound scoping: under `TS3012` brief scope (`as_of='c'`), even an adversarial query taken directly from `TS3012d` returns **`0` `TS3012d` turns** because `team` and `as_of` are injected in Python and pushed into SQL `WHERE` and Chroma `$and`/`$in` filters before candidate pooling.

**Command to run live (~5 s, 0 LLM calls):**
```powershell
python -c "
from catalyst import store, tools
conn = store.connect()
ctx_q = tools.RunContext(team='TS3006', as_of=None, mode='answer')
for h in tools.search(conn, ctx_q, 'LCD screen speech recognition cost', k=3):
    print(h['turn_id'], h['score'], h['speaker'], '->', h['text'][:100])
ctx_b = tools.RunContext(team='TS3012', as_of='c', mode='brief')
hits_d = tools.search(conn, ctx_b, 'scrap speech recognition keep the rubber lose the yellow color', k=8)
print('TS3012 brief hits:', len(hits_d), 'TS3012d leaks:', sum(1 for h in hits_d if h['turn_id'].startswith('TS3012d')))
"
```
*(Expected: top hit `TS3006d#00100`, and `TS3012 brief hits: 8 TS3012d leaks: 0`.)*

---

## Segment 3 (1:45–2:45) — Deterministic Citation Gate & Strict Assembly

**Talking points:**
- Show `catalyst.pipeline assemble` re-verifying all `25/25` answers (`62/62` verbatim quotes, `1–2` sentences, `1–3` citations) and the `22`-bullet `TS3012` brief against `work/catalyst.db`, enforcing Pydantic `extra="forbid"`, and separating `answers.json` from `answers_provenance.json`.
- Mention concrete fixes verified against raw transcripts (`Q14` `7.6 -> 8.2` Euros, `Q15` `15.5 / 12.5 / LCD`, `Q18` Speaker C `"finder function"`, `Q25` fruit colors/prints instead of physical fruit shapes) alongside transparent residual misses (`Q03`, `Q04`).

**Command to run live (~1 s, 0 LLM calls):**
```powershell
python -m catalyst.pipeline assemble
```
*(Expected: `"ok": true, "wrote_answers": true, "verified_count": 25, "total_questions": 25, "brief_status": "verified"`.)*

---

## Segment 4 (2:45–3:45) — Scoped `TS3012` Ledger, Edge-Only Status & Pre-Meeting Brief

**Talking points:**
- Show `TS3012` query-time status derivation (`decision_supersedes`, `action_closed_by`, `disagreement_resolved_by` separated cleanly into `scope='asof_c'` vs `scope='all'`, `0` `TS3012d` edges in `asof_c`).
- Show all 4 brief section prompts (`decided`, `changed`, `open_actions`, `open_disagreements`), every prompt `<2.6K` tokens, `has_d=False`, grounded in explicit `TS3012a` speaker roles (`A=PM, B=UI, C=Industrial, D=Marketing`), producing `22` verified bullets in `work/brief_TS3012.json` (`43/43` quotes verified, `0` `TS3012d` mentions, `0` predictions).

**Command to run live (~1 s, 0 LLM calls):**
```powershell
python -c "
from catalyst import brief, store, tools
conn = store.connect()
ctx = tools.RunContext(team='TS3012', as_of='c', mode='brief')
inp = brief.gather_section_inputs(conn, ctx)
print('Roles:', inp['roles'])
for s in ('decided', 'changed', 'open_actions', 'open_disagreements'):
    p = brief._build_section_prompt(s, 'TS3012', 'c', inp)
    print(f'{s}: items={len(inp[s])} est_tokens={brief._estimate_tokens(brief.SYSTEM_PROMPT+p)} has_d={\"TS3012d\" in p}')
"
```

---

## Segment 5 (3:45–4:45) — Custom Multi-Meeting Eval, Failure Analysis & 10K-Meeting Scaling

**Talking points:**
- **12-question manual-gold custom eval (`eval/eval_report.json` + `eval/human_review_sheet.md`):** `10/12` live-completed & validated (`22/22` verbatim citations, `0.6917` gold retrieval recall on completed runs; `2/12` failed cleanly with `AllProvidersFailed`, zero fabricated answers; `manual_rubric_score` explicitly `PENDING_HUMAN_REVIEW`).
- **No-team-filter retrieval test (`12/12` manual-gold questions):** removing the team filter produces **`94.79%` cross-team contamination at `k=8`** (recall `0.1667 -> 0.0208`) and **`90.62%` at `k=16`** (`0.3167 -> 0.0958`).
- **3 genuine failure modes (`WRITEUP.md` §3):** `C05` (intra-meeting temporal confusion in a 750-turn meeting), `C10` (costing window truncated 14 turns before final cut), and pre-audit `Q16` (second-person role misattribution + missed costing).
- **1,000 / 10,000-meeting cost & time (`eval/cost_time_estimate.json`):** measured CPU embedding **`2.34 centres/s`** (`20.49 h` @ 1K, `204.91 h` / `8.54 d` @ 10K -> move to GPU/hosted batch); Gemini `thoughts_tokens` = `55.6%` of billed reconcile output; full ledger+reconcile extrapolated at **\$36.80** (1K) / **\$367.90** (10K) at verified 2026 `gemini-3.8-flash` pricing, migrating Chroma/SQLite to team-partitioned Qdrant + Postgres.

**Commands to run live (~3 s, 0 LLM calls):**
```powershell
python -m catalyst.pipeline eval --questions eval/eval_questions.json --no-team-test --k 8 --out work/metrics/demo_noteam_check.json
python -m catalyst.cost_estimate | Select-Object -First 45
```
