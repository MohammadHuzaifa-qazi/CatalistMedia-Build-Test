# FINAL AUDIT REPORT — Catalist Media Agentic AI Build Test

**Audit Date:** `2026-11-23`
**Source of Truth:** `Catalist_Agentic_AI_Intern_Build_Test.pdf`, `Catalist_AI_Intern_Build_Test_Data (1)/questions.json`, and all `138` raw AMI `.txt` transcripts.
**Policy:** Every status below is based solely on executed commands, deterministic validators, and direct inspection of raw transcripts. No transcript evidence, answers, citations, metrics, human rubric scores, or API calls were fabricated.

---

## 1. Requirement-by-Requirement Verification Table

| # | Requirement | Status | Evidence / Output File | Test Command & Result | Remaining Limitation |
|---|---|---|---|---|---|
| **1** | **Ingest & index all 138 meetings** (turn-level SQLite + FTS5 + local vector store) | **PASS** | `work/catalyst.db`, `work/chroma`, `catalyst/parse.py`, `catalyst/store.py`, `catalyst/embed.py` | Clean-setup parse + temp SQLite ingest: `meetings=138, turns=48148, fts=48148`, `0` char-offset mismatches across all `48,148` turns; prod Chroma `turns_embeddings=23,821` citable (`>=6`-word) 5-turn windows. | None. |
| **2** | **Searchable hybrid memory, stateless agent loop & multi-meeting reasoning** | **PASS** | `catalyst/tools.py`, `catalyst/agent.py`, `catalyst/status.py`, `catalyst/reconcile.py`, `tests/` | `python -m pytest -q` -> `94 passed in 4.09s`; live `DEMO_SCRIPT.md` Segment 2 hybrid RRF hit `TS3006d#00100` rank #1, `TS3012d` leaks `0/8`; `eval --no-team-test` verified `mode='answer'`/`'brief'` reject `team=None`. | Multi-meeting reasoning quality degrades when free-tier limits force lite/20B fallbacks (see §3 & §4). |
| **3** | **All 25 original questions: schema, length, verbatim citations** | **PASS** | `answers.json`, `answers_provenance.json`, `catalyst/assemble.py`, `catalyst/validate.py` | `python -m catalyst.pipeline assemble` -> `ok=True, verified_count=25/25`; Pydantic `extra="forbid"` PASS; `25/25` `1–2` sentences; `25/25` `1–3` citations; `62/62` (`100%`) verbatim raw-`.txt` matches (`60/62` exact-case, `2/62` case-normalized). | `4/25` (`Q10, Q15, Q23, Q24`) carry `single_meeting` warnings (`21/25` cite `>=2` distinct meetings). |
| **4** | **All 25 original questions: independent factual accuracy vs raw transcripts** | **PARTIAL** | `answers.json`, `answers_provenance.json`, `work/pre_audit_answers_snapshot.json` | Independent turn-by-turn audit against all 138 raw `.txt` files: improved from `4/25` pre-audit to **`13/25` fully accurate**, **`10/25` partially accurate**, **`2/25` incorrect on final conclusion** (`Q03`, `Q04`). See §3 for full 25-row table. | Exhausted daily free-tier limits across all 7 models prevented re-running the remaining partial/incorrect questions on a primary model (`gemini-2.5-flash` / `gpt-oss-120b`); `11/25` ran on `20b` and `2/25` on `3.5-flash-lite`. |
| **5** | **`TS3012` pre-final-meeting brief (`as_of='c'`, excluding `TS3012d`)** | **PASS** | `answers.json` (`brief`), `work/brief_TS3012.json`, `catalyst/brief.py`, `catalyst/reconcile.py` | `validate_brief() == True, errs=[]`; `22` bullets across all 4 required sections (`decided: 5`, `changed: 3`, `open_actions: 7`, `open_disagreements: 7`); `43/43` quotes verified (`42/43` exact-case + `1` leading-letter case diff on `TS3012b#00080`); `0` `TS3012d` references; `0` `asof_c` edge leaks; `0` predictions; generated live on `gemini-2.5-flash`. | Minor nuance: 1 `open_actions` bullet (`TS3012b#00184` / `TS3012c#00385`) attributes a PM-assigned task to Speaker A rather than the assignees (ID/Marketing); all citations remain verbatim valid. |
| **6** | **Custom evaluation (`>=10` custom multi-meeting Qs with manual ground truth)** | **PARTIAL** | `eval/eval_questions.json`, `work/eval_questions.json`, `eval/eval_report.json`, `eval/human_review_sheet.md` | `12` genuinely custom multi-meeting Qs on `12` unused AMI teams (`0` team overlap with Q01–Q25; manual raw-transcript `gold_turn_ids`). Live run: `10/12` validated (`ok=True`, `22/22` valid citations, `0.6917` gold retrieval recall on completed runs); `12/12` completed manual-gold no-team recall benchmark (`94.79%` contamination at `k=8`). | (1) `2/12` (`C08`, `C12`) failed live with `llm_error:AllProvidersFailed` after exhausting all 7 free-tier models (0 fabricated answers; resumable once quota resets). (2) `manual_rubric_score` is intentionally `None` (`PENDING_HUMAN_REVIEW` in `eval/human_review_sheet.md`) awaiting human grader sign-off. |
| **7** | **Three genuine wrong-answer analyses** | **PASS** | `WRITEUP.md` §3, `FINAL_AUDIT_REPORT.md` §5, `eval/eval_report.json`, `work/pre_audit_answers_snapshot.json` | Documented 3 authentic unmodified agent failures (`C05`, `C10` from live custom eval + pre-audit `Q16`) with exact meeting/turn IDs, raw quotes, root causes, and concrete retrieval/prompt/model fixes. | None. |
| **8** | **Provider reliability, rate limits, cooldowns, cache preservation & non-fabrication** | **PASS** | `catalyst/llm.py`, `catalyst/config.py`, `catalyst/agent.py`, `catalyst/brief.py`, `catalyst/evaluate.py`, `tests/test_llm.py`, `tests/test_agent.py`, `tests/test_evaluate.py` | `python -m pytest -q` -> `94 passed` (covering `Retry-After` compound durations, 429 cooldowns, `qwen/qwen3.8-27b` 1,000 OTPM cap, 5xx request-slot refunds, `AllProvidersFailed` non-fabrication, verified answer/brief preservation, and eval checkpoint resume). Live verified against Gemini & Groq 429/503 responses. | Free-tier daily quotas (`20` RPD/model on Gemini, `200K` TPD/model on Groq) remain an external rate ceiling requiring paid keys for uninterrupted bulk runs. |
| **9** | **Architecture accuracy, 10,000-meeting scalability & 1,000-meeting cost/time model** | **PASS** | `ARCHITECTURE.md`, `WRITEUP.md`, `eval/cost_time_estimate.json`, `work/metrics/cost_time_estimate.json`, `catalyst/cost_estimate.py` | Rewrote `ARCHITECTURE.md` to eliminate all stale claims (LangGraph/LanceDB/MiniLM/Llama/Streamlit); verified `python -m catalyst.cost_estimate` outputs measured baselines, verified vs assumption pricing, and linear extrapolations separately. | `1,000`- and `10,000`-meeting figures are linear extrapolations (explicitly tagged `NOT VERIFIED at scale`); Gemini 2.5 Flash & Groq paid prices are tagged `ASSUMPTION`. |
| **10** | **Security, documentation, 3–5 min demo & clean submission ZIP** | **PASS** | `.env.example`, `.gitignore`, `README.md`, `ARCHITECTURE.md`, `WRITEUP.md`, `DEMO_SCRIPT.md`, `PHASE1_DATASET_PROFILE.md`, `PHASE2_DESIGN.md`, `CatalistMedia_BuildTest_Submission.zip` | Switched Gemini auth to `x-goog-api-key` header + `_redact_secrets()`; regex scan over all `46` source/doc/deliverable files + every member inside the final ZIP: **`0` credential hits**, **`.env` excluded**; every `DEMO_SCRIPT.md` command executed cleanly with `0` errors. | None. |

---

## 2. Final Automated Test Count & Live vs. Mocked Distinction

- **Automated pytest suite:** `python -m pytest -q` -> **`94 passed in 4.09s`** (0 skipped, 0 failed).
- **Mocked / offline tests (`94` unit & integration tests in `tests/`):**
  - `test_parse.py`, `test_store.py`, `test_tools.py`, `test_validate.py`, `test_ledger.py`, `test_reconcile.py`, `test_brief.py`, `test_assemble.py`, `test_pipeline.py` — deterministic local SQLite/parser/scope/validation/Pydantic checks.
  - `test_llm.py`, `test_agent.py`, `test_evaluate.py` — mocked HTTP/SDK providers verifying compound `Retry-After` parsing, 429 cooldowns without wasteful retries, Groq 8K TPM clamping & `qwen/qwen3.8-27b` 750-token OTPM cap, 5xx `refund_request()`, `x-goog-api-key` header auth + secret redaction, `AllProvidersFailed` non-fabrication, verified-work preservation on forced re-runs, and incremental custom-eval resume.
- **Live-tested components (real Gemini & Groq API calls + real local SQLite/Chroma indices):**
  - Full 138-transcript ingestion (`48,148` turns, `0` offset mismatches) and full local Chroma embedding (`23,821` vectors, `2.82 h` CPU at `2.34 centres/s`).
  - Live `TS3012` ledger (`20` windows), dual-scope reconcile (`asof_c` & `all`), and 4-section pre-meeting brief (`gemini-2.5-flash`, `22` verified bullets).
  - Live agent runs across all `25` original questions (`answers.json`) and `10 / 12` custom evaluation questions (`eval/eval_report.json`), plus live verification of real HTTP 429 (`Retry-After`, OTPM, daily RPD/TPD) and HTTP 503 failover/preservation behavior (`work/metrics/llm_calls.jsonl`).
  - Live local no-team-filter retrieval recall benchmark across all `12 / 12` manually verified custom questions (`0` LLM calls, real SQLite FTS5 + Chroma).

---

## 3. Original 25-Question Verification Status (`answers.json`)

- **Deterministic validation (`assemble` + raw `.txt` check):** `25/25` `ok=True`, `25/25` `1–2` sentences, `25/25` `1–3` citations, **`62/62` (`100%`)** verbatim single-turn raw-`.txt` matches, `21/25` citing `>=2` distinct meetings, Pydantic `extra="forbid"` PASS.
- **Independent raw-transcript factual audit:** **`13/25` Fully Accurate**, **`10/25` Partially Accurate**, **`2/25` Incorrect on Final Conclusion** (improved from `4/25` fully accurate pre-audit in `work/pre_audit_answers_snapshot.json`).

| QID | Team | Final Model | Mtgs Cited | Validation | Transcript Factual Audit | Verification Notes vs. Raw Transcripts |
|---|---|---|---|---|---|---|
| **Q01** | `TS3006` | `gemini-2.5-flash` | `b, d` | PASS | **Accurate** | Dropped speech rec, kept LCD because losing it requires many extra buttons (`TS3006d#00101, #00422`). |
| **Q02** | `ES2003` | `gemini-2.5-flash` | `c, d` | PASS | **Accurate** | Spongy rubber -> plastic solid case to cut price & keep LCD (`ES2003c#00120`, `ES2003d#00073, #00111`). |
| **Q03** | `TS3009` | `openai/gpt-oss-120b` | `c, d` | PASS | **Incorrect (final outcome)** | Correctly got Speaker B proposing titanium-coloured plastic (`TS3009d#00363, #00366`), but wrongly stated titanium was retained; transcripts cut titanium (`TS3009d#00402, #00419`: `"Titanium gone and add plastic."`). |
| **Q04** | `TS3011` | `openai/gpt-oss-120b` | `c, d` | PASS | **Incorrect (energy pair)** | Said kinetic + batteries instead of **traditional batteries and solar cells** (`TS3011c#00286, #00296`) where solar cells were cut (`TS3011d#00051, #00059`). |
| **Q05** | `TS3007` | `openai/gpt-oss-120b` | `c, d` | PASS | **Partial** | Got kinetic -> batteries right, but missed combined battery+kinetic (`TS3007c#00090, #00337`) and named Speaker C instead of PM Speaker A (`TS3007d#00369`). |
| **Q06** | `ES2008` | `openai/gpt-oss-120b` | `a, d` | PASS | **Accurate** | Dropped find-the-remote (`ES2008d#00046`), Speaker C objected (IR under settee / gimmick, `ES2008d#00040`). |
| **Q07** | `IS1008` | `openai/gpt-oss-20b` | `c, d` | PASS | **Accurate** | Speaker D asked to find display cost (`IS1008c#00144`) and reported ~2–3 Euro (`IS1008d#00076`). |
| **Q08** | `ES2007` | `openai/gpt-oss-20b` | `b, c` | PASS | **Accurate** | Speaker D raised RSI (`ES2007b#00159`); team planned ergonomic anti-RSI design (`ES2007c#00142`). |
| **Q09** | `IS1002` | `openai/gpt-oss-120b` | `b, c` | PASS | **Partial** | Got 25 Euros (`IS1002b#00018`) and Speaker D pushing hardest (`89` Euros, `IS1002b#00105`); stated range `40–100` rather than `35–50` (`IS1002c#00326–#00327`). |
| **Q10** | `TS3003` | `openai/gpt-oss-20b` | `b` (`warn`) | PASS | **Partial** | Got elderly (`>60`) right; missed core reason (company already developing a product for younger customers, `TS3003b#00058, #00098`); `single_meeting`. |
| **Q11** | `ES2011` | `gemini-2.5-flash` | `b, c` | PASS | **Accurate** | PM Speaker A asked whether TV-only excluded DVD (`ES2011b#00229`), later clarified no DVD (`ES2011c#00033`). |
| **Q12** | `ES2004` | `openai/gpt-oss-20b` | `c, d` | PASS | **Partial** | Got dropped voice rec (`ES2004d#00350`); said beeper instead of distinctive object itself making it easier to locate (`ES2004d#00335`). |
| **Q13** | `TS3004` | `openai/gpt-oss-20b` | `b, c, d` | PASS | **Partial** | Got speech rec included in final design (`TS3004d#00014`); attributed last reversal to `d` rather than `c#00603` (confirmed in `d`). |
| **Q14** | `IS1009` | `openai/gpt-oss-20b` | `c, d` | PASS | **Accurate** | Final cost `8.2` Euro (`IS1009d#00209`), pending speech recognition (`IS1009c#00190`) not included (`IS1009d#00254`). *(Fixed from pre-audit `7.6`!)* |
| **Q15** | `ES2016` | `openai/gpt-oss-20b` | `d` (`warn`) | PASS | **Accurate** | First full costing `15.5` Euros (`#00245`), limit `12.5` (`#00189`), removed LCD (`#00256`). *(Fixed from 100% wrong pre-audit! `single_meeting`.)* |
| **Q16** | `TS3010` | `openai/gpt-oss-20b` | `c, d` | PASS | **Partial** | Fixed PM **Speaker A**, **push-buttons**, and **special colour** (`TS3010d#00250`), but still called old control "scroll-wheel" instead of "joystick". |
| **Q17** | `ES2009` | `openai/gpt-oss-20b` | `c, d` | PASS | **Partial** | Got `17.50` Euro touch-screen estimate (`ES2009c#00118`); said dropped in `d` for large buttons instead of `c#00193` for voice recognition. |
| **Q18** | `ES2014` | `openai/gpt-oss-20b` | `b, c` | PASS | **Accurate** | Sensor/buzzer locator -> voice recognition marketed as `"finder function"`, suggested by Speaker C (UI Designer, `ES2014c#00140`). *(Fixed from 100% wrong pre-audit!)* |
| **Q19** | `IS1003` | `openai/gpt-oss-20b` | `c, d` | PASS | **Accurate** | Wood (`IS1003c#00277`) -> plastic (`IS1003d#00384`), suggested by Speaker B. |
| **Q20** | `ES2008` | `gemini-3.5-flash-lite` | `a, d` | PASS | **Partial** | Got Speaker A `"I'll look into that"` (`ES2008a#00076`) and honestly declined to fabricate missing report-back, but missed `ES2008b#00127` (TV only). |
| **Q21** | `IS1007` | `qwen/qwen3.8-27b` | `c, d` | PASS | **Partial** | Got Speaker C (`IS1007c#00255`) and `$15` (`IS1007d#00149`); missed final decision going **without LCD** (`IS1007d#00257–#00283`). |
| **Q22** | `ES2010` | `openai/gpt-oss-120b` | `a, c, d` | PASS | **Accurate** | Speaker B lithium (`a#00091`), Speaker D triple-A lighter (`c#00227`), final lithium (`d#00145`). |
| **Q23** | `ES2006` | `qwen/qwen3.8-27b` | `d` (`warn`) | PASS | **Accurate** | Kept voice rec & LCD (`ES2006d#00417`), swapped scroll for push-buttons (`single_meeting`; also dropped sample speaker for beeper). |
| **Q24** | `ES2002` | `gemini-3.5-flash-lite` | `c` (`warn`) | PASS | **Partial** | Got standard batteries & PM liking kinetic backup (`ES2002c#00268`); missed final `"just battery"` (`ES2002d#00101–#00103`); `single_meeting`. |
| **Q25** | `ES2011` | `qwen/qwen3.8-27b` | `c, d` | PASS | **Accurate** | Decided on fruit colours/prints rather than physical fruit shapes for holding comfort (`ES2011c#00020`, `ES2011d#00228, #00240`). *(Fixed from reversed pre-audit!)* |

---

## 4. Custom Multi-Meeting Evaluation Results (`eval/eval_report.json`)

- **Dataset (`eval/eval_questions.json` & `work/eval_questions.json`):** `12` manually authored multi-meeting questions (`C01–C12`) spanning `12` AMI teams completely disjoint from `Q01–Q25` (`ES2005, ES2012, ES2013, ES2015, IS1000, IS1001, IS1004, IS1005, IS1006, TS3005, TS3008, TS3012`). Every `expected_answer`, `gold_turn_ids`, and `required_facts` entry was authored from direct inspection of raw `.txt` transcripts (zero reliance on agent-generated citations).
- **Live evaluation metrics:**
  - **Passed validation (`ok=True`):** `10 / 12` (`83.33%`)
  - **Failed (`llm_error:AllProvidersFailed`):** `2 / 12` (`C08`, `C12` — `0` fabricated answers)
  - **Citation validity:** `22 / 22` (`100.0%`) verbatim verified single-turn quotes
  - **Multi-meeting citations (`>=2` meetings):** `7 / 12` (`58.33%` overall; `7 / 10` = `70.0%` among completed runs)
  - **Mean gold retrieval recall:** `0.5764` across all 12 (`0.6917` across the 10 completed runs)
  - **Mean gold citation recall:** `0.2583` across all 12 (`0.3100` across the 10 completed runs)
  - **Mean automated `required_facts` recall:** `0.6320` across all 12 (`0.7583` across the 10 completed runs; `4` full-match, `6` partial-match, `2` failed)
  - **Transcript-grounded breakdown of the 10 live-completed answers (preliminary audit; `manual_rubric_score` remains `None` / `PENDING_HUMAN_REVIEW` in `eval/human_review_sheet.md`):** `5` correct (`C01, C03, C04, C06, C09`), `3` partially correct (`C02, C07, C11`), `2` incorrect (`C05, C10`), `2` failed due to provider exhaustion (`C08, C12`).
- **No-team-filter retrieval recall benchmark (`12/12` manual-gold questions, 0 LLM calls):**
  - `k=8`: team-filtered recall `0.1667` vs no-team recall `0.0208` (gap `+0.1459`, **`94.79%`** cross-team contamination)
  - `k=16`: team-filtered recall `0.3167` vs no-team recall `0.0958` (gap `+0.2209`, **`90.62%`** cross-team contamination)

---

## 5. Three Genuine Wrong-Answer Analyses

All three cases below are authentic, unmodified outputs verified directly against raw transcripts (full details in `WRITEUP.md` §3 and `eval/human_review_sheet.md`):

1. **`C05` (`IS1000`, live custom eval, `gemini-flash-lite-latest`):**
   - *Generated:* Claimed `IS1000` cut the LCD screen at 16 Euros, **kept speech recognition**, and compensated by **adding a protective cover** (`IS1000b#00325`, `IS1000d#00120`).
   - *Correct (raw transcripts):* Cut **both** the LCD screen **and** speech recognition to fit `12.50` Euros and added **two extra buttons** (`IS1000c#00161`, `IS1000d#00683`, `#00716`, `#00719`, `#00722`).
   - *Root cause & fix:* Intra-meeting temporal confusion inside a `750+`-turn meeting (conflating early walkthrough turn `#00120` with end-of-meeting costing `#00716–#00722` despite `0.80` retrieval recall). Addressed by late-meeting sequence markers in `EvidenceNotebook`, structured costing ledgers, and pinning a stronger synthesis model.
2. **`C10` (`TS3005`, live custom eval, `gemini-flash-lite-latest`):**
   - *Generated:* *"To bring the total cost down, they dropped the sample speaker and Speaker C proposed replacing the wood finish with a normal special colour."* (`TS3005d#00627`, `TS3005d#00640`).
   - *Correct (raw transcripts):* Dropped sample speaker/sensor (`TS3005d#00627`), then Speaker C proposed **dropping the scroll wheel and using only the touch screen**, lowering the total to **`11.50` Euros** (`TS3005c#00034`, `TS3005d#00554`, `#00627`, `#00629`, `#00654`, `#00661`).
   - *Root cause & fix:* `seq+10` neighbor expansion around `#00627` stopped at `#00637`, missing `#00654–#00661` 27 turns later (`ret_rec=0.3333`). Addressed by contiguous costing-block expansion (`seq+35` over dense currency spans) and structured `figures` ledger indexing.
3. **`Q16` (`TS3010`, pre-audit unmodified run in `work/pre_audit_answers_snapshot.json`, `openai/gpt-oss-20b`):**
   - *Generated (pre-audit):* *"The project manager, Speaker C, favored a scroll-wheel for navigation, but in the final costed design it was replaced by an LCD display, and the scroll wheel itself was given up to meet the budget."*
   - *Correct (raw transcripts):* Project Manager **Speaker A** favoured a **small joystick** (`TS3010c#00201, #00214, #00222`); replaced in final costed design by **push-buttons**, giving up **special colour** (`TS3010d#00212, #00222, #00248, #00250`).
   - *Root cause & fix:* Misattributed second-person `"You're the Project Manager"` (`TS3010b#00082`, spoken by C to A) + missed `TS3010d` costing. Implemented `_extract_explicit_roles` (first-person only) + latest-meeting costing seeding + wider neighbor expansion, which fixed Speaker A, push-buttons, and special colour on `Q16` (and fully fixed `Q14` and `Q15`).

---

## 6. `TS3012` Pre-Final-Meeting Brief Verification

- **Artifact:** `answers.json["brief"]` and `work/brief_TS3012.json` (generated live on `gemini-2.5-flash`, `5` calls, `29.27 s`, `ok=True`).
- **Temporal isolation (`as_of='c'`):** `0` references to `TS3012d` anywhere in the brief; `0` `TS3012d` edges in `decision_supersedes`, `action_closed_by`, or `disagreement_resolved_by` under `scope='asof_c'`; live adversarial search under `mode='brief', as_of='c'` returned `0` `TS3012d` leaks; `0` predictive phrases (`validate_brief() == True, errs=[]`).
- **Completeness & citation validity:** `22` bullets across all 4 required sections (`DECIDED: 5`, `WHAT CHANGED: 3`, `OPEN ACTIONS AND OWNERS: 7`, `UNRESOLVED DISAGREEMENTS: 7`), `43/43` quotes verified against `TS3012a`, `TS3012b`, and `TS3012c`.

---

## 7. Scalability & Cost Summary (`eval/cost_time_estimate.json`)

- **Measured baselines (`VERIFIED`):** `138` meetings, `48,148` turns, `23,821` citable centres (`172.62`/meeting); CPU `fastembed` `BAAI/bge-small-en-v1.5` = **`2.34 centres/s`** (`2.82 h` for 138); `TS3012` ledger = `22.0K` tokens & `46.0 s`/meeting (conservative upper bound: `5.0` windows/mtg vs `3.01` corpus avg); `TS3012` reconcile = `11.6K` prompt + `6.6K` billed output (incl. `55.6%` Gemini `thoughts_tokens`) per team-scope; QA avg across 25 passed runs = `3.44` calls, `12,118` prompt + `879` billed output tokens/question.
- **1,000-meeting linear extrapolation (`NOT VERIFIED at scale`):** `20.49 h` CPU embeddings; `21.06M` prompt + `5.60M` billed output tokens (`17.88 h` serial) -> **\$36.80** (`gemini-3.8-flash` 2026 **VERIFIED** \$0.75/\$3.75), **\$73.61** (2027 **VERIFIED** \$1.50/\$7.50), **\$20.32** (`gemini-2.5-flash` **ASSUMPTION**), **\$7.36** (`groq 120b` **ASSUMPTION**).
- **10,000-meeting readiness (`NOT VERIFIED at scale`):** `~1.73M` vectors (`2.53 GB` raw float32), `204.91 h` (`8.54 d`) CPU embeddings, **\$367.90** (2026 verified price) / **\$735.79** (2027 verified price). Architecturally sound (stateless `<6K`-token prompts, turn-level indexing), requiring GPU/hosted embeddings, team-partitioned Qdrant/Postgres (`pgvector` + `tsvector`), async worker queues, and paid Batch API at 10K scale.

---

## 8. Security Audit

- **Hardening applied during audit:** Found and fixed a transient `?key=` query-parameter leak in `catalyst/llm.py` error strings by switching Gemini auth strictly to the `x-goog-api-key` HTTP header, adding `_redact_secrets()` across all exception/logging paths (with regression test `test_gemini_uses_header_auth_and_redacts_secrets_on_network_error`), and scrubbing historical `?key=` entries from local `work/metrics/llm_calls.jsonl` (which is excluded from git and the submission ZIP).
- **Secrets & `.gitignore` check:** `.env` is listed in `.gitignore` and excluded from the submission ZIP; `.env.example` contains placeholder values only.
- **Automated credential regex scan:** Scanned all `46` source/test/doc/deliverable files + every member inside `CatalistMedia_BuildTest_Submission.zip` for Gemini keys (`AIza...`), Groq keys (`gsk_...`), OAuth tokens (`AQ.Ab8...`), private keys, and `?key=` parameters -> **`0` matches**.

---

## 9. Final Submission ZIP Manifest

- **Exact filename:** `CatalistMedia_BuildTest_Submission.zip`
- **Included (`49` files total):**
  - `catalyst/` (`18` Python modules)
  - `tests/` (`12` pytest test modules — `94` tests)
  - `eval/` (`4` files: `eval_questions.json`, `eval_report.json`, `human_review_sheet.md`, `cost_time_estimate.json`)
  - Deliverables & docs (`12` root files): `answers.json`, `answers_provenance.json`, `README.md`, `ARCHITECTURE.md`, `WRITEUP.md`, `DEMO_SCRIPT.md`, `PHASE1_DATASET_PROFILE.md`, `PHASE2_DESIGN.md`, `FINAL_AUDIT_REPORT.md`, `requirements.txt`, `.env.example`, `.gitignore`
  - Template & dataset references (`3` files): `work/eval_questions.template.json`, `Catalist_AI_Intern_Build_Test_Data (1)/CREDITS.txt`, `Catalist_AI_Intern_Build_Test_Data (1)/questions.json`
- **Explicitly excluded:** `.env`, `.git/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `work/catalyst.db`, `work/chroma/`, `work/rate_state.json`, `work/metrics/llm_calls.jsonl`, `work/pre_audit_*`, `*.pyc`, `*.tmp`, and raw AMI `.txt` transcripts (provided separately in the assignment dataset folder `Catalist_AI_Intern_Build_Test_Data (1)/transcripts/`).
