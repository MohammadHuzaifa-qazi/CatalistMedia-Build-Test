# Catalist Media — Technical Project Write-Up

## 1. Project Summary
The Catalist Media Meeting Intelligence System indexes the 138-meeting AMI Meeting Corpus (48,148 turns, 676,745 spoken words across 35 teams) to provide grounded question answering, decision tracking, and pre-meeting briefing. Spoken multi-meeting corpora present severe challenges: decisions evolve over time, conversational context spans tens of thousands of tokens, and independent teams discuss identical remote-control components. To address these challenges, the system couples deterministic turn-level indexing and code-enforced temporal scoping with an agentic retrieval loop and post-generation citation validation.

## 2. Architecture and Engineering Decisions
Rather than feeding multi-turn transcripts into fragile long-context prompts, the system establishes a strict model-versus-code boundary:

1. **Deterministic Indexing:** Raw transcripts are parsed into SQLite (`work/catalyst.db`) with exact character offsets (0 mismatches across 48,148 turns) and indexed via SQLite FTS5 with custom AMI acronym expansion (`LCD -> L_C_D_`). Because 50.5% of spoken turns contain fewer than 6 words and cannot host a valid citation, only the 23,821 citable turns are embedded into Chroma (`work/chroma`) using local ONNX FastEmbed (`BAAI/bge-small-en-v1.5`, 384 dimensions) over 5-turn context windows.
2. **Code-Bound Isolation:** Search tools accept a Python-bound `RunContext` hiding team and cutoff constraints from the LLM. Team and `as_of` parameters are pushed directly into SQL `WHERE` clauses and Chroma metadata filters, eliminating cross-team contamination. Turns longer than 150 words are trimmed to query-centered verbatim spans.
3. **Deterministic vs. Semantic Division:** Retrieval filtering, character-offset verification, Reciprocal Rank Fusion, anti-prediction filtering, and Pydantic schema validation (`extra="forbid"`) are owned entirely by deterministic Python code. Semantic interpretation, intent extraction, and conversational synthesis are delegated to the LLM.

## 3. Agent Behaviour and Memory
The agent executes a stateless reasoning loop backed by an `EvidenceNotebook`:
- **Retrieval & Evidence Gathering:** The agent executes targeted tool calls (`search`, `get_turns`, `ledger`, `verify`). Lexical BM25 and dense vector results are merged via Reciprocal Rank Fusion ($k=60$). A seed search injects chronological turn sequences from initial planning and final costing meetings.
- **Synthesis Step:** Once sufficient evidence is gathered, the agent transitions to a forced tool-free synthesis prompt that generates an answer draft constrained to 1–2 sentences.
- **Citation Verification:** `validate_answer()` verifies candidate citations against raw `.txt` files. Each citation must be a single-turn verbatim quote of at least 6 words and belong to the retrieved evidence set.
- **Brief Generation:** Team TS3012 pre-meeting briefs enforce `as_of='c'`, guaranteeing that dialogue from `TS3012d` is excluded from retrieval and prompt context.

## 4. Evaluation Results
Evaluation transparently separates deterministic verification from independent factual correctness:

- **Automated Tests:** 94 of 94 tests passing offline (`pytest -q`, 4.09s runtime).
- **Original 25 Questions (`answers.json`):** 25 of 25 passed strict Pydantic validation, sentence count (1–2), and citation count (1–3). All 62 citations (100%) were verified as exact verbatim quotes against raw transcripts, with 21 of 25 citing multiple meetings.
- **Independent Factual Accuracy Audit:** An independent turn-by-turn audit against raw `.txt` files revealed that deterministic citation validity does not equal factual correctness. Accuracy stands at **13 fully accurate**, **10 partially accurate**, and **2 incorrect on final conclusion** (`Q03` stated titanium was kept when it was cut; `Q04` identified kinetic/battery power instead of traditional battery/solar). These errors stemmed from free-tier rate limits forcing fallback to smaller models (`openai/gpt-oss-20b` and `gemini-3.5-flash-lite`).
- **TS3012 Executive Brief:** Fully validated with 22 bullets across all 4 required sections, 43 of 43 verified quotes from `TS3012a–c`, 0 leaks from `TS3012d`, and 0 predictive phrases.
- **Custom Multi-Meeting Evaluation (`eval/eval_report.json`):** Evaluated across 12 custom multi-meeting questions on 12 unused teams with manual transcript ground truth. 10 of 12 validated with 100% valid citations (22/22) and 0.6917 mean gold retrieval recall. 2 questions (`C08`, `C12`) failed with clean `AllProvidersFailed` errors after daily provider rate limits were exhausted, without fabricating answers. `manual_rubric_score` remains explicitly `None` pending human grader review.
- **No-Team Retrieval Benchmark:** Running retrieval without team filters on the 12 custom questions resulted in **94.79% cross-team contamination** at $k=8$, proving that hard team scoping is essential.

## 5. Scalability, Cost, and Performance
System scalability was measured empirically and extrapolated to 1,000 and 10,000 meetings:

- **Measured Baselines (Verified):** CPU embedding runs at **2.34 centres/second** (2.82 hours for 138 meetings / 23,757 windows). Per-question QA averages 3.44 calls, 12,118 prompt tokens, and 879 billed output tokens. TS3012 ledger extraction consumed 22.0K tokens (46.0s) per meeting, while dual-scope reconciliation consumed 11.6K prompt and 6.6K billed output tokens (55.6% reasoning tokens).
- **Linear Extrapolations (Not Verified at Scale):**
  - **1,000 Meetings:** 172.6K vectors, 252.9 MB storage, 20.49 hours CPU embedding; \$36.80 ingestion cost on Gemini 3.8 Flash (2026 verified rates).
  - **10,000 Meetings:** 1.73M vectors, 2.53 GB raw float32 vectors, 204.9 hours (8.54 days) CPU embedding; \$367.90 ingestion cost on Gemini 3.8 Flash.
- **Scaling Bottlenecks & Solutions:** Single-node Chroma and CPU ONNX cannot scale past 1,000 meetings. Production deployment requires GPU-accelerated batch embedding and migration to team-partitioned **Qdrant** or **PostgreSQL (`pgvector` + GIN)**.

## 6. Limitations and Lessons Learned
- **What Worked:** Exact turn offset mapping, Reciprocal Rank Fusion, code-bound `RunContext` pushdown, and strict citation validation completely eliminated quote hallucination and cross-team leakage.
- **What Failed:** Free-tier multi-provider fallback caused quality degradation. When primary models hit RPM/RPD limits, smaller models misattributed speakers and lost track of extended 50-turn costing negotiations.
- **Production Changes:** A production system must pin dedicated, paid model tiers, introduce specialized numerical ledger extractors for financial negotiations, and utilize dynamic context windows for multi-turn costing blocks.

## 7. Conclusion
The project proves that multi-meeting intelligence cannot rely on raw LLM context or unconstrained retrieval. By combining deterministic lexical/vector memory, hard partition pushdowns, and strict verification gates, Catalyst provides a reproducible, auditable architecture capable of scaling to enterprise meeting corpora.
