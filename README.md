# Catalist Media — Agentic AI Meeting Intelligence System

A grounded, retrieval-first meeting intelligence system built on the 138-meeting AMI Meeting Corpus (48,148 speaker turns, 676,745 spoken words across 35 product-design teams). The system provides multi-meeting temporal question answering with deterministically verified verbatim transcript citations, generates scoped pre-final-meeting briefs, and features an independent 12-question custom multi-meeting evaluation benchmark.

---

## 1. Project Overview

### Problem Statement
Modern product design and engineering teams generate dozens of hours of spoken meeting dialogue across project lifecycles. Long-context large language models (LLMs) struggle with long meeting corpora due to:
1. **Context Window Limits & Dilution:** A 4-meeting project series spans 25,000–50,000 words. Dumping multiple full transcripts into an LLM context dilutes attention, inflates inference cost, and causes "lost-in-the-middle" hallucinations.
2. **Temporal Supersession & Drift:** Decisions evolve over time. Features discussed in initial brainstorms (Meeting A or B) are frequently altered, rescinded, or cut during final costing reviews (Meeting C or D). Naive semantic search retrieves early brainstorming turns that contradict final decisions.
3. **Cross-Team Contamination:** In multi-team corpora like AMI, questions regarding "the LCD screen" or "kinetic batteries" yield identical keyword and embedding matches across 35 independent teams. Without hard scoping, multi-meeting queries contaminate reasoning with dialogue from unrelated teams.
4. **Verbatim Traceability:** Real-world meeting intelligence requires evidence-backed verification. Claims must be tied to exact, un-hallucinated spoken quotes.

### System Solution
This system implements an agentic, hybrid memory architecture combining deterministic turn parsing, SQLite FTS5 full-text search, local vector search via Chroma and FastEmbed, an Evidence Notebook retrieval loop, and strict Pydantic verification gates. The system answers cross-meeting queries, tracks decision evolution, and enforces temporal cutoff boundaries (`as_of` scoping) to prevent future information leakage.

---

## 2. Key Features

- **Exact Turn-Level Ingestion:** Ingests all 138 AMI transcripts into SQLite with exact character offsets (`0 / 48,148` offset mismatches against raw `.txt` files).
- **Hybrid Retrieval Memory:** Combines SQLite FTS5 (BM25 with AMI-specific acronym expansion such as `LCD -> L_C_D_`, `TV -> T_V_`) with local dense vector search (`BAAI/bge-small-en-v1.5` via ONNX `fastembed` in Chroma). Reciprocal Rank Fusion (RRF) merges lexically and semantically relevant turns.
- **Strict Code-Bound Scoping:** `team` and `as_of` filters are enforced directly in SQLite `WHERE` clauses and Chroma metadata filters, completely eliminating cross-team contamination.
- **Agentic Multi-Step Reasoning:** A stateless agent loop iterates over tools (`search`, `get_turns`, `ledger`, `verify`), guided by an `EvidenceNotebook` that tracks retrieved turns, chronological sequences, and candidate quotes.
- **Zero-Hallucination Citation Verification:** A deterministic post-processor validates all quotes against raw transcript files. Citations must match single-turn spoken boundaries (`>=6` words) and belong to the retrieved evidence set.
- **Pre-Meeting Brief Generator (`TS3012`):** Generates a 4-section executive briefing (`DECIDED`, `CHANGED`, `OPEN ACTIONS`, `OPEN DISAGREEMENTS`) bound strictly to `as_of='c'`, guaranteeing zero temporal leakage from final meeting `TS3012d`.
- **Multi-Provider Fallback & Resilience:** Multi-model failover ladder across Google Gemini and Groq (`gemini-3.8-flash`, `gemini-2.5-flash`, `groq gpt-oss-120b`, `gpt-oss-20b`, `gemini-3.5-flash-lite`, `groq qwen/qwen3.8-27b`, `gemini-flash-lite-latest`) with compound `Retry-After` handling, TPM token clamping, and secret redaction.
- **Evaluation & Provenance Harness:** Tracks call-by-call token usage, prompt/completion costs, fallback traces, and raw transcript recall across both the original 25 questions and a 12-question custom multi-meeting benchmark.

---

## 3. System Architecture

The architecture enforces a strict boundary between deterministic code execution and LLM semantic synthesis:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Indexing Pipeline"]
        A["138 Raw Transcripts (.txt)"] --> B["parse.py: Tokenizer & Turn Parser"]
        B -->|48,148 Turns & Offsets| C[("SQLite: catalyst.db + FTS5")]
        B -->|23,821 Citable 5-Turn Windows| D["embed.py: FastEmbed ONNX BGE-Small"]
        D --> E[("Chroma Vector Store: turns_embeddings")]
    end

    subgraph RetrievalLayer ["2. Retrieval Layer & Tool Scoping"]
        F["Agent Tool Request"] --> G{"tools.py: RunContext Enforcement"}
        G -->|team & as_of pushdown| H["FTS5 Porter Index (Acronym Expanded)"]
        G -->|team & as_of metadata filter| I["Chroma Dense Vector Search"]
        H & I --> J["Reciprocal Rank Fusion (RRF)"]
        J --> K["tools.py: get_turns & seed_search"]
        K --> L["Verbatim Trimming (<=150 words)"]
    end

    subgraph AgentLoop ["3. Agent Reasoning & Synthesis"]
        L --> M["agent.py: EvidenceNotebook"]
        M --> N["LLM Step Execution (Gemini / Groq via llm.py)"]
        N -->|Iterative tool call| F
        N -->|Finished gathering| O["SYNTHESIS_SYSTEM_PROMPT (No Tools)"]
        O --> P["Draft Answer + Candidate Citations"]
    end

    subgraph Validation ["4. Deterministic Verification & Assembly"]
        P --> Q["validate.py: validate_answer()"]
        Q -->|Quote check against raw .txt| R{"Verbatim & Retrievable?"}
        R -->|Yes| S["assemble.py: Pydantic Schema Validation"]
        R -->|No / Drift| T["Deterministic Overwrite via verify() tool"]
        T --> S
        S --> U["answers.json & answers_provenance.json"]
    end
```

### Architectural Responsibilities: Model vs. Code
| Subsystem | Owner | Execution Responsibility |
|---|---|---|
| **Parsing & Offsets** | Deterministic Code | Exact character-offset mapping against raw text; regex-based speaker extraction. |
| **Index Filtering** | Deterministic Code | Hard SQL `WHERE` and Chroma `$and` pushdown of `team` and `as_of` constraints. |
| **Search Fusion** | Deterministic Code | Reciprocal Rank Fusion ($k=60$) combining BM25 and dense vector similarities. |
| **Dialogue Comprehension** | LLM | Cross-meeting semantic reasoning, intent resolution, and synthesis. |
| **Citation Verification** | Deterministic Code | Substring match against raw `.txt` files; sentence-count verification; retrievability audit. |
| **Rate Limit / Fallback** | Deterministic Code | Exponential backoff, header token tracking, and structured failover without hallucination. |

---

## 4. Technology Stack

- **Runtime:** Python `3.12+` (Verified on Python `3.12.2`)
- **Database & Full-Text Search:** SQLite 3 with FTS5 (`porter` stemmer, custom AMI acronym expansion)
- **Vector Storage:** ChromaDB `v1.5.9` (`chromadb`)
- **Embedding Generation:** FastEmbed `v0.7.4` running ONNX Runtime `v1.23.2` (Model: `BAAI/bge-small-en-v1.5`, 384 dimensions, local CPU inference)
- **Schema & Validation:** Pydantic `v2.7+` with strict `extra="forbid"` models
- **LLM Providers & APIs:**
  - Google Gemini API (`google-genai` SDK `v1.0+` using `x-goog-api-key` header authentication)
  - Groq Cloud API (`groq` SDK `v0.11+`)
- **Testing & Verification:** Pytest `v8.0+` (94 automated unit and integration tests)
- **Token Estimation:** `tiktoken` `v0.7+`

---

## 5. Repository Structure

```
.
├── catalyst/                       # Core application package
│   ├── __init__.py
│   ├── config.py                   # Paths, model lists, rate limits, pricing tables
│   ├── parse.py                    # Transcript parsing & exact character offset calculation
│   ├── store.py                    # SQLite database schema, ingest, and FTS5 indices
│   ├── embed.py                    # FastEmbed ONNX embedding pipeline into ChromaDB
│   ├── tools.py                    # Scoped agent tools (search, get_turns, ledger, verify)
│   ├── validate.py                 # Deterministic validation for answers, citations, and briefs
│   ├── status.py                   # Dynamic temporal status derivation (current/superseded)
│   ├── llm.py                      # Multi-provider client, rate limiter, cache & failover
│   ├── ledger.py                   # Windowed extraction of decisions, actions, disagreements
│   ├── reconcile.py                # Temporal supersession and closure graph reconciliation
│   ├── agent.py                    # Stateless agent loop with EvidenceNotebook
│   ├── brief.py                    # 4-section pre-final-meeting briefing generator (as_of='c')
│   ├── assemble.py                 # Final validator and assembler for answers.json
│   ├── evaluate.py                 # Evaluation harness for custom questions and recall tests
│   ├── cost_estimate.py            # Cost and latency extrapolation model
│   └── pipeline.py                 # Unified CLI driver
├── tests/                          # 94 automated pytest unit and integration tests
│   ├── test_parse.py               # Deterministic parsing and offset tests
│   ├── test_store.py               # SQLite and FTS5 query tests
│   ├── test_tools.py               # Tool scoping, filtering, and verbatim trimming tests
│   ├── test_validate.py            # Citation and length validation tests
│   ├── test_ledger.py              # Ledger extraction tests
│   ├── test_reconcile.py           # Temporal scoping and leak prevention tests
│   ├── test_llm.py                 # Multi-provider failover, backoff, and secret redaction
│   ├── test_agent.py               # Agent loop and synthesis tests
│   ├── test_brief.py               # Brief generation and anti-prediction validation tests
│   ├── test_assemble.py            # Pydantic extra="forbid" assembly tests
│   ├── test_evaluate.py            # Evaluation metrics and checkpoint tests
│   └── test_pipeline.py            # Pipeline CLI interface tests
├── eval/                           # Evaluation benchmark data & reports
│   ├── eval_questions.json         # 12 manually verified custom multi-meeting questions
│   ├── eval_report.json            # Execution results, recall metrics, and failure logs
│   ├── human_review_sheet.md       # Transparent rubric template for human evaluation
│   └── cost_time_estimate.json     # Measured baselines and scale extrapolations
├── Catalist_AI_Intern_Build_Test_Data (1)/  # Dataset: 138 transcripts & questions.json
├── answers.json                    # Final submission deliverable (25 answers + TS3012 brief)
├── answers_provenance.json         # Provenance: model calls, tokens, and verified citations
├── ARCHITECTURE.md                 # Detailed architectural documentation
├── WRITEUP.md                      # Engineering report and wrong-answer case studies
├── DEMO_SCRIPT.md                  # 3-5 minute offline-safe evaluation script
├── FINAL_AUDIT_REPORT.md           # Comprehensive raw-transcript accuracy audit
├── PHASE1_DATASET_PROFILE.md       # Empirical dataset analysis across 138 transcripts
├── PHASE2_DESIGN.md                # System design specification
├── requirements.txt                # Python project dependencies
├── .env.example                    # Environment variable configuration template
└── .gitignore                      # Git exclusion rules (ignores .env, work/ db files)
```

---

## 6. Installation and Setup

### Prerequisites
- Python `3.12+` (tested on 3.12.2 on Windows and Linux)
- Git

### Virtual Environment Setup
```powershell
# Clone the repository
git clone <repo-url>
cd project

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment Variable Configuration
Copy the `.env.example` file to create your local `.env`:
```powershell
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

Configure your API keys in `.env` (never commit real credentials to version control):
```ini
GEMINI_API_KEY="your-gemini-api-key"
GROQ_API_KEY="your-groq-api-key"
```

### Offline Test Suite Verification
Verify the installation offline (~4 seconds, 0 API calls required):
```powershell
python -m pytest -q
```
Expected output: **`94 passed`**.

---

## 7. Running the Project

All pipeline operations run through `python -m catalyst.pipeline`:

### 1. Ingest Transcripts
Parses all 138 transcripts and builds SQLite tables and FTS5 indices:
```powershell
python -m catalyst.pipeline ingest
```
*Output: `work/meetings.jsonl`, `work/turns.jsonl`, `work/catalyst.db` (48,148 turns, 0 offset mismatches).*

### 2. Generate Vector Embeddings
Generates ONNX local embeddings for the 23,821 citable turns into Chroma:
```powershell
python -m catalyst.pipeline embed
```
*Note: This process is resumable. Embedding the full corpus on CPU takes ~2.82 hours at 2.34 centres/second. Existing indices in `work/chroma` are preserved.*

### 3. Answer Questions via Agent Loop
Run the agentic question-answering pipeline on the 25 benchmark questions:
```powershell
# Run all questions (resumable; preserves previously verified answers)
python -m catalyst.pipeline answer --no-ledger --max-steps 3 --out work/answers_work.json

# Run a specific question
python -m catalyst.pipeline answer --question Q01 --max-steps 3
```

### 4. Generate Pre-Meeting Brief (`TS3012`)
Generate the 4-section executive brief for Team TS3012 prior to meeting d:
```powershell
python -m catalyst.pipeline brief --team TS3012 --as-of c
```
*Output: `work/brief_TS3012.json` (verified against `TS3012a–c`, 0 leakage of `TS3012d`).*

### 5. Assemble and Validate Final Submission
Run strict deterministic validation and assemble the final deliverables:
```powershell
python -m catalyst.pipeline assemble
```
*Output: `answers.json` and `answers_provenance.json`.*

### 6. Run Custom Evaluation Suite
Run the 12-question custom multi-meeting evaluation benchmark:
```powershell
python -m catalyst.pipeline eval --questions eval/eval_questions.json --max-steps 3 --k 8 --out eval/eval_report.json
```

### 7. Run Local Retrieval Contamination Test (0 LLM Calls)
Compare team-scoped retrieval versus un-scoped retrieval to measure cross-team contamination:
```powershell
python -m catalyst.pipeline eval --questions eval/eval_questions.json --no-team-test --k 8 --out eval/eval_report.json
```

### 8. Run Scalability and Cost Extrapolation Model
Print the verified cost and time extrapolation model:
```powershell
python -m catalyst.cost_estimate
```

---

## 8. Example Usage

### Sample Question (Q01)
> **Question:** *What did team TS3006 decide regarding the speech recognition feature versus the LCD screen, and what was the main reason for keeping the LCD screen?*

### Actual System Output (`answers.json`)
```json
{
  "question_id": "Q01",
  "answer": "Team TS3006 dropped speech recognition and kept the LCD screen. The reason for keeping the LCD screen was that losing it would necessitate a lot of extra buttons.",
  "citations": [
    {
      "meeting_id": "TS3006b",
      "quote": "And we do have to integrate the screen and the the speech alright."
    },
    {
      "meeting_id": "TS3006d",
      "quote": "Because if you d lose the L_C_D_ screen, we need a lot of Yeah. We need a lot of extra buttons."
    },
    {
      "meeting_id": "TS3006d",
      "quote": "Well yeah but we dropped the speech recognition."
    }
  ]
}
```

### Verbatim Citation Grounding
Every quote maps to an exact speaker turn in the raw transcript:
- `TS3006b`: Turn `#00120` — Speaker A explicitly includes both components during initial planning.
- `TS3006d`: Turn `#00101` — Speaker B warns that eliminating the LCD requires excessive buttons.
- `TS3006d`: Turn `#00422` — Speaker D confirms speech recognition was dropped.

---

## 9. Evaluation and Validation

Evaluation distinguishes between **deterministic schema/citation validity** and **independent factual correctness**.

### Evaluation Summary Table
| Metric / Check | Category | Result | Description / Notes |
|---|---|---|---|
| **Unit & Integration Tests** | Automated Code Quality | **94 / 94 Passed** | Offline pytest suite covering parser, DB, scoping, fallback, and validation. |
| **Q01–Q25 Schema Validity** | Deterministic Format | **25 / 25 Passed** | Strict Pydantic `extra="forbid"`, exactly 1–2 sentences, 1–3 citations per answer. |
| **Q01–Q25 Citation Veracity** | Deterministic Verbatim | **62 / 62 (100%) Verified** | Every quote confirmed against raw `.txt` files; 21/25 cite $\ge 2$ distinct meetings. |
| **Q01–Q25 Factual Correctness** | Independent Manual Audit | **13 Full / 10 Partial / 2 Incorrect** | Manual raw-transcript review; residual errors in Q03 and Q04 caused by free-tier fallback models. |
| **TS3012 Brief Schema & Temporal Scope** | Deterministic Validation | **Passed (22 Bullets)** | All 4 required sections present; 43/43 quotes verified; 0 references or leaks from `TS3012d`. |
| **Custom Evaluation (C01–C12)** | Automated Live Pipeline | **10 Validated / 2 Quota Exhausted** | 22/22 valid citations; mean gold retrieval recall: 0.6917 (completed); 2 provider quota failures. |
| **Custom Evaluation Human Rubric** | Manual Human Evaluation | **Pending Review** | `manual_rubric_score` left as `None` (`eval/human_review_sheet.md` unpopulated/unfabricated). |
| **Cross-Team Contamination** | Retrieval Benchmark | **94.79% Contamination** | Evaluated on 12 manual-gold questions with team filtering disabled at $k=8$. |

### Known Errors & Audit Details
1. **Q03 (`TS3009`):** The model correctly captured Speaker B proposing titanium-colored plastic (`TS3009d#00363`), but concluded titanium was retained. The raw transcript shows titanium was cut in `#00402` (`"Titanium gone and add plastic."`).
2. **Q04 (`TS3011`):** The model stated the team chose kinetic energy and batteries. The raw transcript confirms they chose traditional batteries and solar cells (`TS3011c#00286`), subsequently cutting solar cells in `TS3011d#00051`.
3. **C08 & C12 (Custom Eval):** Failed with `llm_error:AllProvidersFailed` when free-tier daily quotas (20 RPD on Gemini, 200K TPD on Groq) were exhausted. The pipeline recorded clean failures without fabricating answers.

---

## 10. Scalability and Cost Considerations

Measured operational baselines and linear extrapolations to 1,000 and 10,000 meetings (`catalyst/cost_estimate.py`):

### 1. Measured System Baselines
- **Corpus Density:** 138 meetings = 48,148 turns (172.62 citable `ge6`-word turns per meeting).
- **CPU Embedding Throughput:** FastEmbed (`bge-small-en-v1.5`) embeds at **2.34 centres/second** (10,150 s / 2.82 hours for 23,757 windows on local CPU).
- **TS3012 Ledger Extraction:** 22.0K tokens and 46.0 s per meeting (5.0 windows/meeting).
- **TS3012 Reconciliation:** 11.6K prompt tokens + 6.6K billed output tokens (including 3.7K Gemini reasoning tokens, accounting for 55.6% of billed output).
- **Per-Question QA Consumption:** Average of 3.44 calls, 12,118 prompt tokens, and 879 billed output tokens per question across the 25 verified answers.

### 2. Extrapolated Resource Projections
*Note: Figures for 1,000 and 10,000 meetings are mathematical linear extrapolations and are NOT verified at scale.*

| Corpus Scale | Est. Citable Vectors | Vector Storage (Float32) | CPU Embedding Time | Extrapolated Ingestion Cost (Gemini 3.8 Flash 2026) | Extrapolated Ingestion Cost (Gemini 2.5 Flash Assumed) |
|---|---|---|---|---|---|
| **138 Meetings** *(Verified)* | 23,821 | 34.9 MB | 2.82 Hours | \$5.08 *(Verified price)* | \$2.80 *(Assumption)* |
| **1,000 Meetings** *(Extrapolated)* | 172,616 | 252.9 MB | 20.49 Hours | \$36.80 | \$20.32 |
| **10,000 Meetings** *(Extrapolated)* | 1,726,159 | 2.53 GB | 204.91 Hours (8.54 Days) | \$367.90 | \$203.16 |

### 3. Scaling Bottlenecks & Architectural Migration
- **Embedding Bottleneck:** At 10,000 meetings, CPU embedding requires 8.5 days. A production deployment requires GPU-accelerated ONNX or asynchronous hosted batch embeddings.
- **Vector Database Migration:** ChromaDB in-process SQLite engine experiences concurrency limitations beyond ~100,000 vectors. A 10,000-meeting corpus should migrate to **Qdrant** or **PostgreSQL with `pgvector`**, using partitioned indexes by `team_id`.
- **Lexical Search Scaling:** SQLite FTS5 handles 3.5 million turns adequately, but production scale benefits from **Elasticsearch** or **PostgreSQL GIN / `tsvector`** with distributed indexing.

---

## 11. Known Limitations

- **Free-Tier Model Fallback Degrations:** High-volume batch runs exhaust free-tier rate limits (20 RPD on Gemini Flash, 200K TPD on Groq), forcing fallbacks to smaller models (`openai/gpt-oss-20b`, `gemini-3.5-flash-lite`, `qwen/qwen3.8-27b`). These models demonstrate lower reasoning fidelity on multi-speaker numerical costing discussions.
- **Contiguous Costing Window Limits:** In several meetings (e.g., `TS3005d`), budget negotiations span 50+ consecutive turns. Fixed-window retrieval seeds (`seq-5..seq+10`) occasionally truncate the final resolution turn, requiring wider sequence windowing.
- **Single-Meeting Citations on 4 Questions:** While 21 of 25 answers cite multiple meetings, 4 questions (`Q10, Q15, Q23, Q24`) cite evidence from a single meeting where the resolution was concentrated.
- **Unpopulated Custom Rubric Scores:** The 12 custom evaluation questions include complete deterministic fact-matching metrics, but human rubric scores remain unpopulated pending human evaluator scoring.

---

## 12. Security and Data Protection

- **API Key Security:** All API keys are loaded via `python-dotenv` from `.env`. No credentials or hardcoded keys exist in source code or documentation.
- **Git & Package Exclusion:** `.env`, build artifacts, and intermediate SQLite/Chroma databases are excluded via `.gitignore` and omitted from submission archives.
- **Header-Based Authentication:** Gemini requests utilize modern HTTP headers (`x-goog-api-key`) rather than query parameters, preventing accidental URL credential leakage.
- **Automatic Secret Redaction:** `catalyst/llm.py` enforces regex sanitization (`_redact_secrets()`) across all error messages and debug logs, stripping API keys and tokens before outputting stack traces.

---

## 13. Submission Contents

The submission contains all necessary components to inspect, verify, and run the project:
1. `answers.json` — Final answers for Q01–Q25 with verbatim citations, plus the `TS3012` pre-meeting brief.
2. `answers_provenance.json` — Detailed execution provenance, model names, token counts, and citation verification states.
3. `catalyst/` — Complete modular source code.
4. `tests/` — Full suite of 94 automated pytest unit and integration tests.
5. `eval/` — Custom evaluation suite, live reports, human review sheet, and cost models.
6. Documentation — `README.md`, `WRITEUP.md`, `ARCHITECTURE.md`, `DEMO_SCRIPT.md`, and `FINAL_AUDIT_REPORT.md`.

---

## 14. Future Improvements

1. **Dedicated Costing Ledger:** Implement deterministic regex and LLM extraction specialized for numerical budget tables and currency figures across final meetings.
2. **Dynamic Window Expansion:** Expand conversational context dynamically (`seq-10..seq+35`) whenever successive numerical currency mentions are detected.
3. **GPU Indexing Pipeline:** Migrate vector generation to GPU-accelerated batch embedding to process 10,000 meetings in hours rather than days.
4. **Paid Model Pinning:** Run production workloads against dedicated paid endpoints (`gemini-2.5-flash` or `gemini-3.8-flash`) to prevent rate-limit fallback degradation.

---

## 15. Conclusion

Catalist Media's Agentic AI Meeting Intelligence System demonstrates that reliable meeting intelligence requires code-enforced boundaries, strict retrieval scoping, and deterministic citation validation. By anchoring language model reasoning in exact character offsets, hybrid lexical/semantic retrieval, and temporal isolation, the system eliminates hallucinations and information leaks across complex multi-meeting lifecycles.
