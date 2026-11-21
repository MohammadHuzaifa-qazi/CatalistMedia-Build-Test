# ARCHITECTURE — Catalist Agentic AI Meeting Memory

> Free / local stack. Retrieval-first, evidence-backed, scales conceptually to 10,000 meetings.

## 1. One-line summary

Parse meeting transcripts into turn-level records, index them in a hybrid (keyword + vector) memory store, then run a LangGraph agent that retrieves evidence across meetings, tracks how decisions change over time, verifies citations deterministically, and writes `answers.json` + the Team TS3012 brief.

## 2. High-level data flow

```
transcripts/*.txt
      |
      v
[1] PARSER (plain Python)
      |  header + "HH:MM:SS Speaker X" turns
      v
turns.jsonl  (meeting_id, team, letter, timestamp, speaker, text)
      |
      +--> [2] SQLite  (metadata + FTS5 keyword index)
      |
      +--> [3] Vector store (Chroma / LanceDB) with local embeddings
      |
      +--> [4] (optional) structured tables: decisions / actions / disagreements
      |
      v
[5] AGENT (LangGraph)
      tools: search_memory, get_meeting, decision_history, verify_quote
      loop: retrieve -> check -> retrieve more -> reason -> answer
      |
      v
[6] CITATION VALIDATOR (plain Python, deterministic)
      |
      v
answers.json  (25 answers + TS3012 brief)
```

## 3. Layers

### 3.1 Ingestion / Parser (plain code)
- Read each `transcripts/*.txt`.
- Parse the 6-line header: `Meeting`, `Team`, `Meeting number`, `Speakers`, `Times`, `Source`.
- Parse dialogue into **turns**: `timestamp`, `speaker` (A/B/C/D), `text`.
- Emit `turns.jsonl`; each record keeps the **exact original text** (needed for word-for-word citations).
- Preserve offsets / order so a quote can be traced back to its meeting + turn.
- Derive a **role map per team** from self-introductions (usually meeting `a`): e.g. B=Project Manager, C=Marketing, D=Industrial Designer, A=UI Designer. If `a` is missing, use the earliest available meeting.

### 3.2 Memory store
- **SQLite**:
  - `turns` table (meeting_id, team, letter, seq, timestamp, speaker, text).
  - **FTS5** virtual table for keyword / BM25 retrieval.
- **Vector store (Chroma or LanceDB)**:
  - One embedding per chunk (turn or small turn-window).
  - Metadata: meeting_id, team, letter, speaker, timestamp, chunk_id.
- **(Optional) structured memory tables**:
  - `decisions(topic, decision, status[proposed/current/superseded], meeting_id, meeting_seq, speaker, quote, supersedes_id)`
  - `actions(owner, task, status[open/done], meeting_id, quote)`
  - `disagreements(topic, who, position, status[open/resolved], meeting_id, quote)`

### 3.3 Retrieval
- **Hybrid**: SQLite FTS5 (keyword) + vector similarity (semantic), merged/ranked.
- Filters by `team` and optionally `meeting letter` when the question names a team.
- Return chunks with full metadata so citations can be built directly from source text.
- Retrieve across **multiple meetings**; questions require >1 meeting.

### 3.4 Embeddings
- Local `sentence-transformers` (BGE-small-en-v1.5 or all-MiniLM-L6-v2).
- Computed once and **cached to disk**; $0 API cost, no rate limits.

### 3.5 Agent (LangGraph)
Workflow:
1. Understand the question (team, topic, what is being asked).
2. Search memory (hybrid) filtered to the team.
3. Retrieve candidate evidence from multiple meetings.
4. Decide if more retrieval is needed (loop).
5. Compare evidence **chronologically** (a -> b -> c -> d).
6. Resolve historical vs current decision where evidence supports it (don't assume latest mention = final).
7. Draft a 1–2 sentence answer.
8. Attach 1–3 citations from retrieved turns.
9. Hand off to the deterministic citation validator.
10. Return structured output.

### 3.6 Citation verification (plain code — never LLM)
For every citation:
1. Load the transcript for the given `meeting_id`.
2. Normalize case and spacing.
3. Check the quoted words exist **within a single speaker turn**.
4. If not found: reject / re-select from retrieved evidence — **never fabricate or silently replace**.
5. Only validated citations go into `answers.json`.

## 4. Model-vs-code split (explicit)

| Task | Handled by | Why |
|---|---|---|
| File reading, parsing, IDs | Python | deterministic, no reasoning |
| Chunking + metadata | Python | deterministic |
| Exact quote verification | Python | must be exact, no hallucination |
| Counting, costs, JSON validation | Python | arithmetic / schema |
| Token + time measurement | Python | measurement |
| Understanding conversation | LLM | natural language |
| Indirect references | LLM | semantic reasoning |
| Identifying decisions / disagreements | LLM | interpretation |
| Supersession (old vs current) | LLM (+ evidence) | reasoning over history |
| Synthesizing answers / brief | LLM | language generation |

## 5. Scalability (10,000 meetings)

- **Stateless retrieval**: agent loads only retrieved chunks, never the full corpus.
- **Index, not context**: FTS5 + vector index scale with storage, not prompt size.
- **Cached embeddings**: ingest once, query many times.
- **Batched + cached LLM extraction**: structured decision extraction runs per meeting, cached; avoids re-calling on every query.
- **Incremental ingest**: new meetings append to SQLite + vector store; no full rebuild.
- **Production needs**: managed vector DB (Qdrant/pgvector), job queue for ingestion, embedding cache, retry/rate-limit handling, monitoring.

## 6. Stack (free)

| Layer | Choice |
|---|---|
| Language | Python 3.11/3.12 |
| Parser | custom regex |
| Keyword index | SQLite FTS5 |
| Vector store | Chroma (simple) / LanceDB (scale) |
| Embeddings | sentence-transformers (local) |
| LLM | Groq free tier (Llama/Qwen) + Gemini Flash fallback |
| Orchestration | LangGraph |
| Validation | Pydantic + Python |
| Demo | Streamlit / Colab |

## 7. Outputs

- `answers.json` — 25 answers (1–2 sentences, 1–3 verified citations each) + `brief` for Team TS3012.
- README + ~1-page technical write-up.
- Evaluation: 10+ custom questions with manual ground truth, score, 3 honest failures, 1,000-meeting cost/time estimate.
- 3–5 min demo.
