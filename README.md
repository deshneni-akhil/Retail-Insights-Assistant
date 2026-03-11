# Retail Insights Assistant

**Multi-Agent GenAI System for Data Analytics**

Upload any data file and ask questions about it in plain English. The system figures out the SQL, runs it, validates the results, and gives you a clear answer with actual numbers. No SQL knowledge needed.

Built with a multi-agent pipeline — 13 specialized nodes coordinated through LangGraph, with DuckDB for queries, ChromaDB for RAG, and any OpenAI-compatible LLM.

---

## Demo

<video src="assets/demo-walkthrough.mp4" controls width="100%"></video>

---

## Features

- **Natural Language Q&A** — ask questions in English, get answers backed by real data
- **Executive Summaries** — one-click overview of your entire dataset
- **Multi-Format Upload** — CSV, Excel, JSON, PDF, TXT — all handled automatically
- **RAG-Enriched Answers** — upload business docs and the system uses them as context
- **Session Management** — named sessions with isolated data, persistent across restarts
- **Anti-Hallucination** — fact extraction layer ensures LLM only uses verified numbers
- **Semantic Caching** — repeated questions are answered instantly from cache
- **Smart Schema Selection** — handles large databases by picking relevant tables automatically

---

## 🚀 Quick Start

You'll need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/retail-insights-assistant.git
cd retail-insights-assistant
uv sync
```

Create a `.env` file (copy from `.env.example`):

```
LLM_BASE_URL=https://your-endpoint-url
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name
```

Run it:

```bash
uv run streamlit run src/app.py
```

Opens at http://localhost:8501. The `data/sessions/` directory gets created automatically on first run.

---

## 🔑 LLM Setup

This works with any OpenAI-compatible API endpoint. We've tested it with:

- **Azure AI Foundry** - I use Llama 4 scout model
- **OpenAI** — GPT-4o, GPT-4-mini
- **Groq** — fast inference
- **Ollama** — run locally, no API key needed
- **Together AI**, **Cerebras**, etc.

Just point `LLM_BASE_URL` at your provider and set the right API key. The system is provider-agnostic.

Optional — enable LangSmith tracing to see whats happening inside the pipeline:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

---

## 🏗️ Architecture

When a user asks a question, here's what happens:

```
User Question → Preprocess → Route → Generate SQL → Execute → Validate → Enrich → Respond
                    ↑                                              |
                    └──────── retry with failed SQL feedback ←─────┘
```

The pipeline has 13 nodes split across two planes:

**Control Plane** (intelligence):

| Agent | What it does |
|-------|-------------|
| Preprocess | Input sanitization + semantic Q&A cache lookup |
| Router | LLM classifies intent — question, summary, or clarification |
| NL-to-SQL | Converts English to DuckDB SQL using schema-aware prompts |
| Extraction | Runs SQL against DuckDB with safety checks (SELECT only) |
| Validation | Statistical pre-checks + LLM semantic validation |
| Fact Extractor | Pure Python — pulls verifiable numbers from results (no LLM) |
| Retrieval | RAG context from ChromaDB (past Q&A, docs, summaries) |
| Response | LLM generates answer constrained to verified facts only |
| Formatter | Cleans up markdown for display |
| Schema Selector | Picks relevant tables when database has many |
| Summary | Dynamic aggregation + executive summary generation |
| Fallback | Surfaces similar past Q&A when retries are exhausted |

**Data Plane** (storage):

| Component | Purpose |
|-----------|---------|
| DuckDB | SQL engine — file-backed, per-session, zero config |
| ChromaDB | Vector store — Q&A cache, document chunks, table schemas |
| Session Storage | Isolated DuckDB + ChromaDB per user session |

---

## 🔍 How a Query Works

Say you ask: *"Which category had the highest revenue?"*

| Step | Agent | What happens |
|------|-------|-------------|
| 1 | Router | Classifies as "question" → Q&A branch |
| 2 | NL-to-SQL | Generates: `SELECT category, SUM(amount) AS revenue FROM sales GROUP BY category ORDER BY revenue DESC LIMIT 5` |
| 3 | Extraction | Runs SQL on DuckDB → returns SET (28.7M), KURTA (22.8M), ... |
| 4 | Validation | Checks: non-empty, relevant columns, reasonable numbers → PASS |
| 5 | Fact Extractor | Extracts: category=SET, revenue=28,744,012 (verified facts) |
| 6 | Response | LLM writes: "SET had the highest revenue at ₹28.7M, followed by KURTA at ₹22.8M" |

If validation fails, the pipeline retries — feeding the failed SQL and error message back to the LLM so it can fix its mistake (up to 5 retries, then fallback to similar past Q&A).

---

## 🛠️ Tech Stack

- **LangGraph** — multi-agent orchestration with conditional routing and retry loops
- **DuckDB** — embedded SQL engine, zero setup, BigQuery-compatible syntax
- **ChromaDB** — vector store for RAG (4 collections per session)
- **Streamlit** — chat UI with session management
- **OpenAI SDK** — LLM calls (works with any compatible endpoint)
- **Pandas + openpyxl** — Excel/CSV processing
- **PyMuPDF** — PDF text extraction
- **LangSmith** — optional tracing and observability

---

## 🧪 Tests

```bash
uv run pytest tests/ -v
```

136 tests covering: database operations, SQL extraction safety, session management, file uploads, statistical validation, vectorstore operations, fact extraction, input preprocessing, and schema selection.

All tests run without API keys — LLM calls are mocked.

---

## ⚠️ Assumptions & Limitations

Things to keep in mind:

- **Single machine only** — DuckDB is an in-process engine, no distributed queries
- **No authentication** — anyone with access to the URL can use it. No multi-user concurrency controls
- **LLM quality varies** — works best with capable models (GPT-4o, Llama-4-Maverick). Smaller models may produce worse SQL
- **Memory-bound file loading** — Pandas loads entire files into memory, so very large files (1GB+) might be slow or fail
- **Static data only** — no live database connections, you upload files manually
- **No streaming** — responses come back all at once, not token-by-token

---

## 🔮 Possible Improvements

Where this could go next:

- **Scale storage** — swap DuckDB → BigQuery/Snowflake for 100GB+ datasets (SQL stays the same since DuckDB syntax matches BigQuery)
- **Query caching** — add Redis layer, expect 60-80% cache hit rate for repeat questions
- **Model routing** — use cheaper/faster models for simple lookups, reserve expensive models for complex multi-table reasoning
- **Materialized views** — pre-compute common aggregations to cut query time from seconds to milliseconds
- **Auth + multi-user** — add proper authentication and concurrent session handling
- **Tool-use / MCP** — let the LLM decide which tools to call instead of a fixed pipeline
- **Streaming responses** — token-by-token output for better UX

---
