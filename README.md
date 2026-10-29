# Homelab AI

[![CI Pipeline](https://github.com/ssilva9912/HL_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/ssilva9912/HL_AI/actions/workflows/ci.yml)

A modular, fully local Retrieval-Augmented Generation (RAG) platform built in Python.

Homelab AI indexes local documents, retrieves relevant context using hybrid search, reranks results with a cross-encoder, and generates grounded answers using locally hosted large language models through Ollama.

Designed around clean architecture, dependency injection, strict typing, and comprehensive testing, Homelab AI serves as a foundation for local AI assistants, knowledge management systems, and future homelab automation projects.

---

## Project Status

**Status:** Active Development

**Latest Milestone**

- Durable document ingestion, persistent conversations, and streaming chat

**Progress**

```
████████████████████░ 98%
```

**Next Milestone**

- Network-share integration

---

# Features

| Feature | Status |
|---------|:------:|
| Local document indexing | ✅ |
| Semantic chunking | ✅ |
| Ollama embeddings | ✅ |
| In-memory vector store | ✅ |
| BM25 retrieval | ✅ |
| Dense retrieval | ✅ |
| Hybrid Retrieval (RRF) | ✅ |
| Cross-encoder reranking | ✅ |
| Prompt Builder | ✅ |
| Ollama Generator | ✅ |
| RAG Pipeline | ✅ |
| Indexing Service | ✅ |
| End-to-end CLI Demo | ✅ |
| Streamlit GUI | ✅ |
| FastAPI API | ✅ |
| Conversation Memory | ✅ |
| Hybrid document/general inference | ✅ |
| Streaming chat responses | ✅ |
| Docker Support | ✅ |
| Prometheus metrics, request tracing, structured logs | ✅ |
| Grafana dashboard | ✅ |
| Pipeline, embedding and retrieval caching | ✅ |
| Opt-in live web search with SSRF protections | ✅ |
| Corrective RAG (retrieval grading, refinement, web fallback) | ✅ |
| MCP server (`web_search`, `fetch_page`) | ✅ |

---

# Architecture

```text
                  Documents
                      │
                      ▼
               Directory Scanner
                      │
                      ▼
                   Parser
                      │
                      ▼
             Semantic Chunker
                      │
                      ▼
             Ollama Embedder
                      │
                      ▼
              In-Memory Vector Store
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     BM25 Retriever         Dense Retriever
          │                       │
          └───────────┬───────────┘
                      ▼
            Hybrid Retrieval (RRF)
                      │
                      ▼
       Cross-Encoder Reranker
                      │
                      ▼
             Prompt Builder
                      │
                      ▼
            Ollama Generator
                      │
                      ▼
               RAG Pipeline
                      │
                      ▼
               Grounded Answer
```

---

# Project Structure

```text
backend/
│
├── chunking/
├── config/
├── embeddings/
├── indexing/
├── ingestion/
├── interfaces/
├── llm/
├── logging/
├── parser/
├── rag/
├── retrieval/
├── storage/
└── demo.py

tests/
```

---

# Demo

Run the complete end-to-end pipeline.

```bash
uv run python -m backend.demo
```

Pipeline Flow

1. Index local documents
2. Semantic chunking
3. Generate embeddings
4. Hybrid retrieval
5. Cross-encoder reranking
6. Prompt construction
7. Local LLM generation
8. Return grounded answer with citations

---

# Installation

Clone the repository

```bash
git clone https://github.com/ssilva9912/homelab_ai.git

cd homelab_ai
```

Install dependencies

```bash
uv sync
```

---

# Ollama Setup

Install Ollama

https://ollama.com

Recommended models

```bash
ollama pull llama3.1:8b

ollama pull nomic-embed-text
```

Verify installation

```bash
ollama list
```

---

# Running

Run the demo

```bash
uv run python -m backend.demo
```

---

# Docker Deployment

The Docker stack runs PostgreSQL, the FastAPI backend, and the Streamlit
frontend. Ollama continues to run on the host so it can use the host GPU and
existing local models.

Create the local environment file:

```bash
cp .env.example .env
```

Replace `POSTGRES_PASSWORD` in `.env` with a URL-safe random password. Verify
that the required Ollama models are installed on the host:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Build and start the stack:

```bash
docker compose up --build -d
docker compose ps
```

Open the application at <http://127.0.0.1:8501>. The API is available at
<http://127.0.0.1:8000>, and PostgreSQL is bound to localhost only.

A Caddy reverse proxy is also included, listening on port 80 and bound to all
network interfaces so other devices on your LAN can reach the frontend at
`http://<host-ip>/`. Set `HOMELAB_PROXY_PORT` in `.env` to change its port.
The proxy has no authentication yet; only run it on a trusted network.

Useful operations:

```bash
docker compose logs -f api frontend
docker compose restart api frontend
docker compose down
```

Application documents, staged uploads, the embedded Qdrant index, model cache,
and PostgreSQL data use named Docker volumes. `docker compose down` preserves
them; `docker compose down -v` permanently removes them.

On Linux, `host.docker.internal` is mapped through `host-gateway`. If Ollama is
hosted elsewhere, set `HOMELAB_OLLAMA_URL_DOCKER` in `.env`.

---

# Web Search

Local language models only know what they were trained on. Web search lets an
answer use current information. It is **off by default and chosen per question**,
because the question is sent to the search engines behind SearXNG.

Start the search backend (a self-hosted [SearXNG](https://github.com/searxng/searxng),
no API key needed):

```bash
docker compose --profile web up -d
```

Then tick **Search the web** in the sidebar, or send `"web": true` to `/search` or
the conversation message endpoints. This only *allows* a search: with corrective RAG
(below) the web is used when your documents fall short. Citations show the page's
domain and path instead of a filename. If the search backend is down, the question is
still answered from your documents and a warning is logged; if no backend is
configured at all, the request fails with a clear 503.

## Corrective RAG

Retrieval can return passages that look related but do not answer the question.
Corrective RAG ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) checks the
retrieved passages before answering and acts on the result. Every question is graded
by the cross-encoder scores the reranker already produces, so grading adds no model
call:

| Verdict | Best passage score | What happens |
| --- | --- | --- |
| Correct | at or above `HOMELAB_CRAG_CORRECT_THRESHOLD` (2.0) | Answer from your documents. |
| Ambiguous | in between | Use your documents and, if web search is allowed, the web, ranked together. |
| Incorrect | at or below `HOMELAB_CRAG_INCORRECT_THRESHOLD` (-6.0) | Discard the documents and, if web search is allowed, answer from the web. |

Two more steps run on the path that is taken:

- **Knowledge refinement.** Each kept passage is cut to the sentences that bear on the
  question (`HOMELAB_CRAG_STRIP_THRESHOLD`), which removes distracting text and
  shortens the prompt.
- **Query rewriting.** Before a web search the model writes a standalone search query
  from the question and the conversation, so a follow-up like "when was the newest one
  released?" is searchable. This costs one short generation
  (`HOMELAB_CRAG_QUERY_REWRITE=false` skips it).

The question only leaves your machine when the documents did not cover it *and* you
ticked **Search the web**. The chat shows which verdict applied and the query used;
answers are labelled as from documents, the web, or both. The thresholds are
cross-encoder scores, so they are specific to the reranker model: re-tune them if you
change `CrossEncoderReranker.DEFAULT_MODEL_NAME`. `HOMELAB_CRAG_ENABLED=false`
restores plain retrieval, where a ticked web box always searches. Decisions are
counted in `homelab_crag_decisions_total{verdict,sources}`.

Safeguards:

- Only public `http`/`https` addresses are fetched. Hostnames are resolved by the
  app and refused if any address is private, loopback, link-local or otherwise
  non-public; the connection is then pinned to that validated address, and each
  redirect is re-checked.
- Page downloads are limited in time (`HOMELAB_WEB_FETCH_TIMEOUT`), size
  (`HOMELAB_WEB_MAX_PAGE_BYTES`) and content type (HTML and plain text only).
- Web text is marked as untrusted in the prompt so instructions hidden inside a page
  are not followed. This reduces, but cannot eliminate, prompt-injection risk with a
  small local model.

## MCP server

The same search and fetch tools are available to any MCP client (Claude Code,
Claude Desktop, ...) through a stdio server that exposes `web_search` and
`fetch_page`:

```bash
claude mcp add homelab-web -- uv run --directory /path/to/homelab_ai python -m backend.mcp_server
```

or, in a client's JSON configuration:

```json
{
  "mcpServers": {
    "homelab-web": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/homelab_ai", "python", "-m", "backend.mcp_server"]
    }
  }
}
```

It talks to the SearXNG published on `http://127.0.0.1:8888` unless
`HOMELAB_WEB_SEARCH_URL` says otherwise, and refuses private addresses just like the
app does. The app itself calls the search backend directly rather than through MCP,
since Ollama has no MCP client.

---

# Performance and Observability

## Metrics

The API exposes Prometheus metrics at `http://127.0.0.1:8000/metrics`
(disable with `HOMELAB_METRICS_ENABLED=false`). Notable series:

| Metric | What it tells you |
|--------|-------------------|
| `homelab_http_request_duration_seconds` | End-to-end latency per route, streaming included |
| `homelab_rag_stage_duration_seconds{stage}` | Where a question spends its time: `retrieval.bm25`, `retrieval.embed_query`, `retrieval.vector_search`, `retrieval.rerank`, `prompt_build`, `generation`, `generation.first_token`, `generation.total`, plus `corpus_load`, `pipeline_build` and `warmup.*` |
| `homelab_llm_tokens_per_second`, `homelab_llm_prompt_tokens`, `homelab_llm_model_load_seconds` | Model throughput, prompt size and cold starts, taken from Ollama's own timing data |
| `homelab_ollama_request_duration_seconds{operation}` | Time spent waiting on Ollama for `generate`, `stream` and `embed` |
| `homelab_llm_truncated_total` | Answers cut off at `HOMELAB_OLLAMA_NUM_PREDICT` tokens (default 1024) |
| `homelab_cache_events_total{cache,result}` | Hit and miss counts for the `query_embedding`, `retrieval` and `pipeline` caches |
| `homelab_rag_requests_total{endpoint,answer_mode,outcome}` | Answers by outcome (`ok`, `error`, `cancelled`) |

Answer responses also carry the per-request breakdown in `metadata.stages_ms`
(and in the final `complete` event of a streamed answer). Stage timings are leaf
timings, so they do not overlap, but they need not add up to the total.

Every response has an `X-Request-ID` header (a caller-supplied one is kept),
and the same id appears on each log line for that request. Set
`HOMELAB_LOG_FORMAT=json` for structured logs.

## Dashboards

Start Prometheus and Grafana alongside the stack:

```bash
docker compose --profile observability up -d
```

Grafana is at <http://127.0.0.1:3001> with the **Homelab AI** dashboard
preloaded; Prometheus is at <http://127.0.0.1:9090>. Both bind to localhost
only. Set `GRAFANA_ADMIN_PASSWORD` in `.env` before first start.

## Benchmarking

```bash
uv run python scripts/benchmark_rag.py --runs 12 --json results.json
```

The script asks a mix of new and repeated questions and reports cold, new and
repeat latencies with the stage breakdown for each request.

## What is cached

| Layer | Scope | Invalidated by |
|-------|-------|----------------|
| Cross-encoder model, Ollama client | Process lifetime | Restart |
| BM25 index and pipeline objects | One corpus snapshot | Ingesting or deleting a document |
| Retrieval results | `(question, top_k)` per corpus snapshot, TTL `HOMELAB_QUERY_CACHE_TTL_SECONDS` | Ingest/delete, TTL |
| Query embeddings | Question text | LRU eviction |
| Models in Ollama memory | `HOMELAB_OLLAMA_KEEP_ALIVE` (default 30 minutes) | Idle timeout |

Final model answers are intentionally not cached: they depend on conversation
history and sampling. At startup the API loads the corpus, reranker and both
Ollama models in the background (`HOMELAB_WARMUP_ON_STARTUP`).

---

# Development

Run tests

```bash
uv run pytest
```

Run the integration demo

### Windows PowerShell

```powershell
$env:RUN_OLLAMA_INTEGRATION="1"

uv run pytest tests/test_demo.py -v

Remove-Item Env:RUN_OLLAMA_INTEGRATION
```

Lint

```bash
uv run ruff check .
```

Formatting

```bash
uv run ruff format --check .
```

Type checking

```bash
uv run python -m mypy backend
```

---

# Current Quality

- Ruff formatting
- Ruff linting
- Strict mypy type checking
- 254 automated tests (253 passed, 1 skipped)
- Integration testing
- Dependency Injection
- Protocol-based interfaces
- Modular architecture

---

# Roadmap

## Version 1.0

- ✅ Local RAG Backend
- ✅ Hybrid Retrieval
- ✅ Cross-Encoder Reranking
- ✅ Prompt Builder
- ✅ Ollama Generator
- ✅ RAG Pipeline
- ✅ End-to-End Demo

---

## Version 1.1

- ✅ Streamlit GUI
- ✅ FastAPI REST API
- ✅ Persistent Vector Database
- ✅ PDF Support
- ✅ Markdown Support

---

## Version 1.2

- ✅ Conversation Memory
- ✅ Streaming Responses
- ✅ Docker Deployment
- Benchmark Suite
- Additional Embedding Providers
- Additional LLM Providers

---

# Technology Stack

- Python 3.12
- Ollama
- sentence-transformers
- rank-bm25
- httpx
- Ruff
- Mypy
- Pytest
- uv

---

# Design Goals

Homelab AI is built around several engineering principles.

- Local-first execution
- Modular architecture
- Strong typing
- Test-driven development
- Dependency injection
- Interface-driven design
- Easily swappable components

---

# License

This project is released under the MIT License.
