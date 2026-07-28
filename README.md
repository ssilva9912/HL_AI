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
