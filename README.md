# Homelab AI

[![CI Pipeline](https://github.com/ssilva9912/HL_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/ssilva9912/HL_AI/actions/workflows/ci.yml)

A modular, fully local Retrieval-Augmented Generation (RAG) platform built in Python.

Homelab AI indexes local documents, retrieves relevant context using hybrid search, reranks results with a cross-encoder, and generates grounded answers using locally hosted large language models through Ollama.

Designed around clean architecture, dependency injection, strict typing, and comprehensive testing, Homelab AI serves as a foundation for local AI assistants, knowledge management systems, and future homelab automation projects.

---

## Project Status

**Status:** Active Development

**Latest Milestone**

- End-to-End Local RAG Backend Complete

**Progress**

```
████████████████████░ 96%
```

**Next Milestone**

- Streamlit GUI

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
| Streamlit GUI | 🚧 |
| FastAPI API | 🚧 |
| Conversation Memory | 🚧 |
| Docker Support | 🚧 |

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
- 109 automated tests
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

- Streamlit GUI
- FastAPI REST API
- Persistent Vector Database
- PDF Support
- Markdown Support

---

## Version 1.2

- Conversation Memory
- Streaming Responses
- Docker Deployment
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