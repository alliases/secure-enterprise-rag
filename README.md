# 🔐 Secure Enterprise RAG

> A production-grade Retrieval-Augmented Generation (RAG) system with real-time PII masking, role-based access control, and full audit logging — purpose-built for HR and enterprise document intelligence.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-74%25-yellow)](https://github.com/alliases/secure-enterprise-rag)

---

## 📌 What It Does

Secure Enterprise RAG lets authorized users query confidential HR documents in natural language. Before any text reaches an external LLM or vector database, **all PII is masked by Microsoft Presidio**. Responses are de-masked on the fly — but only for users whose role permits it. Every action is written to a tamper-evident audit log.

**Proven behavior (manual E2E tests):**

| Role | Query | Response |
|---|---|---|
| `viewer` | "What is the salary of John Doe and his employee ID?" | `$150,000` and `[EMPLOYEE_ID_1]` ← masked |
| `hr_manager` | Same query | `$150,000` and `4500-1234` ← de-masked |

---

## ✨ Features

- **PII Masking Pipeline** — Microsoft Presidio detects and replaces names, emails, phone numbers, credit cards, and custom employee IDs (`DDDD-DDDD` format) with incremental tokens (`[PERSON_1]`, `[EMPLOYEE_ID_1]`, etc.) before any data leaves the system
- **LangGraph Orchestration** — a typed, stateful graph with conditional edges: `query_analyzer → retriever → synthesizer → validator → demasker`
- **RBAC De-masking** — `hr_manager` users see real PII in responses; `viewer` users see tokens; `admin` users see everything
- **Vector Retrieval with Metadata Filtering** — Qdrant filters by `department_id` and `access_level` to prevent cross-department data leakage
- **Structured Audit Logging** — every `login`, `ingest`, `query`, and `demask` event is persisted to PostgreSQL; logs are PII-sanitized via a custom `structlog` processor
- **Async-First Architecture** — FastAPI + SQLAlchemy asyncio + asyncpg + aioredis throughout
- **Retry & Graceful Degradation** — `tenacity` exponential backoff on LLM and embedding calls; Redis failures return masked text with a warning rather than a 500
- **Docker-first Deployment** — four-service `docker-compose` stack (FastAPI, Qdrant, Redis, PostgreSQL) with a non-root user, multi-stage Dockerfile, and AOF-persisted Redis

---

## 🏗 Architecture

```
User Request (JWT)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (Auth → RBAC → Endpoint)                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  LangGraph RAG Graph   │
          │                        │
          │  query_analyzer_node   │  ← Presidio masks query PII
          │         │              │
          │  retriever_node        │  ← Qdrant (dept + access filter)
          │         │              │
          │  synthesizer_node      │  ← OpenAI GPT-4o (masked context only)
          │         │              │
          │  validator_node        │  ← Heuristic prompt-injection guard
          │         │              │
          │  demasking_node        │  ← Redis lookup, RBAC check
          └────────────────────────┘
                       │
                       ▼
             Final Response (masked or de-masked)
                       │
                       ▼
               AuditLog → PostgreSQL
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Vector DB | [Qdrant](https://qdrant.tech/) (self-hosted) |
| PII Engine | [Microsoft Presidio](https://microsoft.github.io/presidio/) + Custom Regex |
| Embeddings | OpenAI `text-embedding-3-small` / `sentence-transformers` (local) |
| LLM | OpenAI GPT-4o |
| Mapping Store | [Redis](https://redis.io/) (AOF persistence, 30-day TTL) |
| Database | PostgreSQL 16 + SQLAlchemy asyncio + Alembic |
| Auth | JWT (python-jose) + Argon2id password hashing |
| Logging | [structlog](https://www.structlog.org/) (JSON, PII-sanitized) |
| Infrastructure | Docker + Docker Compose |
| Parsing | PyMuPDF (`fitz`), python-docx |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## 📦 Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- An OpenAI API key

### 1. Clone & configure

```bash
git clone https://github.com/alliases/secure-enterprise-rag.git
cd secure-enterprise-rag
cp .env.example .env
# Edit .env — set OPENAI_API_KEY, JWT_SECRET, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
```

### 2. Start all services

```bash
docker-compose up -d --build
```

All four services start on their default ports:

| Service | Port |
|---|---|
| FastAPI app | `8000` |
| Qdrant | `6333` |
| Redis | `6379` |
| PostgreSQL | `5432` |

### 3. Apply database migrations

```bash
docker-compose exec app alembic upgrade head
```

### 4. Seed test users

```bash
docker-compose exec app python seed.py
```

This creates two users:

| Email | Password | Role |
|---|---|---|
| `hr@example.com` | `12345` | `hr_manager` |
| `viewer@example.com` | `12345` | `viewer` |

### 5. Verify

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "version": "0.1.0"}
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🚀 Usage

### Authenticate

```bash
curl -X POST http://localhost:8000/auth/login \
  -F "username=hr@example.com" \
  -F "password=12345"
# → {"access_token": "<JWT>", "token_type": "bearer"}
```

### Upload a document

```bash
curl -X POST http://localhost:8000/ingest/ \
  -H "Authorization: Bearer <JWT>" \
  -F "department_id=hr_dept" \
  -F "access_level=1" \
  -F "file=@/path/to/report.pdf"
# → {"document_id": "<UUID>", "status": "pending", ...}
```

### Check ingestion status

```bash
curl http://localhost:8000/ingest/<document_id>/status \
  -H "Authorization: Bearer <JWT>"
# → {"status": "done", "chunk_count": 12, ...}
```

### Query

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer <JWT>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the salary of John Doe and what is his employee ID?"}'
```

**Response for `hr_manager`:**
```json
{
  "answer": "The salary for John Doe is $150,000, and his internal employee ID is 4500-1234.",
  "sources": ["<document_uuid>"]
}
```

**Response for `viewer`:**
```json
{
  "answer": "The salary for John Doe is $150,000, and his internal employee ID is [EMPLOYEE_ID_1].",
  "sources": ["<document_uuid>"]
}
```

---

## 🔐 Security Model

```
Document text
    │
    ├─ Presidio PII detection (score threshold: 0.4)
    │   Detected: PERSON, EMAIL_ADDRESS, PHONE_NUMBER,
    │             CREDIT_CARD, EMPLOYEE_ID (custom regex \b\d{4}-\d{4}\b)
    │
    ├─ Token substitution: "John Doe" → [PERSON_1]
    │                      "4500-1234" → [EMPLOYEE_ID_1]
    │
    ├─ Masked text → OpenAI Embeddings → Qdrant
    │
    └─ Mappings (token ↔ original) → Redis (TTL: 30 days)
                                       key: pii:{doc_id}:{token}

Query path:
    User query → Presidio mask → Qdrant (dept+level filter)
              → LLM (masked context only)
              → Validator (prompt-injection guard)
              → De-masker (Redis lookup, RBAC check)
              → Response
```

**RBAC rules:**

| Role | Same department | Other department |
|---|---|---|
| `admin` | De-masked | De-masked |
| `hr_manager` | De-masked | Masked |
| `viewer` | Masked | Masked |

---

## 🗄 Environment Variables

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_DSN` | Async PostgreSQL connection string | `postgresql+asyncpg://user:pass@postgres:5432/rag_db` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379` |
| `QDRANT_HOST` | Qdrant hostname | `qdrant` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `OPENAI_API_KEY` | OpenAI API key (SecretStr) | `sk-...` |
| `JWT_SECRET` | JWT signing secret (SecretStr) | `change-me-in-production` |
| `JWT_ALGORITHM` | JWT algorithm | `HS256` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `LLM_MODEL` | LLM model name | `gpt-4o` |
| `CHUNK_SIZE` | Text chunk size (chars) | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap (chars) | `200` |
| `LOG_LEVEL` | Logging level | `INFO` |

---

## 🧪 Running Tests

```bash
# Install dev dependencies
poetry install

# Run all tests with coverage
poetry run pytest --cov=app --cov-report=term-missing
```

Current coverage: **74%** (780 statements, 206 missed).

Coverage by module:

| Module | Coverage |
|---|---|
| `app/masking/` | 92–100% |
| `app/graph/` | 93–100% |
| `app/auth/` | 69–100% |
| `app/ingestion/parser.py` | 83% |
| `app/api/endpoints/query.py` | 88% |
| `app/ingestion/pipeline.py` | 33% ⚠️ |
| `app/vectorstore/qdrant_client.py` | 34% ⚠️ |
| `app/dependencies.py` | 38% ⚠️ |

---

## 📁 Project Structure

```
secure-enterprise-rag/
├── app/
│   ├── api/endpoints/      # auth, ingest, query, health
│   ├── auth/               # JWT, Argon2id, RBAC
│   ├── graph/              # LangGraph state, nodes, builder
│   ├── ingestion/          # parser (PDF/DOCX), chunker, pipeline
│   ├── llm/                # provider (OpenAI), prompts
│   ├── masking/            # Presidio engine, Redis store, de-masker
│   ├── vectorstore/        # Qdrant client, embedder, retriever
│   ├── db/                 # SQLAlchemy models, session, audit log
│   ├── logging_config/     # structlog JSON + PII sanitizer
│   ├── config.py           # Pydantic BaseSettings
│   ├── dependencies.py     # FastAPI Depends factories
│   └── main.py             # lifespan, app factory
├── alembic/                # DB migrations
├── tests/                  # unit, integration, e2e
├── docker-compose.yml
├── Dockerfile
├── seed.py
└── .env.example
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes — the hook chain runs `ruff` (lint + format) and `pyright` on every commit
5. Open a pull request against `main`

Pre-commit hooks enforce: trailing whitespace, YAML/TOML validity, no large files (>5 MB), no private key leaks, no debug statements, `ruff` lint/format, and `pyright` type checking.

---

## 📄 License

[MIT](LICENSE)
