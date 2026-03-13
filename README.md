# ABIDE AI Agent Server

AI-powered Bible meditation agent server built with **FastAPI**, **LangGraph**, and **Google Gemini**.

This server powers the ABIDE app — it provides SSE-streaming meditation conversations, theology-aware RAG search, deep verse analysis, and daily compass recommendations.

## Architecture

```
main.py                    # FastAPI app factory (lifespan, CORS)
app/
├── config/settings.py     # Pydantic Settings (env vars)
├── models/schemas.py      # Request/response Pydantic models
├── routers/agent.py       # SSE streaming endpoints
├── services/
│   ├── database.py        # Async PostgreSQL (psycopg3 + pgvector)
│   ├── redis_store.py     # Redis session persistence
│   └── rag.py             # Embedding + theology vector search
└── agents/
    ├── state.py           # MeditationState TypedDict (LangGraph)
    ├── nodes.py           # Agent node functions (stubbed)
    └── graph.py           # StateGraph definition & execution
```

### Agent Nodes (Multi-Agent Pattern)

The meditation agent uses a **Supervisor → Worker** pattern via LangGraph:

| Node | Role |
|------|------|
| **Supervisor** | Analyzes state and routes to the next node |
| **Planner** | Builds question strategy using RAG theology context |
| **Counselor** | Converses with the user in a warm, Socratic tone |
| **Observer** | Scores meditation depth (0–100) |
| **Scribe** | Compiles conversation into a structured meditation note |
| **Confirm End** | Re-confirms end intent when depth is low |
| **Wrap Up** | Asks whether to generate a note before closing |

> **Note:** Node implementations are stubbed in the public repository. See function docstrings for detailed descriptions.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent/meditate/start` | Start meditation session (SSE) |
| `POST` | `/agent/meditate/chat` | Continue meditation (SSE) |
| `POST` | `/agent/ask` | Theology Q&A (SSE) |
| `POST` | `/agent/deep-lens` | Deep verse analysis (JSON) |
| `POST` | `/agent/compass` | Daily compass recommendation (JSON) |
| `GET`  | `/` | Health check |

All meditation/ask endpoints stream responses via **Server-Sent Events (SSE)**.

## Tech Stack

- **Python** ≥ 3.12
- **FastAPI** — async web framework
- **LangGraph** — multi-agent state machine
- **LangChain + Google Gemini** — LLM orchestration
- **PostgreSQL + pgvector** — Bible data + theology vector search
- **Redis** — session state persistence
- **psycopg3** — async PostgreSQL driver with connection pool

## Setup

### 1. Prerequisites

- Python 3.12+
- PostgreSQL 16+ with pgvector extension
- Redis 7+

### 2. Environment Variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `GEMINI_API_KEY` | Google Gemini API key |
| `AI_SERVER_PORT` | Server port (default: 8000) |
| `CORE_SERVER_URL` | Core API server URL |
| `LOG_LEVEL` | Logging level (default: INFO) |

### 3. Install & Run

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn main:app --reload --port 8000
```

### 4. Docker

```bash
docker build -t abide-ai .
docker run -p 8000:8000 --env-file .env abide-ai
```

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Test
pytest
```

## License

This project is provided for reference and educational purposes.
