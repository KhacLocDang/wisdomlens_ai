# WisdomLens AI

WisdomLens AI helps users explore life questions through multiple perspectives:

- Buddhism
- Western philosophy
- Psychology
- Christianity
- Eastern philosophy
- Natural sciences

This project provides structured information and perspectives so humans and AI can reason together. It is **not** a therapist or deep life advisor.

## Tech stack

- **Backend:** FastAPI (Python) + Google Gemini API
- **Database:** PostgreSQL 16
- **Frontend:** Streamlit (Python)
- **Run:** Docker Compose

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (Windows)

## Setup (one time)

From the project root (`wisdomlens_ai`):

```powershell
copy .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your-real-api-key-here
```

> **Note:** Use file `.env` (not `.env.example`). Docker reads `.env` at the project root. Do not commit `.env` to Git.

Optional variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FAKE_ANSWERS` | `false` | Set to `true` to skip Gemini and return static placeholder answers |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Default Gemini model when UI does not send `model` |
| `USE_RAG` | `false` | Set to `true` to make `/ask` use retrieval context by default |
| `RAG_MIN_SCORE` | `0.35` | Minimum cosine similarity score for a retrieved chunk to be used as context |
| `POSTGRES_USER` | `wisdomlens` | PostgreSQL username (local dev) |
| `POSTGRES_PASSWORD` | `wisdomlens` | PostgreSQL password (local dev — change for production) |
| `POSTGRES_DB` | `wisdomlens` | PostgreSQL database name |

PostgreSQL credentials are read from `.env` by Docker Compose. `DATABASE_URL` for the backend is built automatically. Each successful `/ask` request is saved to the `inquiries` table.

Schema is managed by **Alembic**. Migrations run automatically on startup (`alembic upgrade head`).

## Run with Docker

```powershell
docker compose up --build
```

- **Frontend:** http://localhost:8501
- **Backend:** http://localhost:8000
- **Health check:** http://localhost:8000/health
- **API docs:** http://localhost:8000/docs

Stop with `Ctrl+C`, or run in background: `docker compose up -d --build`

## Example API requests

Ask a question (Vietnamese + model):

```powershell
curl -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Vì sao con người sợ thất bại?\", \"language\": \"vi\", \"model\": \"gemini-2.5-flash\"}"
```

Ask a question (English):

```powershell
curl -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Why are humans afraid of failure?\", \"language\": \"en\"}"
```

`language` accepts `"vi"` (default) or `"en"`.  
`model` is optional — pick an id from `GET /models`, or omit to use `GEMINI_MODEL`.
`use_rag` is optional — set it to `true` to force RAG for one request, or omit it to use `USE_RAG`.

Ask with RAG enabled for one request:

```powershell
curl -X POST "http://localhost:8000/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"Why do humans suffer?\", \"language\": \"en\", \"use_rag\": true}"
```

List available Gemini models (filtered for text generation):

```powershell
curl "http://localhost:8000/models"
```

List saved questions:

```powershell
curl "http://localhost:8000/inquiries?limit=20"
```

Search saved questions:

```powershell
curl "http://localhost:8000/inquiries?q=thất bại&limit=20"
```

Get one saved question by id:

```powershell
curl "http://localhost:8000/inquiries/1"
```

## UI

The Streamlit app includes tabs for:

- **Hỏi (Ask)** — ask a new question, choose language, Gemini model, and whether to use RAG for that request
- **Lịch sử (History)** — browse saved questions from PostgreSQL, including stored RAG source metadata
- **Semantic Retrieval** — search stored chunks directly
- **Tài liệu (Documents)** — inspect uploaded documents and their chunks

## Feature: Select Multiple Perspectives

- **Backend:** Added a `perspectives` JSONB column to the `inquiries` table to store answers for any number of perspectives. The column is populated from the request's `perspectives` list.
- **API:** The `/ask` endpoint now accepts an optional `"perspectives": ["buddhism", "psychology", ...]` field. The backend builds a dynamic prompt that includes each selected perspective and returns a dictionary of answers.
- **Frontend:** Introduced a multiselect UI (`st.multiselect`) defined by the `PERSPECTIVES` constant, allowing users to pick any combination of available perspectives.
- **Migration:** Alembic migration creates the `perspectives` column and keeps the older perspective columns nullable for backward compatibility.
- **Schema:** `GeminiWisdomFields` now includes a `perspectives: dict[str, str]` field, and responses store each perspective's answer in this dict.

## Database migrations (Alembic)

Schema changes are managed with Alembic. Migrations run automatically on `docker compose up`.

To run migrations manually inside the backend container:

```powershell
docker compose exec backend alembic upgrade head
```

To create a new migration after changing a model:

```powershell
docker compose exec backend alembic revision --autogenerate -m "describe change"
docker compose exec backend alembic upgrade head
```

> **Important:** Never use `docker compose down -v` unless you intentionally want to erase all data. Use `docker compose down` (without `-v`) to stop safely.

## Backup and restore

### Backup (pg_dump → local + OneDrive)

```powershell
.\scripts\backup.ps1
```

Saves a `.sql` file to `backups/` and copies it to `%USERPROFILE%\OneDrive\WisdomLens_Backups\`.

- Local backups: kept for **14 days**
- OneDrive backups: kept for **30 days**

To schedule daily backups automatically, add the script to Windows Task Scheduler:

- **Action:** `powershell.exe -ExecutionPolicy Bypass -File "<project-root>\scripts\backup.ps1"`
- **Trigger:** Daily at your preferred time

Replace `<project-root>` with where you cloned this repo (e.g. `%USERPROFILE%\dev\wisdomlens_ai`).

### Restore

```powershell
.\scripts\restore.ps1 -BackupFile ".\backups\wisdomlens_2026-06-17_0800.sql"
```

## Current limitations

- Answers come from Gemini (or static placeholders if `USE_FAKE_ANSWERS=true`)
- RAG is optional and only uses stored chunks that pass the similarity threshold
- History shows all saved questions, with no login or per-user filtering yet
- No authentication or user accounts
- Source citations are stored as metadata for future UI work, but not rendered inline yet

## Next steps

1. Render source citations inline in the UI
2. Improve retrieval ranking and chunk metadata
3. Add authentication and per-user history
4. Add local model support via Ollama (e.g. Llama, Gemma)
