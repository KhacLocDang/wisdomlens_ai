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

- **Backend:** FastAPI (Python) + multiple LLM providers (Gemini by default, Claude as second provider)
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

Open `.env` and set your provider credentials:

```env
GEMINI_API_KEY=your-real-gemini-key-here
CLAUDE_API_KEY=your-real-claude-key-here
AI_PROVIDER=gemini
```

> **Note:** Use file `.env` (not `.env.example`). Docker reads `.env` at the project root. Do not commit `.env` to Git.

Optional variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FAKE_ANSWERS` | `false` | Set to `true` to skip model generation and return static placeholder answers |
| `AI_PROVIDER` | `gemini` | Default provider for generation (`gemini` or `claude`) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Default Gemini model when UI does not send `model` |
| `CLAUDE_MODEL` | `claude-3-5-haiku-20241022` | Default Claude model when using the Claude provider |
| `USE_RAG` | `false` | Set to `true` to make `/ask` use retrieval context by default |
| `RAG_MIN_SCORE` | `0.35` | Minimum cosine similarity score for a retrieved chunk to be used as context |
| `POSTGRES_USER` | `wisdomlens` | PostgreSQL username (local dev) |
| `POSTGRES_PASSWORD` | `wisdomlens` | PostgreSQL password (local dev — change for production) |
| `POSTGRES_DB` | `wisdomlens` | PostgreSQL database name |
| `DATABASE_URL` | `postgresql+psycopg2://...` | Full SQLAlchemy DB connection string specifying `psycopg2` driver dialect |

PostgreSQL credentials are read from `.env` by Docker Compose. `DATABASE_URL` for the backend uses `postgresql+psycopg2://` scheme to ensure compatibility with `psycopg2-binary`. Each successful `/ask` request is saved to the `inquiries` table.

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

List available models for the default provider (Gemini by default):

```powershell
curl "http://localhost:8000/models"
```

List available models for Claude explicitly:

```powershell
curl "http://localhost:8000/models?provider=claude"
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

- **Hỏi (Ask)** — ask a new question, choose language, provider (Gemini or Claude), model, and whether to use RAG for that request
- **Lịch sử (History)** — browse saved questions from PostgreSQL, including stored RAG source metadata
- **Semantic Retrieval** — search stored chunks directly
- **Tài liệu (Documents)** — inspect uploaded documents and their chunks

## Feature: Select Multiple Perspectives

- **Backend:** Added a `perspectives` JSONB column to the `inquiries` table to store answers for any number of perspectives. The column is populated from the request's `perspectives` list.
- **API:** The `/ask` endpoint now accepts an optional `"perspectives": ["buddhism", "psychology", ...]` field. The backend builds a dynamic prompt that includes each selected perspective and returns a dictionary of answers.
- **Frontend:** Introduced a multiselect UI (`st.multiselect`) defined by the `PERSPECTIVES` constant, allowing users to pick any combination of available perspectives.
- **Migration:** Alembic migration creates the `perspectives` column and keeps the older perspective columns nullable for backward compatibility.
- **Schema:** `GeminiWisdomFields` now includes a `perspectives: dict[str, str]` field, and responses store each perspective's answer in this dict.

## Feature: Multiple LLM Providers (Gemini & Claude)

WisdomLens AI allows flexible switching between AI providers:
- **Supported Providers:** Google Gemini (`gemini`) and Anthropic Claude (`claude`).
- **Default behavior:** Gemini remains the default provider for backward compatibility.
- **Configuration (`.env`):** Set `CLAUDE_API_KEY=sk-ant-...`, optionally set `AI_PROVIDER=claude`, and keep `GEMINI_API_KEY` for Gemini usage.
- **Streamlit UI:** Select provider from the "Nhà cung cấp AI" dropdown in the Ask tab; model list updates automatically.
- **API `/ask`:** Pass `"provider": "claude"` and `"model": "claude-3-5-haiku-20241022"` in the JSON payload.
- **Extensibility:** The provider layer is designed so future providers such as GPT-compatible APIs or open-source local models can be added with the same contract.

## Feature: Text-to-Speech (TTS)

WisdomLens AI includes native Text-to-Speech (TTS) using the Google Gemini API:
- **Audio Output:** Converts synthesized wisdom answers into expressive speech on demand.
- **Prebuilt Voices:** Choose from 5 voices (`Aoede`, `Kore`, `Puck`, `Charon`, `Fenrir`).
- **Configuration (`.env`):**
  - `GEMINI_TTS_MODEL`: default TTS model (e.g. `gemini-3.8-flash-tts`).
  - `GEMINI_TTS_VOICE`: default voice (e.g. `Aoede`).
- **Streamlit UI:** Click **"🔊 Đọc câu trả lời (Read Aloud)"** in the answer panel to generate and listen to the audio directly in your browser.

### Saved Audio for Answers

Audio can also be saved with an inquiry from the History answer view:

- **Create and replay:** Generate audio on demand, then play the saved file later from that inquiry.
- **Storage:** Audio files are stored outside PostgreSQL at `data/audio/inquiry_<id>.wav`. The inquiry row stores the filename, MIME type, voice, TTS model, and creation time.
- **Docker persistence:** Compose mounts `./data/audio` on the host at `/data/audio` in the backend container, so container recreation does not discard the audio files.
- **Delete:** The answer's delete control removes the application file and clears its metadata from PostgreSQL. It does not delete copies already backed up to OneDrive.
- **Scope:** Saved-audio controls are available for inquiries opened from History. The immediate `/ask` response does not include its saved inquiry ID yet.

The audio API is available at `POST /inquiries/{inquiry_id}/audio` to synthesize and save, `GET` on the same path to stream the saved file, and `DELETE` to remove the application copy.

## Research Agent (Experimental)

A standalone exploratory tool (`analysis/research_agent.py`) that reads saved inquiries from PostgreSQL (read-only) and uses Gemini to identify recurring life themes, compare perspective patterns, and generate hypotheses for further research. Reports are saved to `analysis/research_report.md`.

Quick commands:

```powershell
# Verify DB connection, schema, prompt size, and API key configuration without calling Gemini:
python analysis/research_agent.py --check-only

# Run full analysis and generate report:
python analysis/research_agent.py

# Preview sample prepared records:
python analysis/research_agent.py --preview --preview-records 3

# Customize analysis:
python analysis/research_agent.py --limit 100 --model gemini-2.5-flash
```

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

### Backup (database and audio → local + OneDrive)

```powershell
.\scripts\backup.ps1
```

Saves a `.sql` file to `backups/` and copies it to `%USERPROFILE%\OneDrive\WisdomLens_Backups\`. It also copies audio files referenced by the current database into `%USERPROFILE%\OneDrive\WisdomLens_Backups\audio\`.

- Local backups: kept for **14 days**
- OneDrive SQL backups: kept for **30 days**
- OneDrive audio: stored in one fixed folder and not automatically deleted or age-pruned. Files with the same name are updated on the next backup; old audio copies are left for manual cleanup.

The script reads distinct `audio_filename` values from `inquiries` and copies only those files from `data/audio`. If a referenced file is missing locally, the script stops and reports the missing file.

To schedule daily backups automatically, add the script to Windows Task Scheduler:

- **Action:** `powershell.exe -ExecutionPolicy Bypass -File "<project-root>\scripts\backup.ps1"`
- **Trigger:** Daily at your preferred time

Replace `<project-root>` with where you cloned this repo (e.g. `%USERPROFILE%\dev\wisdomlens_ai`).

### Restore

```powershell
.\scripts\restore.ps1 -BackupFile ".\backups\wisdomlens_2026-06-17_0800.sql"
```

## Current limitations

- Gemini remains the default provider; Claude is available as a second supported option
- Static placeholder answers are still used when `USE_FAKE_ANSWERS=true`
- RAG is optional and only uses stored chunks that pass the similarity threshold
- History shows all saved questions, with no login or per-user filtering yet
- No authentication or user accounts
- Source citations are stored as metadata for future UI work, but not rendered inline yet

## Next steps

1. Render source citations inline in the UI
2. Improve retrieval ranking and chunk metadata
3. Add authentication and per-user history
4. Add support for GPT-compatible providers and open-source local models via the same provider abstraction
5. Add local model support via Ollama (e.g. Llama, Gemma)

## License & Usage Notice

WisdomLens AI is a personal project created for learning, research, and experimentation.

The source code is publicly available for reference. You may download and run the project using your own API keys for personal, educational, research, experimental, or legitimate business purposes.

Please use this project responsibly and do not use it for illegal, harmful, malicious, abusive, or unethical purposes.

This repository may contain third-party libraries, resources, documentation, or other materials that are subject to their respective licenses and terms. Their rights remain with their respective owners.

Unless otherwise stated, no general license is granted to redistribute, relicense, or commercially distribute the source code itself without permission from the author.

