# CA-RAG

Azure-only RAG application with:

- `frontend/`: Next.js newsprint UI
- `backend/`: FastAPI API over the existing Python document-processing code
- Azure AI Foundry / Azure OpenAI for chat + embeddings
- Azure AI Search for hybrid keyword + vector retrieval

## What Changed

- Removed the active UI surface for model switching
- Removed the active UI surface for evaluation flows
- Replaced the Streamlit-first UX with a simple upload-then-chat workflow
- Kept the Python backend and reused the existing chunking/prompt utilities where useful

## Architecture

1. User uploads one PDF or TXT file from the Next.js app.
2. FastAPI extracts text, chunks it, creates embeddings through Azure, and indexes the chunks into Azure AI Search.
3. Chat requests run hybrid retrieval against the indexed session data.
4. The Azure chat deployment answers using only the retrieved context.

## Required Environment

Copy `.env.example` to `.env` and fill in:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`

Optional tuning:

- `AZURE_OPENAI_EMBEDDING_DIMENSIONS`
- `AZURE_SEARCH_INDEX_NAME`
- `FRONTEND_ORIGIN`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`
- `RAG_CHUNKING_STRATEGY`
- `AZURE_OPENAI_TEMPERATURE`

## Local Run

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the frontend:

```bash
npm run dev
```

If the backend is not on `http://127.0.0.1:8000`, set:

```bash
NEXT_PUBLIC_API_BASE_URL=http://your-backend-host:8000
```

## Verification

The current branch includes:

- a production-clean `next build`
- Python compile checks for `backend`, `prompts`, `models`, and `utils`
- GitHub Actions in `.github/workflows/ca-rag.yml` for the `CA-RAG` branch

## Notes

- The legacy Streamlit code is still present in the repo, but it is no longer the intended app surface.
- The new active flow is `upload -> index -> chat`.
