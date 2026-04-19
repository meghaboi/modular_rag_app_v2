# CA-RAG

Azure-only RAG application with:

- `frontend/`: Next.js newsprint UI
- `backend/`: FastAPI API over the existing Python document-processing flow
- Azure AI Foundry for chat and embeddings
- Azure AI Search for hybrid keyword + vector retrieval

## Runtime Shape

- Main chat model: `Kimi-K2.5`
- Embeddings: `embed-v-4-0`
- Retrieval: Azure AI Search hybrid search
- Active product flow: `upload -> index -> chat`
- No active model switcher in the user-facing app
- No active evaluation surface in the user-facing app

## Auth

The backend now supports two auth modes:

1. Preferred: Microsoft Entra ID through your local Azure login (`az login`)
2. Optional fallback: API keys

That means you do not need a `.env` file for secrets if your signed-in identity has access to:

- the Azure AI Foundry resource / deployments
- the Azure AI Search service

You still need to tell the app which Foundry resource and Search service to talk to, either through shell environment variables or `.env`.

## Config

Copy `.env.example` to `.env` only if you want local file-based config. The main values are:

- `AZURE_FOUNDRY_RESOURCE_NAME` or `AZURE_FOUNDRY_BASE_URL`
- `AZURE_FOUNDRY_CHAT_DEPLOYMENT`
- `AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT`
- `AZURE_SEARCH_SERVICE_NAME` or `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX_NAME`

Optional only for key-based auth:

- `AZURE_FOUNDRY_API_KEY`
- `AZURE_SEARCH_API_KEY`

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

The branch includes:

- a production-clean `next build`
- Python compile checks for `backend`, `prompts`, `models`, and `utils`
- GitHub Actions in `.github/workflows/ca-rag.yml` for the `CA-RAG` branch

## Azure Deploy

Current public endpoints:

- Frontend: `https://caragv2fe36972.z30.web.core.windows.net/`
- Backend API: `https://caragv2api36972.azurewebsites.net/`

Repeatable deploy commands:

```powershell
.\scripts\deploy-backend.ps1
.\scripts\deploy-frontend.ps1
```

The frontend deploy script builds `frontend/` as a static export and uploads it to the Azure Storage `$web` container. The backend deploy script packages only the FastAPI runtime files needed by Azure App Service.

GitHub Actions:

- `.github/workflows/ca-rag.yml` now runs CI on pull requests and performs real Azure deploys on `push` to `CA-RAG` or manual dispatch.
- Backend deploys use Azure federated login plus an app-settings sync step, so App Service config stays in source control instead of drifting in the portal.
- Frontend deploys reuse the Azure login and pull the active storage account key at runtime, so there is no separate static-site deploy secret to rotate.
- Required repo secrets are:
  `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`,
  `AZURE_FOUNDRY_CHAT_API_KEY`, `AZURE_FOUNDRY_EMBEDDING_API_KEY`,
  `AZURE_FOUNDRY_CHAT_FALLBACK_API_KEY`, `AZURE_SEARCH_API_KEY`

## Notes

- The legacy Streamlit code still exists in the repo, but the intended app surface is the Next.js frontend plus FastAPI backend.
- If you want full zero-config runtime beyond auth, the next step is provisioning a dedicated v2 Foundry resource and Search service and then wiring their names into the app defaults.
