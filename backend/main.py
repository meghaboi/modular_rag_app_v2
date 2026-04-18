from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.azure_rag import AzureRagError, AzureRagService
from backend.config import Settings, SettingsError
from backend.documents import DocumentProcessingError, extract_text_from_upload
from backend.models import ChatRequest, ChatResponse, UploadResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    session_id: str
    file_name: str
    chunk_count: int


app = FastAPI(
    title="CA-RAG Backend",
    version="1.0.0",
    description="Azure-only RAG backend for the CA-RAG Next.js frontend.",
)

_service: AzureRagService | None = None
_settings: Settings | None = None
_sessions: dict[str, SessionRecord] = {}


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def get_service() -> AzureRagService:
    global _service
    if _service is None:
        _service = AzureRagService(get_settings())
    return _service


def _load_cors_origins() -> list[str]:
    try:
        return [get_settings().frontend_origin]
    except SettingsError:
        return ["http://localhost:3000"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        settings = get_settings()
    except SettingsError as exc:
        return {"status": "misconfigured", "detail": str(exc)}

    return {
        "status": "ok",
        "provider": "azure-ai-foundry",
        "search_index": settings.azure_search_index_name,
    }


@app.post("/api/sessions", response_model=UploadResponse)
async def create_session(file: UploadFile = File(...)) -> UploadResponse:
    try:
        text, file_name = await extract_text_from_upload(file)
        session_id = str(uuid4())
        chunk_count = get_service().ingest_document(session_id, file_name, text)
    except SettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (DocumentProcessingError, AzureRagError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload failure")
        raise HTTPException(status_code=500, detail="Upload failed.") from exc

    _sessions[session_id] = SessionRecord(
        session_id=session_id,
        file_name=file_name,
        chunk_count=chunk_count,
    )
    return UploadResponse(
        session_id=session_id,
        file_name=file_name,
        chunk_count=chunk_count,
        message="File indexed and ready for chat.",
    )


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return asdict(session)


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest) -> ChatResponse:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        answer, contexts, usage = get_service().answer_question(
            session_id=session_id,
            question=payload.question,
            history=payload.history,
        )
    except SettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AzureRagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat failure")
        raise HTTPException(status_code=500, detail="Chat failed.") from exc

    return ChatResponse(answer=answer, contexts=contexts, usage=usage)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        get_service().delete_session(session_id)
    except SettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected session deletion failure")
        raise HTTPException(status_code=500, detail="Session cleanup failed.") from exc

    _sessions.pop(session_id, None)
    return {"status": "deleted"}
