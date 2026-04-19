from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.azure_rag import AzureRagError, AzureRagService
from backend.config import Settings, SettingsError
from backend.documents import DocumentProcessingError, extract_text_from_upload
from backend.models import (
    ChatRequest,
    ChatResponse,
    CreateFolderRequest,
    SessionResponse,
    WorkspaceDocument,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    document_id: str
    file_name: str
    folder_path: str
    file_path: str
    chunk_count: int


@dataclass
class SessionRecord:
    session_id: str
    workspace_name: str
    chunk_count: int = 0
    folders: set[str] = field(default_factory=set)
    documents: list[DocumentRecord] = field(default_factory=list)


app = FastAPI(
    title="CA-RAG Backend",
    version="1.0.0",
    description="Azure-only RAG backend for the CA-RAG Next.js frontend.",
)

_service: AzureRagService | None = None
_settings: Settings | None = None
_sessions: dict[str, SessionRecord] = {}
_workspace_state_path = (
    Path("/home/carag-workspaces.json")
    if Path("/home").exists()
    else Path(".runtime/carag-workspaces.json")
)


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


def _normalize_folder_path(folder_path: str) -> str:
    cleaned = folder_path.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return ""

    parts = [part.strip() for part in cleaned.split("/") if part.strip() and part not in {".", ".."}]
    if not parts:
        return ""
    return "/".join(parts)


def _compose_file_path(folder_path: str, file_name: str) -> str:
    return f"{folder_path}/{file_name}" if folder_path else file_name


def _get_session_or_404(session_id: str) -> SessionRecord:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _serialize_sessions() -> list[dict[str, Any]]:
    return [
        {
            "session_id": session.session_id,
            "workspace_name": session.workspace_name,
            "chunk_count": session.chunk_count,
            "folders": sorted(session.folders),
            "documents": [
                {
                    "document_id": document.document_id,
                    "file_name": document.file_name,
                    "folder_path": document.folder_path,
                    "file_path": document.file_path,
                    "chunk_count": document.chunk_count,
                }
                for document in session.documents
            ],
        }
        for session in _sessions.values()
    ]


def _persist_sessions() -> None:
    _workspace_state_path.parent.mkdir(parents=True, exist_ok=True)
    _workspace_state_path.write_text(
        json.dumps(_serialize_sessions(), indent=2),
        encoding="utf-8",
    )


def _load_sessions() -> None:
    if not _workspace_state_path.exists():
        return

    try:
        raw_sessions = json.loads(_workspace_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to load persisted workspace state from %s", _workspace_state_path)
        return

    for item in raw_sessions:
        documents = [
            DocumentRecord(
                document_id=document["document_id"],
                file_name=document["file_name"],
                folder_path=document.get("folder_path", ""),
                file_path=document["file_path"],
                chunk_count=document["chunk_count"],
            )
            for document in item.get("documents", [])
        ]
        session = SessionRecord(
            session_id=item["session_id"],
            workspace_name=item["workspace_name"],
            chunk_count=item.get("chunk_count", 0),
            folders=set(item.get("folders", [])),
            documents=documents,
        )
        _sessions[session.session_id] = session


def _build_session_response(
    session: SessionRecord,
    message: str | None = None,
    latest_file_name: str | None = None,
) -> SessionResponse:
    documents = [
        WorkspaceDocument(
            document_id=document.document_id,
            file_name=document.file_name,
            folder_path=document.folder_path,
            file_path=document.file_path,
            chunk_count=document.chunk_count,
        )
        for document in session.documents
    ]
    return SessionResponse(
        session_id=session.session_id,
        workspace_name=session.workspace_name,
        chunk_count=session.chunk_count,
        document_count=len(session.documents),
        folders=sorted(session.folders),
        documents=documents,
        message=message,
        file_name=latest_file_name,
    )


def _create_session_record(workspace_name: str | None) -> SessionRecord:
    session_id = str(uuid4())
    default_name = f"Workspace {session_id[:8]}"
    session = SessionRecord(
        session_id=session_id,
        workspace_name=(workspace_name or default_name).strip() or default_name,
    )
    _sessions[session_id] = session
    _persist_sessions()
    return session


def _attach_document_to_session(
    session: SessionRecord,
    *,
    file_name: str,
    text: str,
    folder_path: str,
) -> tuple[str, int]:
    normalized_folder_path = _normalize_folder_path(folder_path)
    if normalized_folder_path:
        session.folders.add(normalized_folder_path)

    document_id, chunk_count = get_service().ingest_document(
        session_id=session.session_id,
        file_name=file_name,
        text=text,
        folder_path=normalized_folder_path,
    )
    document = DocumentRecord(
        document_id=document_id,
        file_name=file_name,
        folder_path=normalized_folder_path,
        file_path=_compose_file_path(normalized_folder_path, file_name),
        chunk_count=chunk_count,
    )
    session.documents.append(document)
    session.chunk_count += chunk_count
    _persist_sessions()
    return document.file_name, document.chunk_count


app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_load_sessions()


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
        "chat_deployment": settings.foundry_chat_deployment,
        "embedding_deployment": settings.foundry_embedding_deployment,
    }


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    file: UploadFile | None = File(default=None),
    workspace_name: str | None = Form(default=None),
    folder_path: str = Form(default=""),
) -> SessionResponse:
    session: SessionRecord | None = None
    try:
        session = _create_session_record(workspace_name)

        if file is None:
            return _build_session_response(session, message="Workspace created.")

        text, file_name = await extract_text_from_upload(file)
        latest_file_name, _ = _attach_document_to_session(
            session,
            file_name=file_name,
            text=text,
            folder_path=folder_path,
        )
        return _build_session_response(
            session,
            message="Workspace created and file indexed.",
            latest_file_name=latest_file_name,
        )
    except SettingsError as exc:
        if session and not session.documents:
            _sessions.pop(session.session_id, None)
            _persist_sessions()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DocumentProcessingError as exc:
        if session and not session.documents:
            _sessions.pop(session.session_id, None)
            _persist_sessions()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AzureRagError as exc:
        if session and not session.documents:
            _sessions.pop(session.session_id, None)
            _persist_sessions()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if session and not session.documents:
            _sessions.pop(session.session_id, None)
            _persist_sessions()
        logger.exception("Unexpected workspace creation failure")
        raise HTTPException(status_code=500, detail="Workspace creation failed.") from exc


@app.post("/api/sessions/{session_id}/folders", response_model=SessionResponse)
def create_folder(session_id: str, payload: CreateFolderRequest) -> SessionResponse:
    session = _get_session_or_404(session_id)
    folder_path = _normalize_folder_path(payload.folder_path)
    if not folder_path:
        raise HTTPException(status_code=400, detail="Folder path cannot be empty.")

    session.folders.add(folder_path)
    _persist_sessions()
    return _build_session_response(session, message="Folder created.")


@app.post("/api/sessions/{session_id}/documents", response_model=SessionResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    folder_path: str = Form(default=""),
) -> SessionResponse:
    session = _get_session_or_404(session_id)

    try:
        text, file_name = await extract_text_from_upload(file)
        latest_file_name, _ = _attach_document_to_session(
            session,
            file_name=file_name,
            text=text,
            folder_path=folder_path,
        )
        return _build_session_response(
            session,
            message="File indexed and added to workspace.",
            latest_file_name=latest_file_name,
        )
    except SettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AzureRagError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected document upload failure")
        raise HTTPException(status_code=500, detail="Upload failed.") from exc


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = _get_session_or_404(session_id)
    return _build_session_response(session)


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest) -> ChatResponse:
    _get_session_or_404(session_id)

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
    _get_session_or_404(session_id)

    try:
        get_service().delete_session(session_id)
    except SettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected session deletion failure")
        raise HTTPException(status_code=500, detail="Session cleanup failed.") from exc

    _sessions.pop(session_id, None)
    _persist_sessions()
    return {"status": "deleted"}
