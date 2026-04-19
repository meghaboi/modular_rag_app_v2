from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SourceChunk(BaseModel):
    document_id: str
    file_name: str
    folder_path: str
    file_path: str
    chunk_id: int
    content: str
    score: float | None = None


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class WorkspaceDocument(BaseModel):
    document_id: str
    file_name: str
    folder_path: str
    file_path: str
    chunk_count: int


class SessionResponse(BaseModel):
    session_id: str
    workspace_name: str
    chunk_count: int
    document_count: int
    folders: list[str] = Field(default_factory=list)
    documents: list[WorkspaceDocument] = Field(default_factory=list)
    message: str | None = None
    file_name: str | None = None


class CreateFolderRequest(BaseModel):
    folder_path: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    contexts: list[SourceChunk]
    usage: UsageStats
