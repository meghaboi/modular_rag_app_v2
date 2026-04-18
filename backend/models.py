from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SourceChunk(BaseModel):
    file_name: str
    chunk_id: int
    content: str
    score: float | None = None


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    contexts: list[SourceChunk]
    usage: UsageStats
