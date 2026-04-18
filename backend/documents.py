from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import UploadFile
from PyPDF2 import PdfReader


class DocumentProcessingError(RuntimeError):
    """Raised when an uploaded document cannot be processed."""


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    compact_lines = [line for line in lines if line.strip()]
    return "\n".join(compact_lines).strip()


async def extract_text_from_upload(upload: UploadFile) -> tuple[str, str]:
    file_name = upload.filename or "document.txt"
    extension = Path(file_name).suffix.lower() or ".txt"
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type '{extension}'. Upload a PDF or TXT file."
        )

    file_bytes = await upload.read()
    if not file_bytes:
        raise DocumentProcessingError("The uploaded file is empty.")

    if extension == ".pdf":
        return _extract_pdf_text(file_bytes, file_name), file_name

    try:
        decoded = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = file_bytes.decode("utf-8", errors="ignore")

    cleaned = _clean_text(decoded)
    if not cleaned:
        raise DocumentProcessingError("The uploaded text file did not contain readable text.")
    return cleaned, file_name


def _extract_pdf_text(file_bytes: bytes, file_name: str) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise DocumentProcessingError(f"Failed to read PDF '{file_name}'.") from exc

    cleaned = _clean_text("\n".join(text_parts))
    if not cleaned:
        raise DocumentProcessingError(
            f"PDF '{file_name}' did not yield extractable text."
        )
    return cleaned
