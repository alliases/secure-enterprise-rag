# File: app/api/endpoints/ingest.py
# Purpose: Endpoints for document uploading and ingestion status tracking.

import shutil
import uuid
from pathlib import Path
from typing import Any

import magic
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import require_role
from app.db.models import Document
from app.dependencies import get_current_user, get_db_session, get_qdrant, get_redis
from app.ingestion.pipeline import run_ingestion
from app.logging_config.setup import get_logger

logger = get_logger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


MAX_FILE_SIZE_MB = 50
ALLOWED_MIMES = {
    "application/pdf": ["pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ["docx"],
    "application/msword": ["doc"],
    "text/plain": ["txt", "md", "csv"],
    "text/markdown": ["md"],
    "text/csv": ["csv"],
}


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    department_id: str = Form(...),
    access_level: int = Form(...),
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(require_role(["hr_manager", "admin"])),
    db: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> dict[str, Any]:
    """
    Accepts a document file, validates magic bytes and size, saves it temporarily,
    and starts the ingestion background task.
    """
    # 1. Size Validation (Protect against memory exhaustion)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    # 2. Magic Bytes Validation (Protect against CWE-434 arbitrary file upload)
    mime_type = magic.from_buffer(content[:2048], mime=True)

    # Fallback allowance for generic text variants detected by magic (e.g. text/x-script)
    if mime_type not in ALLOWED_MIMES and not mime_type.startswith("text/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unsupported file content type: {mime_type}",
        )

    await file.seek(0)  # Reset file pointer for standard save logic

    # 3. Extension Alignment Validation
    file_ext = Path(file.filename or "").suffix.lstrip(".").lower()
    allowed_exts = [ext for exts in ALLOWED_MIMES.values() for ext in exts]
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(set(allowed_exts))}",
        )

    document_id = uuid.uuid4()

    logger.info(
        "Document upload validated and ingestion scheduled",
        document_id=str(document_id),
        filename=file.filename,
        file_size_bytes=len(content),
        mime_type=mime_type,
        department_id=department_id,
        user_id=current_user["user_id"],
    )
    temp_file_path = UPLOAD_DIR / f"{document_id}.{file_ext}"

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("Failed to save uploaded file", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file locally.",
        ) from e

    # Create DB record with pending status
    new_document = Document(
        id=document_id,
        filename=file.filename or "unknown",
        department_id=department_id,
        access_level=access_level,
        status="pending",
        uploaded_by=uuid.UUID(current_user["user_id"]),
    )
    db.add(new_document)
    await db.commit()

    # Pass session_factory to the background task to ensure a fresh DB session
    session_factory = request.app.state.session_factory

    background_tasks.add_task(
        run_ingestion,
        file_path=temp_file_path,
        file_name=file.filename or "unknown",
        file_type=file_ext,
        document_id=str(document_id),
        department_id=department_id,
        access_level=access_level,
        user_id=current_user["user_id"],
        redis=redis,
        qdrant=qdrant,
        session_factory=session_factory,
    )

    return {
        "document_id": str(document_id),
        "status": "pending",
        "message": "Document ingestion started in the background",
    }


@router.get("/{document_id}/status")
async def get_ingestion_status(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Returns the current ingestion status of a specific document.
    """
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    # Enforce RBAC: users can only check status of documents from their department or if admin
    if (
        current_user["role"] != "admin"
        and doc.department_id != current_user["department_id"]
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this document's status",
        )

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
    }
