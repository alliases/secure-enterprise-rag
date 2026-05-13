# File: app/api/endpoints/ingest.py
# Purpose: Endpoints for document uploading and ingestion status tracking.
import uuid
from pathlib import Path
from typing import Any

import aiofiles
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
from app.ingestion.deduplicator import (
    check_exact_duplicate,
    compute_file_hash_stream,
)
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
    # 1. Size Validation (Protect against memory exhaustion via metadata)
    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning("Upload rejected: File size exceeds limit", file_size=file.size)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max allowed size is {MAX_FILE_SIZE_MB}MB.",
        )

    # 2. Magic Bytes Validation (Protect against CWE-434 arbitrary file upload)
    header_chunk = await file.read(2048)
    mime_type = magic.from_buffer(header_chunk, mime=True)

    # Fallback allowance for generic text variants detected by magic (e.g. text/x-script)
    if mime_type not in ALLOWED_MIMES and not mime_type.startswith("text/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unsupported file content type: {mime_type}",
        )

    # 3. Extension Alignment Validation
    file_ext = Path(file.filename or "").suffix.lstrip(".").lower()
    allowed_exts = [ext for exts in ALLOWED_MIMES.values() for ext in exts]
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(set(allowed_exts))}",
        )

    # 4. Level 1 Deduplication: Exact SHA-256 Match via Async Streaming
    file_hash = await compute_file_hash_stream(file)

    # 4.1 Redis Distributed Lock to prevent Race Conditions on identical concurrent uploads
    lock_key = f"lock:ingest:{department_id}:{file_hash}"
    lock_acquired = await redis.set(lock_key, "locked", nx=True, ex=300)  # 5 min TTL

    if not lock_acquired:
        logger.warning(
            "Upload blocked by Redis lock (Race Condition prevented)",
            file_hash=file_hash,
            department_id=department_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A file with this exact content is currently being processed. Please wait.",
        )

    existing_id = await check_exact_duplicate(db, file_hash, department_id)

    if existing_id:
        await redis.delete(lock_key)  # Fast cleanup if it's already successfully in DB
        logger.info(
            "Exact document duplicate prevented at Level 1",
            file_hash=file_hash,
            existing_document_id=existing_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "duplicate_document",
                "existing_document_id": existing_id,
                "message": "An identical document already exists in this department.",
            },
        )

    document_id = uuid.uuid4()

    logger.info(
        "Document upload validated and ingestion scheduled",
        document_id=str(document_id),
        filename=file.filename,
        file_size_bytes=file.size or 0,
        mime_type=mime_type,
        department_id=department_id,
        user_id=current_user["user_id"],
    )
    temp_file_path = UPLOAD_DIR / f"{document_id}.{file_ext}"

    try:
        await file.seek(0)
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(65536):
                await buffer.write(chunk)
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
        file_hash=file_hash,
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
