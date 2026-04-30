# File: app/api/endpoints/ingest.py
# Purpose: Endpoints for document uploading and ingestion status tracking.

import shutil
import uuid
from pathlib import Path
from typing import Any

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

from app.db.models import Document
from app.dependencies import get_current_user, get_db_session, get_qdrant, get_redis
from app.ingestion.pipeline import run_ingestion
from app.logging_config.setup import get_logger

logger = get_logger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    department_id: str = Form(...),
    access_level: int = Form(...),
    file: UploadFile = File(...),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
) -> dict[str, Any]:
    """
    Accepts a document file, saves it temporarily, and starts the ingestion background task.
    """
    file_ext = Path(file.filename or "").suffix.lstrip(".")
    if file_ext.lower() not in ["pdf", "docx", "doc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Only PDF and DOCX are allowed.",
        )

    document_id = uuid.uuid4()
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
