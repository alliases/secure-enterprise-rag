# File: app/db/audit_log.py
# Purpose: Functions for writing and retrieving structured audit events.

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog


async def log_event(
    session: AsyncSession,
    user_id: uuid.UUID,
    action: str,
    details: dict[str, Any],
    ip_address: str | None = None,
) -> uuid.UUID:
    """
    Creates a record in the audit_log table.
    Acts as a centralized interface to ensure consistency.
    """
    audit_entry = AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=ip_address,
    )
    session.add(audit_entry)
    await session.commit()

    return audit_entry.id


async def get_audit_trail(
    session: AsyncSession,
    user_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    limit: int = 100,
) -> list[AuditLog]:
    """
    Retrieves the audit trail, strictly ordered by time descending.
    Allows filtering by user or specific document inside the JSONB payload.
    """
    stmt = select(AuditLog)

    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)

    if document_id:
        # PostgreSQL JSONB operator extraction to filter by nested document_id
        # op("->>") extracts JSON object field as text
        stmt = stmt.where(AuditLog.details.op("->>")("document_id") == str(document_id))

    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit)

    result = await session.execute(stmt)
    return list(result.scalars().all())
