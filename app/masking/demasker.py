# File: app/masking/demasker.py
# Purpose: De-masking layer to restore PII in LLM responses for authorized users.

from typing import Any

from redis.asyncio import Redis

from app.auth.rbac import check_permission
from app.logging_config.setup import get_logger
from app.masking.mapping_store import retrieve_mappings

logger = get_logger(__name__)


async def demask_response(
    response_text: str,
    document_ids: list[str],
    target_department_id: str,
    redis: Redis,
    user: dict[str, Any],
) -> str:
    """
    Restores original PII values in the generated LLM response.
    Queries Redis for mappings associated with the source documents.
    Enforces RBAC: only users with 'view_unmasked' permission for the
    target department will receive the de-masked text.
    """
    # 1. Enforce Role-Based Access Control
    has_access = check_permission(
        user=user, target_department_id=target_department_id, action="view_unmasked"
    )

    if not has_access:
        logger.info(
            "Access denied for de-masking. Returning masked response.",
            user_id=user.get("user_id"),
            role=user.get("role"),
            target_department=target_department_id,
        )
        return response_text

    # 2. Retrieve and merge mappings for all referenced documents
    merged_mappings: dict[str, str] = {}
    for doc_id in document_ids:
        doc_mappings = await retrieve_mappings(redis, doc_id)
        merged_mappings.update(doc_mappings)

    if not merged_mappings:
        logger.warning(
            "No mappings found for documents in Redis.",
            document_ids=document_ids,
        )
        return response_text

    # 3. Execute text replacement
    demasked_text = response_text
    replaced_count = 0

    for token, original_value in merged_mappings.items():
        if token in demasked_text:
            demasked_text = demasked_text.replace(token, original_value)
            replaced_count += 1

    # 4. Generate structured audit log event (without leaking actual PII)
    logger.info(
        "De-masking execution successful",
        user_id=user.get("user_id"),
        document_ids=document_ids,
        replaced_tokens_count=replaced_count,
    )

    return demasked_text
