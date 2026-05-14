# File: app/masking/mapping_store.py
# Purpose: CRUD operations for PII mappings in Redis.
import re
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import cast

from cryptography.fernet import Fernet
from redis.asyncio import Redis

from app.config import get_settings
from app.logging_config.setup import get_logger

logger = get_logger(__name__)

# Constants
TTL_DAYS = 30
PREFIX = "pii"


def _get_cipher() -> Fernet:
    """Instantiates the Fernet symmetric encryption cipher."""
    settings = get_settings()
    key = settings.redis_encryption_key.get_secret_value().encode("utf-8")
    return Fernet(key)


async def store_mappings(
    redis: Redis, document_id: str, mappings: dict[str, str]
) -> int:
    """
    Encrypts and stores PII mappings in Redis using a pipeline.
    Keys are formatted as: pii:{document_id}:{token}
    """
    if not mappings:
        return 0

    cipher = _get_cipher()

    async with redis.pipeline(transaction=True) as pipe:
        for token, original_value in mappings.items():
            key = f"{PREFIX}:{document_id}:{token}"
            encrypted_value = cipher.encrypt(original_value.encode("utf-8")).decode(
                "utf-8"
            )
            pipe.setex(name=key, time=timedelta(days=TTL_DAYS), value=encrypted_value)

        results = await pipe.execute()

    return len(results)


async def retrieve_mappings(redis: Redis, document_id: str) -> dict[str, str]:
    """
    Retrieves all PII mappings associated with a specific document_id.
    Uses SCAN over KEYS to avoid blocking the Redis event loop.
    """
    match_pattern = f"{PREFIX}:{document_id}:*"
    mappings: dict[str, str] = {}

    # Cast scan_iter to provide explicit types and ignore third-party stub incomplete types
    scan_iterator = cast(AsyncIterator[bytes], redis.scan_iter(match=match_pattern))  # type: ignore[reportUnknownMemberType]

    cipher = _get_cipher()

    async for key_bytes in scan_iterator:
        key = key_bytes.decode("utf-8")

        value_bytes = cast(bytes | None, await redis.get(key))

        if value_bytes:
            token = key.split(":")[-1]
            try:
                decrypted_value = cipher.decrypt(value_bytes).decode("utf-8")
                mappings[token] = decrypted_value
            except Exception as e:
                logger.error("Failed to decrypt PII mapping", key=key, error=str(e))
                continue

    return mappings


async def delete_mappings(redis: Redis, document_id: str) -> int:
    """
    Deletes all mappings for a document to comply with GDPR 'Right to be forgotten'.
    """
    match_pattern = f"{PREFIX}:{document_id}:*"
    keys_to_delete: list[bytes] = []

    # Cast scan_iter to provide explicit types and ignore third-party stub incomplete types
    scan_iterator = cast(AsyncIterator[bytes], redis.scan_iter(match=match_pattern))  # type: ignore[reportUnknownMemberType]

    async for key_bytes in scan_iterator:
        keys_to_delete.append(key_bytes)

    if not keys_to_delete:
        return 0

    # Delete all matched keys in a single command
    deleted_count = await redis.delete(*keys_to_delete)
    return deleted_count


async def get_max_token_indices(redis: Redis, document_id: str) -> dict[str, int]:
    """
    Hydrates the maximum token index for each entity type from an existing document.
    Used to prevent token collisions during incremental document updates.
    """
    mappings = await retrieve_mappings(redis, document_id)
    counters: dict[str, int] = {}

    for token in mappings:
        # Parses tokens like [PERSON_1], [FINANCIAL_12]
        match = re.match(r"\[([A-Z_]+)_(\d+)\]", token)
        if match:
            entity_type = match.group(1)
            index = int(match.group(2))
            if index > counters.get(entity_type, 0):
                counters[entity_type] = index

    return counters
