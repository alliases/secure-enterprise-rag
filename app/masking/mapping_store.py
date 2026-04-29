# File: app/masking/mapping_store.py
# Purpose: CRUD operations for PII mappings in Redis.

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import cast

from redis.asyncio import Redis

# Constants
TTL_DAYS = 30
PREFIX = "pii"


async def store_mappings(
    redis: Redis, document_id: str, mappings: dict[str, str]
) -> int:
    """
    Stores PII mappings in Redis using a pipeline for atomic execution and performance.
    Keys are formatted as: pii:{document_id}:{token}
    """
    if not mappings:
        return 0

    # Using a pipeline to minimize network round-trips
    async with redis.pipeline(transaction=True) as pipe:
        for token, original_value in mappings.items():
            key = f"{PREFIX}:{document_id}:{token}"
            pipe.setex(name=key, time=timedelta(days=TTL_DAYS), value=original_value)

        # Execute all commands in the pipeline atomically
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

    async for key_bytes in scan_iterator:
        key = key_bytes.decode("utf-8")

        # Cast get result to resolve Unknown type
        value_bytes = cast(bytes | None, await redis.get(key))

        if value_bytes:
            # Extract the token part from the key (pii:doc_id:token)
            token = key.split(":")[-1]
            mappings[token] = value_bytes.decode("utf-8")

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
