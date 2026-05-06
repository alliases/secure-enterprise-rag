"""
File: tests/test_dependencies.py
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Final
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import create_access_token
from app.config import get_settings
from app.db.models import User
from app.dependencies import get_current_user, get_qdrant, get_redis

# Define a constant for dummy test hashes to avoid Ruff S106 & S105 false positives
MOCK_HASHED_PASSWORD: Final[str] = "dummy_hashed_password_for_testing"


@pytest.mark.asyncio
async def test_get_current_user_valid_token(db_session: AsyncSession) -> None:
    """Happy path: Valid token and active user returns user payload."""
    user_id = uuid.uuid4()
    test_user = User(
        id=user_id,
        email="active@example.com",
        hashed_password=MOCK_HASHED_PASSWORD,
        role_name="hr_manager",
        department_id="hr",
        is_active=True,
    )
    db_session.add(test_user)
    await db_session.commit()

    token = create_access_token({"sub": str(user_id)})

    result = await get_current_user(token=token, db=db_session)

    assert result["user_id"] == str(user_id)
    assert result["email"] == "active@example.com"
    assert result["role"] == "hr_manager"
    assert result["department_id"] == "hr"


@pytest.mark.asyncio
async def test_get_current_user_inactive_user(db_session: AsyncSession) -> None:
    """Security check: Inactive user in DB should trigger 403 Forbidden."""
    user_id = uuid.uuid4()
    test_user = User(
        id=user_id,
        email="inactive@example.com",
        hashed_password=MOCK_HASHED_PASSWORD,
        role_name="viewer",
        department_id="hr",
        is_active=False,  # <--- Inactive!
    )
    db_session.add(test_user)
    await db_session.commit()

    token = create_access_token({"sub": str(user_id)})

    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=token, db=db_session)

    assert exc.value.status_code == 403
    assert exc.value.detail == "Inactive user account"


@pytest.mark.asyncio
async def test_get_current_user_deleted_user(db_session: AsyncSession) -> None:
    """Security check: Valid token but user no longer in DB should trigger 401."""
    # We generate a token for an ID that doesn't exist in the database
    random_id = str(uuid.uuid4())
    token = create_access_token({"sub": random_id})

    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=token, db=db_session)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Could not validate credentials"


@pytest.mark.asyncio
async def test_get_current_user_expired_token(db_session: AsyncSession) -> None:
    """Security check: Expired JWT should trigger 401."""
    settings = get_settings()
    expire = datetime.now(UTC) - timedelta(minutes=10)  # Expired 10 mins ago

    expired_token = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": expire},
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(HTTPException) as exc:
        await get_current_user(token=expired_token, db=db_session)

    assert exc.value.status_code == 401
    assert "Could not validate credentials" in exc.value.detail


@pytest.mark.asyncio
async def test_state_dependencies() -> None:
    """Verifies that dependency injection pulls correct instances from app state."""
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.redis = "mock_redis_instance"
    mock_request.app.state.qdrant = "mock_qdrant_instance"

    redis_result = await get_redis(mock_request)
    qdrant_result = await get_qdrant(mock_request)

    assert redis_result == "mock_redis_instance"
    assert qdrant_result == "mock_qdrant_instance"
