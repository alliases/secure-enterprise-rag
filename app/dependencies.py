from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import decode_token
from app.db.models import User

# Defines the scheme for Swagger UI integration and token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
limiter = Limiter(key_func=get_remote_address)


async def get_redis(request: Request) -> Redis:
    """
    Returns an active Redis client from the global application state.
    """
    return request.app.state.redis


async def get_qdrant(request: Request) -> AsyncQdrantClient:
    """
    Returns an active Qdrant client from the global application state.
    """
    return request.app.state.qdrant


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession]:
    # ... (залишається без змін)
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """
    Validates the provided JWT, checks user existence and active status in DB.
    Raises 401 if token is invalid, expired, or user is disabled.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
    # Catch the error as 'e' and chain it using 'from e'
    except JWTError as e:
        raise credentials_exception from e

    # Execute DB lookup to ensure user hasn't been deleted or deactivated
    stmt = select(User).where(User.id == user_id_str)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user account"
        )

    return {
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role_name,
        "department_id": user.department_id,
    }
