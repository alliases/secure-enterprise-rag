# File: app/api/endpoints/auth.py
# Purpose: Authentication endpoints for login and token refresh.

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import create_access_token
from app.auth.security import verify_password
from app.db.models import AuditLog, User
from app.dependencies import get_db_session
from app.rate_limit import limiter

router = APIRouter()


@router.post("/login")
@limiter.limit("5/minute")  # type: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    """
    Authenticates a user and returns a JWT access token.
    Complies with OAuth2 specification (username = email).
    """
    stmt = select(User).where(User.email == form_data.username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        # Prevent user enumeration by logging the failure but keeping the error generic
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # Generate JWT payload
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "role": user.role_name,
            "department_id": user.department_id,
        }
    )

    # Record login event in Audit Log
    client_ip = request.client.host if request.client else "unknown"
    audit_entry = AuditLog(
        user_id=user.id,
        action="login",
        details={"event": "successful_login"},
        ip_address=client_ip,
    )
    db.add(audit_entry)

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/refresh")
async def refresh_token() -> dict[str, Any]:
    """
    TODO: Implement refresh token rotation logic.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Refresh token rotation not implemented yet.",
    )
