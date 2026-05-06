# File: app/api/endpoints/auth.py
# Purpose: Authentication endpoints for login and token refresh.
import secrets
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import create_access_token
from app.auth.security import get_password_hash, verify_password
from app.db.models import AuditLog, User
from app.dependencies import get_db_session
from app.logging_config.setup import get_logger
from app.rate_limit import limiter

logger = get_logger(__name__)
router = APIRouter()


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    """
    Standardized response model for authentication endpoints.
    Eliminates the use of dict[str, Any].
    """

    access_token: str
    refresh_token: str
    token_type: str = Field(default="bearer")
    expires_in: int = Field(default=1800)


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")  # type: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """
    Authenticates a user and returns a JWT access token.
    Complies with OAuth2 specification (username = email).
    """
    stmt = select(User).where(User.email == form_data.username)
    user = await db.scalar(stmt)

    if not user or not verify_password(form_data.password, user.hashed_password):
        # Record failed login attempt for compliance (SOC2/ISO27001)
        client_ip = request.client.host if request.client else "unknown"
        audit_entry = AuditLog(
            user_id=user.id if user else None,
            action="login_failed",
            details={
                "email_attempted": form_data.username,
                "reason": "invalid_credentials",
            },
            ip_address=client_ip,
        )
        db.add(audit_entry)
        await db.commit()

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

    # Generate opaque refresh token and hash it for secure storage
    raw_secret = secrets.token_urlsafe(32)
    refresh_token = f"{user.id}:{raw_secret}"
    user.hashed_refresh_token = get_password_hash(raw_secret)
    await db.commit()

    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token, expires_in=1800
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")  # type: ignore[reportUntypedFunctionDecorator, reportUnknownMemberType]
async def refresh_token(
    request: Request,
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """
    Validates the refresh token, performs rotation, and invalidates sessions upon reuse detection.
    """
    try:
        user_id_str, raw_secret = payload.refresh_token.split(":", 1)
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        # Suppress exception context to prevent internal traceback leakage
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid refresh token format",
        ) from None

    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive user"
        )

    if not user.hashed_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or session logged out",
        )

    # Verify secret. If verification fails, it's a potential reuse attack.
    if not verify_password(raw_secret, user.hashed_refresh_token):
        logger.warning(
            "Refresh token reuse detected, wiping sessions", user_id=str(user.id)
        )
        user.hashed_refresh_token = None
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token compromised. All sessions invalidated.",
        )

    # Token rotation: Generate new tokens and invalidate the old one
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "role": user.role_name,
            "department_id": user.department_id,
        }
    )

    new_raw_secret = secrets.token_urlsafe(32)
    new_refresh_token = f"{user.id}:{new_raw_secret}"
    user.hashed_refresh_token = get_password_hash(new_raw_secret)
    await db.commit()

    return TokenResponse(
        access_token=access_token, refresh_token=new_refresh_token, expires_in=1800
    )
