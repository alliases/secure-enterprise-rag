# File: app/auth/jwt_handler.py
# Purpose: JWT encoding and decoding utilities.

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import jwt

from app.config import get_settings


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """
    Creates a secure JWT token with user context payload.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        # Default expiration is 30 minutes
        expire = datetime.now(UTC) + timedelta(minutes=30)

    to_encode.update({"exp": expire})

    # settings.jwt_secret is a SecretStr, requires get_secret_value()
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def decode_token(token: str) -> dict[str, Any]:
    """
    Decodes and validates a JWT token.
    Raises jose.JWTError if token is expired, tampered with, or invalid.
    """
    settings = get_settings()

    payload = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
    )
    return payload
