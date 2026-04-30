# File: tests/test_auth.py
# Purpose: Unit tests for authentication utilities, JWT handling, and RBAC logic.

from datetime import timedelta

import pytest
from jose import JWTError

from app.auth.jwt_handler import create_access_token, decode_token
from app.auth.rbac import check_permission
from app.auth.security import get_password_hash, verify_password


def test_password_hashing() -> None:
    """Verifies that Argon2id hashing generates distinct hashes and verifies correctly."""
    password = "SecurePassword123!"
    hashed = get_password_hash(password)

    # The hash must never equal the plaintext password
    assert password != hashed
    # Verification of the correct password must succeed
    assert verify_password(password, hashed) is True
    # Verification of an incorrect password must fail gracefully
    assert verify_password("WrongPassword!", hashed) is False


def test_create_and_decode_jwt() -> None:
    """Verifies JWT creation and successful decoding with payload preservation."""
    payload = {"sub": "user-123", "role": "hr_manager", "department_id": "dept-1"}
    token = create_access_token(data=payload)

    decoded = decode_token(token)
    assert decoded["sub"] == "user-123"
    assert decoded["role"] == "hr_manager"
    assert decoded["department_id"] == "dept-1"
    # Ensure the expiration claim was automatically injected
    assert "exp" in decoded


def test_expired_jwt() -> None:
    """Ensures expired tokens strictly raise a JWTError to prevent replay attacks."""
    payload = {"sub": "user-123"}
    # Create a token that explicitly expired 1 minute ago
    token = create_access_token(data=payload, expires_delta=timedelta(minutes=-1))

    with pytest.raises(JWTError):
        decode_token(token)


def test_rbac_hr_manager() -> None:
    """Verifies strict HR Manager boundary permissions."""
    user = {"role": "hr_manager", "department_id": "dept-1"}

    # Authorized: Can view unmasked data FOR THEIR OWN department
    assert check_permission(user, "dept-1", "view_unmasked") is True
    # Unauthorized: CANNOT view unmasked data for a DIFFERENT department
    assert check_permission(user, "dept-2", "view_unmasked") is False
    # Authorized: Can always view masked data (safe fallback)
    assert check_permission(user, "dept-2", "view_masked") is True


def test_rbac_admin() -> None:
    """Verifies Admin role has global unmasked access."""
    user = {"role": "admin", "department_id": "it-dept"}

    # Authorized: Admin bypasses department constraints
    assert check_permission(user, "dept-1", "view_unmasked") is True
    assert check_permission(user, "dept-2", "view_unmasked") is True


def test_rbac_viewer() -> None:
    """Verifies Viewer role has strictly read-only masked access."""
    user = {"role": "viewer", "department_id": "dept-1"}

    # Unauthorized: Cannot view unmasked data even for their own department
    assert check_permission(user, "dept-1", "view_unmasked") is False
    # Authorized: Can only view masked data
    assert check_permission(user, "dept-1", "view_masked") is True
