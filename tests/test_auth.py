# File: tests/test_auth.py
# Purpose: Unit tests for authentication utilities, JWT handling, and RBAC logic.

import secrets
import uuid
from collections.abc import AsyncGenerator
from datetime import timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt_handler import create_access_token, decode_token
from app.auth.rbac import check_permission
from app.auth.security import get_password_hash, verify_password
from app.db.models import AuditLog, Role, User
from app.dependencies import get_db_session
from app.main import app


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


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    """
    Provides an async HTTP client for the FastAPI app.
    Overrides the DB session dependency to use the isolated Testcontainers session.
    """
    app.dependency_overrides[get_db_session] = lambda: db_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def active_user(db_session: AsyncSession) -> User:
    """Creates a standard active user for testing."""
    hr_role = Role(name="hr_manager", permissions=["view_unmasked", "upload_docs"])
    db_session.add(hr_role)
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        hashed_password=get_password_hash("SecurePass123!"),
        role_name="hr_manager",
        department_id="hr",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.mark.asyncio
async def test_login_success_returns_jwt(
    client: AsyncClient, db_session: AsyncSession, active_user: User
) -> None:
    """Verifies that valid credentials return tokens and log the event."""
    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "SecurePass123!"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # Verify AuditLog creation
    audit_stmt = select(AuditLog).where(AuditLog.user_id == active_user.id)
    audit_logs = (await db_session.scalars(audit_stmt)).all()
    assert len(audit_logs) == 1
    assert audit_logs[0].action == "login"
    assert audit_logs[0].details["event"] == "successful_login"


@pytest.mark.asyncio
async def test_login_wrong_password_returns_401(
    client: AsyncClient, db_session: AsyncSession, active_user: User
) -> None:
    """Verifies that invalid password is rejected and logged."""
    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "WrongPassword!"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"

    # Verify Failed Login AuditLog creation (Task 1.8 requirement)
    audit_stmt = select(AuditLog).where(AuditLog.user_id == active_user.id)
    audit_logs = (await db_session.scalars(audit_stmt)).all()
    assert len(audit_logs) == 1
    assert audit_logs[0].action == "login_failed"


@pytest.mark.asyncio
async def test_login_unknown_email_returns_401(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Verifies that an unknown email is rejected."""
    response = await client.post(
        "/auth/login",
        data={"username": "nobody@example.com", "password": "Password123!"},
    )

    assert response.status_code == 401

    # Verify Failed Login AuditLog creation with None user_id
    audit_stmt = select(AuditLog).where(AuditLog.action == "login_failed")
    audit_logs = (await db_session.scalars(audit_stmt)).all()
    assert len(audit_logs) == 1
    assert audit_logs[0].user_id is None
    assert audit_logs[0].details["email_attempted"] == "nobody@example.com"


@pytest.mark.asyncio
async def test_login_inactive_user_returns_400(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Verifies that disabled accounts cannot log in."""
    viewer_role = Role(name="viewer", permissions=["view_masked"])
    db_session.add(viewer_role)
    user = User(
        id=uuid.uuid4(),
        email="inactive@example.com",
        hashed_password=get_password_hash("Pass123!"),
        role_name="viewer",
        department_id="hr",
        is_active=False,
    )
    db_session.add(user)
    await db_session.commit()

    response = await client.post(
        "/auth/login",
        data={"username": "inactive@example.com", "password": "Pass123!"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Inactive user"


@pytest.mark.asyncio
async def test_refresh_token_rotation(
    client: AsyncClient, db_session: AsyncSession, active_user: User
) -> None:
    """Verifies successful token rotation."""
    raw_secret = secrets.token_urlsafe(32)
    active_user.hashed_refresh_token = get_password_hash(raw_secret)
    await db_session.commit()

    valid_refresh = f"{active_user.id}:{raw_secret}"
    response = await client.post("/auth/refresh", json={"refresh_token": valid_refresh})

    assert response.status_code == 200
    data = response.json()
    assert data["refresh_token"] != valid_refresh  # Token was rotated


@pytest.mark.asyncio
async def test_refresh_token_reuse_detection(
    client: AsyncClient, db_session: AsyncSession, active_user: User
) -> None:
    """Verifies that using an invalid/old refresh token wipes the session."""
    active_user.hashed_refresh_token = get_password_hash("current_valid_secret")
    await db_session.commit()

    # Attempt to use a compromised/old secret
    compromised_token = f"{active_user.id}:old_stolen_secret"
    response = await client.post(
        "/auth/refresh", json={"refresh_token": compromised_token}
    )

    assert response.status_code == 401
    assert "compromised" in response.json()["detail"].lower()

    # Verify session is wiped in DB
    await db_session.refresh(active_user)
    assert active_user.hashed_refresh_token is None
