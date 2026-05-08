import uuid
from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.rbac import require_role
from app.auth.security import get_password_hash
from app.db.models import Role, User
from app.dependencies import get_db_session
from app.logging_config.setup import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    role_name: str
    department_id: str
    is_active: bool = True


class UserUpdate(BaseModel):
    role_name: str | None = None
    department_id: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    role_name: str
    department_id: str
    is_active: bool

    # Pydantic v2 correct way to enable ORM mode
    model_config = ConfigDict(from_attributes=True)


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db_session),
    admin_user: Mapping[str, str] = Depends(require_role(["admin"])),
) -> User:
    """Creates a new user account with a specific role and department."""
    role_stmt = select(Role).where(Role.name == payload.role_name)
    role = await db.scalar(role_stmt)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role '{payload.role_name}' does not exist",
        )

    existing_user_stmt = select(User).where(User.email == payload.email)
    if await db.scalar(existing_user_stmt):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    new_user = User(
        id=uuid.uuid4(),
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        role_name=payload.role_name,
        department_id=payload.department_id,
        is_active=payload.is_active,
    )
    db.add(new_user)
    await db.commit()

    logger.info(
        "New user created by admin",
        admin_id=admin_user["user_id"],
        target_user=payload.email,
    )
    return new_user


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db_session),
    admin_user: Mapping[str, str] = Depends(require_role(["admin"])),
) -> list[User]:
    """Lists all registered users in the system."""
    stmt = select(User).order_by(User.email)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db_session),
    admin_user: Mapping[str, str] = Depends(require_role(["admin"])),
) -> User:
    """Updates user status, role, or department."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if payload.role_name is not None:
        role_stmt = select(Role).where(Role.name == payload.role_name)
        if not await db.scalar(role_stmt):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Role does not exist"
            )
        user.role_name = payload.role_name

    if payload.department_id is not None:
        user.department_id = payload.department_id

    if payload.is_active is not None:
        user.is_active = payload.is_active

    await db.commit()
    logger.info(
        "User updated by admin",
        admin_id=admin_user["user_id"],
        target_user_id=str(user_id),
    )
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
    admin_user: Mapping[str, str] = Depends(require_role(["admin"])),
) -> None:
    """Forcefully deletes a user account (use with caution)."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if str(user.id) == admin_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin cannot delete their own account via this endpoint",
        )

    await db.delete(user)
    await db.commit()
    logger.warning(
        "User deleted by admin",
        admin_id=admin_user["user_id"],
        target_user_id=str(user_id),
    )
