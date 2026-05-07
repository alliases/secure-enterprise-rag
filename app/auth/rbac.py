# File: app/auth/rbac.py
from collections.abc import Callable
from typing import Any

from fastapi import Depends, HTTPException, status

from app.dependencies import get_current_user


def check_permission(
    user: dict[str, Any], target_department_id: str, action: str
) -> bool:
    role = user.get("role")
    user_dept = user.get("department_id")

    if role == "admin":
        return True

    if action == "view_unmasked":
        return role == "hr_manager" and user_dept == target_department_id

    return action == "view_masked"


def require_role(
    allowed_roles: list[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    # КРИТИЧНО: Depends(get_current_user) має бути тут
    def role_checker(
        current_user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access this resource",
            )
        return current_user

    return role_checker
