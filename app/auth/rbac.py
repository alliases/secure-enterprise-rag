# File: app/auth/rbac.py
# Purpose: Core Role-Based Access Control logic for the application.

from typing import Any

from fastapi import HTTPException, status


def check_permission(
    user: dict[str, Any], target_department_id: str, action: str
) -> bool:
    """
    Evaluates if a user has the right to perform a specific action
    on data belonging to a target department.
    """
    role = user.get("role")
    user_dept = user.get("department_id")

    if role == "admin":
        return True

    if action == "view_unmasked":
        return role == "hr_manager" and user_dept == target_department_id

    return action == "view_masked"


def require_role(allowed_roles: list[str]) -> Any:
    """
    FastAPI dependency factory to enforce endpoint-level role access.
    Usage: Depends(require_role(["admin", "hr_manager"]))
    """

    def role_checker(current_user: dict[str, Any]) -> dict[str, Any]:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access this resource",
            )
        return current_user

    return role_checker
