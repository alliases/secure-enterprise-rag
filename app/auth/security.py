# File: app/auth/security.py
# Purpose: Password hashing and verification using bcrypt.

from passlib.context import CryptContext

# Define the bcrypt context. "deprecated=auto" ensures that if we upgrade
# algorithms in the future, old hashes will be transparently re-hashed.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Safely compares a plaintext password against a bcrypt hash.
    Protects against timing attacks natively.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generates a secure bcrypt hash for a new password.
    """
    return pwd_context.hash(password)
