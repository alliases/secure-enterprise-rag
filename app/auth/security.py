from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Initialize Argon2id hasher with default OWASP recommended parameters
# Memory cost, time cost, and parallelism are managed automatically
ph = PasswordHasher()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Safely compares a plaintext password against an Argon2id hash.
    Catches VerifyMismatchError to return a simple boolean for the auth flow.
    """
    try:
        # Note: argon2 requires the hash as the first argument
        return ph.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False


def get_password_hash(password: str) -> str:
    """
    Generates a secure Argon2id hash for a new password.
    """
    return ph.hash(password)
