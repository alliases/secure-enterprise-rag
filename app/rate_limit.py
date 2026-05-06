from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize limiter using client IP address as the strict identifier
limiter = Limiter(key_func=get_remote_address)
