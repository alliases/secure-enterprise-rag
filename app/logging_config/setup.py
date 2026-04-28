# File: app/logging_config/setup.py
# Purpose: Structured JSON logging configuration with PII sanitization.

import logging
import re

import structlog
from structlog.typing import EventDict

from app.config import get_settings

# Regex patterns for identifying potential PII in logs
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"\+?380\d{9}|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def pii_sanitizer(
    logger: logging.Logger, name: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor that redacts PII from log messages.
    Acts as a failsafe before logs are shipped to external systems.
    """
    for key, value in event_dict.items():
        if isinstance(value, str):
            value = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
            value = PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
            value = CREDIT_CARD_PATTERN.sub("[REDACTED_CC]", value)
            event_dict[key] = value
    return event_dict


def configure_logging() -> None:
    """
    Initializes global structlog configuration.
    Must be called exactly once during application startup.
    """
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            pii_sanitizer,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=settings.log_level.upper(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Retrieves a bound logger instance with the given module name.
    """
    return structlog.get_logger(name)
