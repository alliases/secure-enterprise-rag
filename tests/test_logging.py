"""
File: tests/test_logging.py
"""

import logging
from collections.abc import MutableMapping
from typing import Any

from app.logging_config.setup import pii_sanitizer


def test_pii_sanitizer_no_pii() -> None:
    """Verifies that normal log messages are not altered."""
    event_dict: MutableMapping[str, Any] = {
        "event": "User logged in successfully",
        "user_id": "uuid-1234",
    }

    result = pii_sanitizer(logging.getLogger(), "test", event_dict.copy())

    assert result["event"] == "User logged in successfully"
    assert result["user_id"] == "uuid-1234"
    assert "security_alert" not in result


def test_pii_sanitizer_redacts_email() -> None:
    """Verifies that email addresses are redacted from log messages and metadata."""
    event_dict: MutableMapping[str, Any] = {
        "event": "Failed login attempt for john.doe@example.com",
        "level": "info",
    }

    result = pii_sanitizer(logging.getLogger(), "test", event_dict.copy())

    assert "john.doe@example.com" not in result["event"]
    assert "[REDACTED_EMAIL]" in result["event"]
    assert result["security_alert"] == "pii_leak_attempt_prevented"
    assert result["level"] == "warning"  # Level should be escalated


def test_pii_sanitizer_redacts_phone() -> None:
    """Verifies that phone numbers are correctly redacted."""
    event_dict: MutableMapping[str, Any] = {
        "event": "User updated profile",
        "contact_info": "Phone number is +380123456789",
    }

    result = pii_sanitizer(logging.getLogger(), "test", event_dict.copy())

    assert "+380123456789" not in result["contact_info"]
    assert "[REDACTED_PHONE]" in result["contact_info"]
    assert result["security_alert"] == "pii_leak_attempt_prevented"


def test_pii_sanitizer_redacts_credit_card() -> None:
    """Verifies that credit card numbers are redacted."""
    event_dict: MutableMapping[str, Any] = {
        "event": "Payment processed for card 1234-5678-9012-3456",
    }

    result = pii_sanitizer(logging.getLogger(), "test", event_dict.copy())

    assert "1234-5678-9012-3456" not in result["event"]
    assert "[REDACTED_CC]" in result["event"]
    assert result["security_alert"] == "pii_leak_attempt_prevented"


def test_pii_sanitizer_multiple_pii_types() -> None:
    """Verifies redaction of multiple PII types across different keys in a single event."""
    event_dict: MutableMapping[str, Any] = {
        "event": "User alice@test.com paid with 4111222233334444",
        "metadata": "Contact: 123-456-7890",
    }

    result = pii_sanitizer(logging.getLogger(), "test", event_dict.copy())

    # Check email and CC in event
    assert "alice@test.com" not in result["event"]
    assert "[REDACTED_EMAIL]" in result["event"]
    assert "4111222233334444" not in result["event"]
    assert "[REDACTED_CC]" in result["event"]

    # Check phone in metadata
    assert "123-456-7890" not in result["metadata"]
    assert "[REDACTED_PHONE]" in result["metadata"]

    # Check security flag
    assert result["security_alert"] == "pii_leak_attempt_prevented"
