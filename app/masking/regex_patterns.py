# File: app/masking/regex_patterns.py
# Purpose: Custom regex recognizers for domain-specific PII formats.

from presidio_analyzer import Pattern, PatternRecognizer


def get_custom_recognizers() -> list[PatternRecognizer]:
    """
    Returns a list of custom pattern recognizers.
    Includes employee internal IDs (format: DDDD-DDDD).
    """
    employee_id_pattern = Pattern(
        name="employee_id_regex",
        regex=r"\b\d{4}-\d{4}\b",
        score=0.8,
    )

    employee_id_recognizer = PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[employee_id_pattern],
    )

    return [employee_id_recognizer]
