# File: app/masking/regex_patterns.py
# Purpose: Custom regex recognizers for domain-specific PII formats.

from presidio_analyzer import Pattern, PatternRecognizer


def get_custom_recognizers() -> list[PatternRecognizer]:
    """
    Returns a list of custom pattern recognizers.
    Includes employee internal IDs (format: DDDD-DDDD) and Financial Data.
    """
    # 1. Employee ID Recognizer
    # Bumped score to 1.0 to ensure this custom pattern overrules
    # generic built-in recognizers (like DATE_TIME)
    employee_id_pattern = Pattern(
        name="employee_id_regex",
        regex=r"\b\d{4}-\d{4}\b",
        score=1.0,
    )

    employee_id_recognizer = PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[employee_id_pattern],
    )

    # 2. Financial Data Recognizer (e.g., $150,000 or $150,000.00)
    financial_pattern = Pattern(
        name="financial_regex",
        regex=r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
        score=1.0,
    )

    financial_recognizer = PatternRecognizer(
        supported_entity="FINANCIAL",
        patterns=[financial_pattern],
    )

    return [employee_id_recognizer, financial_recognizer]
