# File: app/masking/presidio_engine.py
# Purpose: PII identification and masking logic using Microsoft Presidio.
import re
from dataclasses import dataclass, field
from typing import cast

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from app.masking.regex_patterns import get_custom_recognizers
from app.metrics import PII_ENTITIES_TOTAL

_EMBEDDING_NORMALIZE_REGEX = re.compile(r"\[([A-Z_]+)_\d+\]")


def normalize_for_embedding(text: str) -> str:
    """
    Strips numerical indices from PII tokens (e.g., [PERSON_1] -> [PERSON]).
    This is critical for semantic embedding consistency. It prevents index shifting
    caused by document amendments from artificially lowering the cosine similarity
    between identical paragraphs.
    """
    # Matches any uppercase string with underscores inside brackets, followed by _ and digits
    return _EMBEDDING_NORMALIZE_REGEX.sub(r"[\1]", text)


@dataclass
class MaskedResult:
    """
    Data container for the masked text and its corresponding PII mappings.
    """

    masked_text: str
    mappings: dict[str, str] = field(default_factory=lambda: cast(dict[str, str], {}))


def initialize_analyzer() -> AnalyzerEngine:
    """
    Initializes the Presidio AnalyzerEngine and registers custom recognizers.
    """
    analyzer = AnalyzerEngine()

    for recognizer in get_custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


# Singleton instantiation to prevent reloading NLP models on every request
_analyzer = initialize_analyzer()


def analyze_text(text: str, language: str = "en") -> list[RecognizerResult]:
    """
    Identifies PII entities in the provided text.
    Uses a lowered score threshold (0.4) to prioritize recall over precision,
    reducing the risk of PII leakage.
    """
    return _analyzer.analyze(text=text, language=language, score_threshold=0.4)


def mask_text(
    text: str,
    analyzer_results: list[RecognizerResult],
    entity_counters: dict[str, int] | None = None,
) -> MaskedResult:
    """
    Replaces identified PII entities with incremental tokens (e.g., [PERSON_1]).
    Extracts mappings for later de-masking.

    Args:
        text: The raw text string to mask.
        analyzer_results: List of PII entities detected by Presidio.
        entity_counters: External state dictionary to maintain token index continuity
                         across multiple chunks of the same document.
    """
    # 1. Resolve overlaps using Greedy Interval Scheduling
    # Sort priority: Earliest start -> Longest length -> Highest confidence score
    sorted_for_overlap = sorted(
        analyzer_results, key=lambda x: (x.start, -x.end, -x.score)
    )
    filtered_results: list[RecognizerResult] = []

    for res in sorted_for_overlap:
        # If the start of the current entity is before the end of the last added entity, it's an overlap. Skip it.
        if not filtered_results or res.start >= filtered_results[-1].end:
            filtered_results.append(res)

    # 2. Sort results left-to-right (ascending) to maintain natural chronological token indexing.
    # String index shifting is prevented later by applying replacements right-to-left.
    sorted_ltr_results = sorted(filtered_results, key=lambda x: x.start)

    masked_text = text
    mappings: dict[str, str] = {}

    # Initialize local state if no external state is provided (fallback/testing)
    if entity_counters is None:
        entity_counters = {}

    # Store operations to apply them right-to-left later
    replacements: list[tuple[int, int, str, str]] = []

    for result in sorted_ltr_results:
        entity_type = result.entity_type
        original_value = text[result.start : result.end]
        PII_ENTITIES_TOTAL.labels(entity_type=entity_type).inc()

        # Assign incrementing IDs naturally from the start of the document
        count = entity_counters.get(entity_type, 0) + 1
        entity_counters[entity_type] = count

        token = f"[{entity_type}_{count}]"
        replacements.append((result.start, result.end, token, original_value))

    # 3. Apply replacements right-to-left to prevent string index shifting
    for start, end, token, original_value in reversed(replacements):
        mappings[token] = original_value
        masked_text = masked_text[:start] + token + masked_text[end:]

    return MaskedResult(masked_text=masked_text, mappings=mappings)
