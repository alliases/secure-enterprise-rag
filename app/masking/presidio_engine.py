# File: app/masking/presidio_engine.py
# Purpose: PII identification and masking logic using Microsoft Presidio.
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from app.masking.regex_patterns import get_custom_recognizers
from app.metrics import PII_ENTITIES_TOTAL


@dataclass
class MaskedResult:
    """
    Data container for the masked text and its corresponding PII mappings.
    """

    masked_text: str
    mappings: dict[str, str] = field(default_factory=lambda: {})


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


def mask_text(text: str, analyzer_results: list[RecognizerResult]) -> MaskedResult:
    """
    Replaces identified PII entities with incremental tokens (e.g., [PERSON_1]).
    Extracts mappings for later de-masking.
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

    # 2. Sort results by start position descending to avoid index shifting during string manipulation
    sorted_results = sorted(filtered_results, key=lambda x: x.start, reverse=True)

    masked_text = text
    mappings: dict[str, str] = {}
    entity_counters: dict[str, int] = {}

    for result in sorted_results:
        entity_type = result.entity_type
        original_value = text[result.start : result.end]
        PII_ENTITIES_TOTAL.labels(entity_type=entity_type).inc()
        # Increment counter to generate unique tokens per entity type (e.g., PERSON_1, PERSON_2)
        count = entity_counters.get(entity_type, 0) + 1
        entity_counters[entity_type] = count

        token = f"[{entity_type}_{count}]"
        mappings[token] = original_value

        # Replace original text with the generated token
        masked_text = masked_text[: result.start] + token + masked_text[result.end :]

    return MaskedResult(masked_text=masked_text, mappings=mappings)
