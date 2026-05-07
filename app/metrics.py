"""
File: app/metrics.py
Task: 3.1 - Prometheus Custom Metrics
"""

from prometheus_client import Counter, Histogram

# Metric for tracking total number of ingested documents by status
INGESTION_TOTAL = Counter(
    "document_ingestion_total",
    "Total document ingestion outcomes",
    ["status"],  # Labels: 'done', 'error'
)

# Metric for tracking specific PII entities intercepted by the masking engine
PII_ENTITIES_TOTAL = Counter(
    "pii_entities_found_total",
    "Total PII entities detected and masked by type",
    ["entity_type"],  # Labels: 'PERSON', 'EMAIL_ADDRESS', etc.
)

# Metric for measuring the latency of the entire LangGraph RAG pipeline
RAG_QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Full RAG pipeline latency in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
)
