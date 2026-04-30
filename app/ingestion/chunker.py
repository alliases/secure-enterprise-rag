# File: app/ingestion/chunker.py
# Purpose: Split raw text into manageable chunks with semantic overlap.

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


@dataclass
class Chunk:
    """
    Data container representing a single text chunk with its routing metadata.
    """

    text: str
    metadata: dict[str, Any]
    chunk_index: int


def chunk_text(
    text: str,
    document_id: str,
    department_id: str,
    access_level: int,
    source_filename: str,
) -> list[Chunk]:
    """
    Splits the input text into chunks using RecursiveCharacterTextSplitter.
    Injects document-level metadata into each chunk for downstream vector filtering (RBAC).
    """
    settings = get_settings()

    # The separator list is prioritized: paragraphs -> sentences -> words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)
    processed_chunks: list[Chunk] = []

    for index, chunk_text in enumerate(raw_chunks):
        # Isolate metadata per chunk to prevent reference mutation bugs
        chunk_metadata: dict[str, Any] = {
            "document_id": document_id,
            "department_id": department_id,
            "access_level": access_level,
            "source_filename": source_filename,
            "chunk_index": index,
        }

        processed_chunks.append(
            Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_index=index,
            )
        )

    return processed_chunks
