# File: app/ingestion/chunker.py
# Purpose: Split raw text into manageable chunks with semantic overlap.
import re
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
    Splits the input text strictly by paragraphs first to guarantee semantic isolation.
    Falls back to RecursiveCharacterTextSplitter only for oversized paragraphs.
    """
    settings = get_settings()

    # Removed \n\n from separators as it is handled manually prior to LangChain
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n", ". ", " ", ""],
        length_function=len,
    )

    # 1. Strict explicit split by paragraph boundaries
    paragraphs = [p.strip() for p in re.split(r"\r?\n[ \t]*\r?\n", text) if p.strip()]

    final_raw_chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > settings.chunk_size:
            final_raw_chunks.extend(splitter.split_text(paragraph))
        else:
            final_raw_chunks.append(paragraph)

    processed_chunks: list[Chunk] = []

    for index, chunk_text_part in enumerate(final_raw_chunks):
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
                text=chunk_text_part,
                metadata=chunk_metadata,
                chunk_index=index,
            )
        )

    return processed_chunks
