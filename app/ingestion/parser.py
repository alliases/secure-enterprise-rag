# File: app/ingestion/parser.py
# Purpose: Extract raw text and metadata from uploaded PDF and DOCX files.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from unstructured.documents.elements import Table  # type: ignore[import-untyped]
from unstructured.partition.docx import partition_docx  # type: ignore[import-untyped]
from unstructured.partition.pdf import partition_pdf  # type: ignore[import-untyped]
from unstructured.partition.text import partition_text  # type: ignore[import-untyped]

from app.logging_config.setup import get_logger

logger = get_logger(__name__)


class ParseError(Exception):
    """Custom exception raised when document parsing fails."""

    pass


@dataclass
class ParsedDocument:
    """Data container for the extracted text and its associated metadata."""

    text: str
    metadata: dict[str, Any]


def _extract_text_and_tables(elements: list[Any]) -> str:
    """
    Iterates through unstructured elements, preserving structural integrity.
    If an element is a Table, it attempts to extract the HTML or Markdown
    representation to maintain cell relationships for the LLM.
    """
    text_blocks: list[str] = []

    for element in elements:
        if isinstance(element, Table):
            # Safe dynamic attribute access for untyped library
            metadata = getattr(element, "metadata", None)
            md_text = getattr(metadata, "text_as_markdown", None) if metadata else None
            html_text = getattr(metadata, "text_as_html", None) if metadata else None

            if md_text:
                text_blocks.append(str(md_text))
            elif html_text:
                text_blocks.append(str(html_text))
            else:
                text_blocks.append(str(element))
        else:
            text_blocks.append(str(element))

    return "\n\n".join(text_blocks)


def parse_pdf(file_path: Path) -> tuple[str, int]:
    """
    Extracts text from a PDF file using unstructured.io.
    Utilizes 'hi_res' strategy to trigger layout detection and table extraction.
    """
    try:
        # Cast to list[Any] to bypass strict type checking for untyped external returns
        elements = cast(
            list[Any],
            partition_pdf(
                filename=str(file_path),
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"],
            ),
        )

        # Deduce page count securely using getattr
        page_numbers: list[int] = []
        for el in elements:
            meta = getattr(el, "metadata", None)
            if meta:
                p_num = getattr(meta, "page_number", None)
                if isinstance(p_num, int):
                    page_numbers.append(p_num)

        page_count = max(page_numbers) if page_numbers else 1

        return _extract_text_and_tables(elements), int(page_count)
    except Exception as e:
        logger.error("PDF parsing failed", file_path=str(file_path), error=str(e))
        raise ParseError(f"Failed to parse PDF document: {e!s}") from e


def parse_docx(file_path: Path) -> str:
    """
    Extracts text from a DOCX file using unstructured.io.
    Automatically identifies and extracts standard paragraphs and tables.
    """
    try:
        elements = cast(list[Any], partition_docx(filename=str(file_path)))

        return _extract_text_and_tables(elements)
    except Exception as e:
        logger.error("DOCX parsing failed", file_path=str(file_path), error=str(e))
        raise ParseError(f"Failed to parse DOCX document: {e!s}") from e


def parse_text(file_path: Path) -> str:
    """
    Extracts raw text from plain text formats (.txt, .md, .csv).
    """
    try:
        elements = cast(list[Any], partition_text(filename=str(file_path)))
        return _extract_text_and_tables(elements)
    except Exception as e:
        logger.error("Text parsing failed", file_path=str(file_path), error=str(e))
        raise ParseError(f"Failed to parse text document: {e!s}") from e


def parse_document(file_path: Path, file_name: str, file_type: str) -> ParsedDocument:
    """
    Main entry point for document parsing.
    Delegates to specific parsers based on the provided file_type.
    """
    if not file_path.exists():
        raise ParseError(f"File not found: {file_path}")

    file_type_lower = file_type.lower()
    extracted_text = ""
    page_count = 0

    if file_type_lower == "pdf":
        extracted_text, page_count = parse_pdf(file_path)
    elif file_type_lower in ["docx", "doc"]:
        extracted_text = parse_docx(file_path)
        # DOCX lacks a strict pagination concept without rendering engines
        page_count = 1
    elif file_type_lower in ["txt", "md", "csv"]:
        extracted_text = parse_text(file_path)
        page_count = 1
    else:
        logger.warning("Unsupported file type attempted", file_type=file_type)
        raise ParseError(f"Unsupported file type: {file_type}")

    metadata = {
        "file_name": file_name,
        "file_type": file_type_lower,
        "page_count": page_count,
    }

    return ParsedDocument(text=extracted_text, metadata=metadata)
