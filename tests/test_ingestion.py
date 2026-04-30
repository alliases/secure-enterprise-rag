# File: tests/test_ingestion.py
# Purpose: Unit tests for document parsing and chunking logic using pytest-mock.

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from app.ingestion.chunker import chunk_text
from app.ingestion.parser import ParseError, parse_document


def test_parse_pdf_mocked(mocker: MockerFixture) -> None:
    """Verifies PDF text extraction and page counting using mocked PyMuPDF."""
    # Flat mocking using mocker fixture
    mock_fitz = mocker.patch("app.ingestion.parser.fitz.open")
    mocker.patch("pathlib.Path.exists", return_value=True)

    # Setup mock PDF document structure
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2

    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 content."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 content."

    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz.return_value.__enter__.return_value = mock_doc

    parsed = parse_document(Path("dummy.pdf"), "dummy.pdf", "pdf")

    assert parsed.metadata["page_count"] == 2
    assert "Page 1 content." in parsed.text
    assert "Page 2 content." in parsed.text


def test_parse_docx_mocked(mocker: MockerFixture) -> None:
    """Verifies DOCX paragraph extraction using mocked python-docx."""
    mock_docx = mocker.patch("app.ingestion.parser.docx.Document")
    mocker.patch("pathlib.Path.exists", return_value=True)

    mock_doc = MagicMock()
    mock_para = MagicMock()
    mock_para.text = "Paragraph content."

    mock_doc.paragraphs = [mock_para]
    mock_doc.tables = []
    mock_docx.return_value = mock_doc

    parsed = parse_document(Path("dummy.docx"), "dummy.docx", "docx")

    assert parsed.metadata["page_count"] == 1
    assert "Paragraph content." in parsed.text


def test_parse_unsupported_type(mocker: MockerFixture) -> None:
    """Ensures parser strictly rejects unauthorized file types."""
    mocker.patch("pathlib.Path.exists", return_value=True)

    with pytest.raises(ParseError, match="Unsupported file type"):
        parse_document(Path("dummy.txt"), "dummy.txt", "txt")


def test_parse_file_not_found() -> None:
    """Ensures proper error handling when the file does not exist on disk."""
    # No mocking needed here, we want it to fail naturally on missing file
    with pytest.raises(ParseError, match="File not found"):
        parse_document(Path("non_existent.pdf"), "non_existent.pdf", "pdf")


def test_chunker_logic() -> None:
    """Verifies semantic chunking and correct RBAC metadata propagation."""
    text = "A" * 5000

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        department_id="dept-1",
        access_level=2,
        source_filename="test.pdf",
    )

    assert len(chunks) > 1

    first_chunk = chunks[0]
    assert first_chunk.metadata["document_id"] == "doc-1"
    assert first_chunk.metadata["department_id"] == "dept-1"
    assert first_chunk.metadata["access_level"] == 2
    assert first_chunk.metadata["source_filename"] == "test.pdf"
    assert first_chunk.chunk_index == 0
