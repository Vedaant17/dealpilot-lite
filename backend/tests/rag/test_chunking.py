from backend.app.rag.chunking import (
    chunk_document,
    chunk_extraction,
    chunk_text,
    validate_chunk_config,
)
from backend.app.rag.extraction import ExtractionResult, PageExtraction


def _extraction(pages):
    return ExtractionResult(
        file_format="pdf",
        page_count=len(pages),
        pages=[PageExtraction(page_number=i + 1, text=text) for i, text in enumerate(pages)],
        text="\n\n".join(pages),
    )


def test_single_page_txt():
    extraction = _extraction(["Hello world"])
    chunks = chunk_extraction(extraction, "doc-1", "aurora-retail")
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].char_count == 11
    assert chunks[0].token_count_estimate == 3
    assert chunks[0].chunking_version == "char-v1-3200-400"


def test_multi_page_provenance():
    extraction = _extraction(["Page one text", "Page two text"])
    chunks = chunk_extraction(extraction, "doc-1", "aurora-retail", chunk_size=12, chunk_overlap=2)
    assert len(chunks) >= 2
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2


def test_empty_and_whitespace():
    empty = _extraction(["", "   "])
    assert chunk_extraction(empty, "doc-1", "aurora-retail") == []


def test_long_unbroken_string():
    text = "a" * 4000
    chunks = chunk_text(
        text, document_id="doc-1", company_id="aurora-retail", chunk_size=1000, chunk_overlap=100
    )
    assert len(chunks) >= 4
    assert all(len(chunk.text) <= 1000 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_exact_boundary_and_overlap():
    text = "word " * 700
    chunks = chunk_text(
        text, document_id="doc-1", company_id="aurora-retail", chunk_size=100, chunk_overlap=20
    )
    assert len(chunks) > 1
    assert chunks[0].text.endswith("word")
    assert chunks[1].text.startswith("word")


def test_determinism():
    extraction = _extraction(["Deterministic chunking test."])
    first = chunk_extraction(extraction, "doc-1", "aurora-retail")
    second = chunk_extraction(extraction, "doc-1", "aurora-retail")
    assert first == second


def test_text_hash_and_version():
    extraction = _extraction(["Hash me"])
    chunks = chunk_extraction(extraction, "doc-1", "aurora-retail", chunk_size=10, chunk_overlap=2)
    assert chunks[0].text_hash
    assert chunks[0].chunking_version == "char-v1-10-2"


def test_invalid_config():
    try:
        validate_chunk_config(0, 0)
        raise AssertionError("expected error")
    except ValueError:
        pass
    try:
        validate_chunk_config(10, 10)
        raise AssertionError("expected error")
    except ValueError:
        pass


def test_aliases():
    extraction = _extraction(["Alias test"])
    assert chunk_document(extraction, "doc-1", "aurora-retail") == chunk_extraction(
        extraction, "doc-1", "aurora-retail"
    )
