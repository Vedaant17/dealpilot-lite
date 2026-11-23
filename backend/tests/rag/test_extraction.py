from pypdf import PdfWriter

from backend.app.rag.extraction import extract_pdf, extract_txt


def _make_pdf(texts):
    writer = PdfWriter()
    for text in texts:
        writer.add_blank_page(width=200, height=200)
        # pypdf blank pages cannot easily carry text; use annotation-free simple PDF via page content stream
    # Instead create a minimal text-bearing PDF manually.
    return None


def test_txt_extraction_and_bom():
    result = extract_txt(b"\xef\xbb\xbftext line\n\nsecond")
    assert result.file_format == "txt"
    assert result.page_count == 1
    assert result.text == "text line\n\nsecond"
    assert result.has_usable_text is True


def test_txt_empty_and_invalid():
    empty = extract_txt(b"   ")
    assert empty.has_usable_text is False
    assert empty.extraction_error == "no extractable text"
    try:
        extract_txt(b"\xff")
        raise AssertionError("expected extraction error")
    except ValueError:
        pass


def test_pdf_extraction_pages():
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 5 0 R /Resources << /Font << /F1 6 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 7 0 R /Resources << /Font << /F1 6 0 R >> >> >>\nendobj\n"
        b"5 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 200 Td (Page one) Tj ET\nendstream\nendobj\n"
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"7 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 200 Td (Page two) Tj ET\nendstream\nendobj\n"
        b"xref\n0 8\n0000000000 65535 f \n"
        b"trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    result = extract_pdf(pdf_bytes)
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "Page one" in result.pages[0].text
    assert "Page two" in result.pages[1].text


def test_pdf_corrupt_and_empty_text():
    try:
        extract_pdf(b"not a pdf")
        raise AssertionError("expected extraction error")
    except ValueError:
        pass

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n"
        b"xref\n0 5\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )
    result = extract_pdf(pdf_bytes)
    assert result.has_usable_text is False
    assert result.extraction_error == "no extractable text"
