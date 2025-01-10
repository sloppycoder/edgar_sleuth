from sleuth.edgar import SECFiling
from sleuth.splitter import chunk_text, trim_html_content


def test_chunk_filing():
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    filing_path, filing_content = filing.get_doc_content("485BPOS", max_items=1)[0]

    assert filing_path.endswith(".html") or filing_path.endswith(".htm")

    trimmed_html = trim_html_content(filing_content)
    chunks = chunk_text(trimmed_html, method="spacy")

    assert len(chunks) == 271
    # no chunk is empty or too short
    assert all(chunk and len(chunk) > 10 for chunk in chunks)
