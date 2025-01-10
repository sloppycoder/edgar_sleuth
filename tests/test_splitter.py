from sleuth.edgar import SECFiling
from sleuth.splitter import chunk_text, trim_html_content


def test_chunk_filing():
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    _, html_content = filing.get_doc_content("485BPOS", max_items=1)[0]
    text_content = trim_html_content(html_content)
    chunks = chunk_text(text_content, method="spacy")
    assert len(chunks) == 271
