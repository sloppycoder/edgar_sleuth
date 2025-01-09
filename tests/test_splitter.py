from edgar import SECFiling
from splitter import chunk_text, default_text_converter


def test_chunk_filing():
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    _, html_content = filing.get_doc_content("485BPOS", max_items=1)[0]
    text_content = default_text_converter().handle(html_content)
    chunks = chunk_text(text_content, method="spacy")
    assert len(chunks) == 272
