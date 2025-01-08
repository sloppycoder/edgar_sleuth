from edgar import SECFiling
from splitter import chunk_text


def test_chunk_filing():
    cik = "1035018"
    accession_number = "0001193125-20-000327"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    _, html_content = filing.get_doc_content("485BPOS", max_items=1)[0]
    chunks = chunk_text(html_content, method="spacy")
    assert len(chunks) > 1
