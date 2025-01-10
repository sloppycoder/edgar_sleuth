from sleuth.edgar import SECFiling
from sleuth.splitter import chunk_filing


def test_chunk_filing():
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)
    n_chunks, chunks = chunk_filing(
        filing=filing,
        form_type="485BPOS",
        method="spacy",
        tags=["pytest"],
        table_name="",  # empty so that nothing writes to database
    )
    assert n_chunks == 271
    # no chunk is empty or too short
    assert all(chunk and len(chunk) > 10 for chunk in chunks)
