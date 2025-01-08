from edgar import SECFiling
from edgar_sleuth import chunk_filing, get_embeddings


def test_chunk_and_embedding_filing(clean_db):
    # this filing is so large that it triggers warning in spacy
    # filing = SECFiling(idx_filename="edgar/data/1002427/0001133228-24-004879.txt")
    filing = SECFiling(cik="1035018", accession_number="0001193125-20-000327")
    tags = ["pytest"]
    n_chunks = chunk_filing(
        filing,
        form_type="485BPOS",
        tags=tags,
        table_name="filing_text_chunks",
    )
    assert n_chunks == 12

    n_embeddings = get_embeddings(
        text_table_name="filing_text_chunks",
        cik=filing.cik,
        accession_number=filing.accession_number,
        tags=tags,
        embedding_table_name="filing_chunks_embeddings",
    )
    assert n_embeddings == n_chunks
