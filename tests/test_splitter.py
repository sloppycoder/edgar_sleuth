from datastore import save_chunks
from edgar import SECFiling, edgar_file
from splitter import chunk_text


def test_chunk_filing(clean_db):
    chunks = chunk_text("edgar/data/1035018/0001193125-20-000327.txt", method="spacy")
    filing = SECFiling("1035018", "edgar/data/1035018/0001193125-20-000327.txt")
    assert filing

    main_filing_html = filing.get_doc_by_type("485BPOS")[0]
    contents = edgar_file(main_filing_html)
    assert contents

    chunks = chunk_text(contents, method="spacy")
    assert len(chunks) > 1

    save_chunks(
        cik="1035018",
        accession_number="0001193125-20-000327",
        chunks=chunks,
        table_name="filing_text_chunks",
        tags=["pytest"],
        create_table=True,
    )
