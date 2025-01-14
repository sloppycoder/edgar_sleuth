import random

from sleuth.datastore import (
    execute_insertmany,
    execute_query,
    get_chunks,
    save_chunks,
)


def test_execute_query():
    result = execute_query("SELECT * FROM information_schema.tables LIMIT 10")
    assert len(result) == 10
    assert "table_name" in result[0].keys()


def test_get_n_save_chunks(clean_db):
    text_chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Jelly beans flavor the rainbow with joy.",
        "Zebras zigzag swiftly across the savannah.",
        "Bright stars twinkle in the midnight sky.",
    ]
    cik = "12345678"
    accession_number = "0001111111-88-666666"
    table_name = "filing_text_chunks_tmp"
    save_chunks(
        cik=cik,
        accession_number=accession_number,
        chunks=text_chunks,
        table_name=table_name,
        create_table=True,
    )

    result = get_chunks(
        cik=cik,
        accession_number=accession_number,
        table_name=table_name,
    )
    assert result and len(result) == 5
    assert "quickly in fog" in result[1]["chunk_text"]

    result = get_chunks(
        cik=cik,
        accession_number=accession_number,
        table_name=table_name,
        chunk_nums=[1, 3],
    )
    assert result and len(result) == 2
    assert "Zebras zigzag" in result[1]["chunk_text"]


def test_save_embedding_chunks(clean_db):
    dimension = 768
    embedding_chunks = [
        _rand_vec(dimension),
        _rand_vec(dimension),
        _rand_vec(dimension),
    ]
    save_chunks(
        cik="12345678",
        accession_number="0001111111-88-666666",
        chunks=embedding_chunks,
        table_name="filing_chunks_embeddings",
        create_table=True,
    )


def test_execute_insertmany(clean_db):
    table_name = "some_funny_table"
    data = [
        {"n_trustee": 1, "cik": "1343", "accession_number": "0001111111-88-666666"},
        {"n_trustee": 2, "cik": "1344", "accession_number": "1111111111-22-333333"},
        {"n_trustee": 9, "cik": "8888", "accession_number": "8888888888-88-888888"},
    ]
    execute_query(
        f"create table {table_name} (n_trustee int, cik text, accession_number text)"
    )
    assert execute_insertmany(table_name=table_name, data=data, create_table=False)


def test_relevant_chunks_with_distances(clean_db):
    # TODO: implement this, need data in test db to work
    pass


def _rand_vec(dimension: int) -> list[float]:
    return [random.uniform(0.0, 1.0) for _ in range(dimension)]
