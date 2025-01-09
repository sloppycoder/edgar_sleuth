import random

from sleuth.datastore import execute_query, initialize_search_phrases, save_chunks


def test_execute_query():
    result = execute_query("SELECT * FROM information_schema.tables LIMIT 10")
    assert len(result) == 10
    assert "table_name" in result[0].keys()


def test_save_text_chunks(clean_db):
    text_chunks = [
        "The quick brown fox jumps over the lazy dog.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Jelly beans flavor the rainbow with joy.",
        "Zebras zigzag swiftly across the savannah.",
        "Bright stars twinkle in the midnight sky.",
    ]
    save_chunks(
        cik="12345678",
        accession_number="0001111111-88-666666",
        chunks=text_chunks,
        table_name="filing_text_chunks",
        tags=["testing", "pytest"],
        create_table=True,
    )


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
        tags=["testing", "pytest"],
        create_table=True,
    )


def test_initialize_search_phrases(clean_db):
    dimension = 768
    search_phrase_table_name = "search_phrase_embeddings"
    random_data = [
        ("The quick brown", _rand_vec(dimension)),
        ("A wizard's job", _rand_vec(dimension)),
    ]
    initialize_search_phrases(
        table_name=search_phrase_table_name,
        tags=["pytest"],
        data=random_data,
    )


def _rand_vec(dimension: int) -> list[float]:
    return [random.uniform(0.0, 1.0) for _ in range(dimension)]
