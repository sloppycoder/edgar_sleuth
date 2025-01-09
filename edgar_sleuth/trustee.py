import logging

from datastore import execute_query
from llm.embedding import GEMINI_EMBEDDING_MODEL, batch_embedding

logger = logging.getLogger(__name__)

TRUSTEE_COMP_SEARCH_PHRASES = [
    "Trustee Compensation",
    "Independent Director Compensation",
    "Board Director Compensation",
    "Interested Person Compensation",
]


def create_search_phrases_embeddings(table_name: str, tags: list[str] = []) -> None:
    tags = tags or ["gemini-768"]

    embeddings = batch_embedding(
        TRUSTEE_COMP_SEARCH_PHRASES,
        GEMINI_EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
    )

    execute_query(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            phrase VARCHAR(255) PRIMARY KEY,
            phrase_embedding VECTOR(768),
            tags TEXT[]
        )""")

    execute_query(
        f"DELETE FROM {table_name} WHERE phrase = ANY(%s) AND tags = %s",
        (TRUSTEE_COMP_SEARCH_PHRASES, tags),
    )

    for n, phrase in enumerate(TRUSTEE_COMP_SEARCH_PHRASES):
        execute_query(
            f"INSERT INTO {table_name} (phrase, phrase_embedding, tags) VALUES (%s, %s, %s)",  # noqa: E501
            (phrase, embeddings[n], tags),
        )
