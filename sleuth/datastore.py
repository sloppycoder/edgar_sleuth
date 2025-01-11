import logging
import re
from functools import lru_cache
from typing import Any

import psycopg

import config

logger = logging.getLogger(__name__)

_sql_select_regex = re.compile(r"\bSELECT\b.*?\bFROM\b", re.DOTALL | re.IGNORECASE)


class DatabaseException(Exception):
    pass


# use lru_cache to make a singleton
@lru_cache(maxsize=1)
def _conn() -> psycopg.Connection:
    return psycopg.connect(config.database_url)


def relevant_chunks_with_distances(
    cik: str,
    accession_number: str,
    embedding_table_name: str,
    search_phrase_table_name: str,
    search_phrase_tag: str,
    embedding_tag: str,
    limit: int = 1000,
):
    """
    Perform a vector search and return the most relevent chunks numbers
    along with their distance
    """
    result = execute_query(
        f"""
        SELECT
            cik, accession_number, phrase, chunk_num,
            embedding <=> phrase_embedding as distance
        FROM
            {search_phrase_table_name} phrases,
            {embedding_table_name} docs
        WHERE
            cik = %s AND accession_number = %s
            AND %s = ANY(phrases.tags) AND %s = ANY(docs.tags)
        ORDER BY
            embedding <=> phrase_embedding
        LIMIT {limit};
    """,
        (cik, accession_number, search_phrase_tag, embedding_tag),
    )
    return result


def get_chunks(
    cik: str,
    accession_number: str,
    table_name: str,
    tag: str,
    chunk_nums: list[int] = [],
) -> list[dict[str, Any]]:
    col = "embeddings" if "embedding" in table_name else "chunk_text"

    query = f"""
        SELECT cik, accession_number, chunk_num, {col}
        FROM {table_name}
        WHERE cik = %s AND accession_number = %s AND %s = ANY(tags)
    """
    params = (cik, accession_number, tag)
    if chunk_nums:
        query += " AND chunk_num = ANY(%s)"
        params += (chunk_nums,)

    return execute_query(query + " ORDER BY chunk_num", params)


def save_chunks(
    cik: str,
    accession_number: str,
    chunks: list[str] | list[list[float]],
    table_name: str,
    tags: list[str] = [],
    create_table: bool = False,
) -> None:
    if len(chunks) == 0:
        return

    # dimenion only matters when saving embeddings
    dimension = len(chunks[0]) if isinstance(chunks[0], list) else 0

    if create_table:
        _create_table(table_name, dimension=dimension)

    try:
        _conn().execute(
            f"""
            DELETE FROM {table_name} WHERE cik = %s
            AND accession_number = %s AND tags = %s
        """,  # pyright: ignore
            (cik, accession_number, tags),
        )
    except psycopg.errors.UndefinedTable:
        _conn().rollback()

    data = [
        (
            cik,
            accession_number,
            chunk_num,
            text_or_embedding,
            tags,
        )
        for chunk_num, text_or_embedding in enumerate(chunks)
    ]
    with _conn().cursor() as cur:
        col = "embedding" if dimension > 10 else "chunk_text"
        cur.executemany(
            f"""
            INSERT INTO {table_name} (cik, accession_number, chunk_num, {col}, tags)
            VALUES (%s, %s, %s, %s, %s)
            """,  # pyright: ignore
            data,
        )
    _conn().commit()


def execute_query(query, params=None) -> list[dict[str, Any]]:
    result = []
    with _conn().cursor() as cur:
        logger.debug(f"Executing query: {query}\nwith parameters: {params}")

        try:
            if _sql_select_regex.search(query):
                # it's a select
                cur.execute(query, params)
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]  # pyright: ignore
                return [dict(zip(column_names, row)) for row in rows]
            else:
                # it's not a select
                cur.execute(query, params)
                _conn().commit()
                return []  # empty list means success

        except psycopg.errors.SyntaxError as e:
            logger.info(f"Syntax error: {str(e)} {query}")
            raise DatabaseException from e
        except psycopg.Error as e:
            logger.info(f"Database error: {e} when executing {query}")
            _conn().rollback()
            raise DatabaseException from e

    return result


def execute_insertmany(
    table_name: str, data: list[dict[str, Any]], create_table: bool
) -> bool:
    if len(data) == 0:
        return False

    dimension = 0
    for _, value in data[0].items():
        if isinstance(value, list):
            if isinstance(value[0], float):
                dimension = len(value)
                break

    if create_table:
        _create_table(table_name, dimension)

    columns = ", ".join(data[0].keys())
    placeholders = ", ".join(["%s"] * len(data[0]))
    bindings = [tuple(item.values()) for item in data]
    query = f" INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    try:
        with _conn().cursor() as cur:
            cur.executemany(query, bindings)  # pyright: ignore
        _conn().commit()
        logger.debug(f"Inserted {len(data)} rows into {table_name}")
        return True
    except psycopg.errors.SyntaxError as e:
        logger.info(f"Syntax error: {str(e)} {query}")
        raise DatabaseException from e
    except psycopg.Error as e:
        logger.info(f"Database error: {e} when executing {query}")
        _conn().rollback()
        raise DatabaseException from e

    return False


def _create_table(table_name: str, dimension: int = 0):
    if table_name.startswith("filing_text_chunks"):
        statement = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cik VARCHAR(10) NOT NULL,
            accession_number VARCHAR(20) NOT NULL,
            chunk_num INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            tags TEXT[])
        """
    elif table_name.startswith("filing_chunks_embeddings"):
        statement = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cik VARCHAR(10) NOT NULL,
            accession_number VARCHAR(20) NOT NULL,
            chunk_num INTEGER NOT NULL,
            embedding VECTOR ({dimension}) NOT NULL,
            tags TEXT[])
        """
    elif table_name.startswith("search_phrase_embeddings"):
        statement = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            phrase VARCHAR(255) PRIMARY KEY,
            phrase_embedding VECTOR({dimension}),
            tags TEXT[]
        )"""
    elif table_name.startswith("trustee_comp_results"):
        statement = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cik VARCHAR(10) NOT NULL,
            accession_number VARCHAR(20) NOT NULL,
            model VARCHAR(32) NOT NULL,
            response TEXT NOT NULL,
            comp_info JSONB,
            n_trustees INTEGER,
            tags TEXT[]
        )"""
    else:
        raise ValueError(f"Do not know how to create table {table_name}")

    return _conn().execute(statement)  # pyright: ignore
