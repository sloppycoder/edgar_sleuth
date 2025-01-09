import logging
import re
from functools import lru_cache
from typing import Any

import psycopg

import config

logger = logging.getLogger(__name__)

sql_select_regex = re.compile(r"\bSELECT\b.*?\bFROM\b", re.DOTALL | re.IGNORECASE)


def _gen_create_statement(table_name: str, dimension: int = 0) -> str:
    if table_name.startswith("filing_text_chunks"):
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cik VARCHAR(10) NOT NULL,
            accession_number VARCHAR(20) NOT NULL,
            chunk_num INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            tags TEXT[])
        """
    elif table_name.startswith("filing_chunks_embeddings"):
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            cik VARCHAR(10) NOT NULL,
            accession_number VARCHAR(20) NOT NULL,
            chunk_num INTEGER NOT NULL,
            embedding VECTOR ({dimension}) NOT NULL,
            tags TEXT[])
        """
    else:
        raise ValueError(f"Do not know how to create table {table_name}")


# use lru_cache to make a singleton
@lru_cache(maxsize=1)
def _conn() -> psycopg.Connection:
    return psycopg.connect(config.database_url)


def save_chunks(
    cik: str,
    accession_number: str,
    chunks: list[str] | list[list[float]],
    table_name: str,
    tags: list[str] = [],
    create_table: bool = False,
    dimension: int = 0,
):
    if create_table:
        statement = _gen_create_statement(table_name, dimension=dimension)
        _conn().execute(statement)  # pyright: ignore

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
            chunk_num + 1,
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


def get_chunks(
    cik: str, accession_number: str, table_name: str, tags: list[str] = []
) -> list[dict[str, Any]]:
    col = "embeddings" if "embedding" in table_name else "chunk_text"

    results = execute_query(
        f"""
        SELECT cik, accession_number, chunk_num, {col}
        FROM {table_name}
        WHERE cik = %s AND accession_number = %s AND tags = %s
    """,
        (cik, accession_number, tags),
    )
    return results


def execute_query(query, params=None) -> list[dict[str, Any]]:
    result = []
    try:
        with _conn().cursor() as cur:
            if sql_select_regex.search(query):
                # it's a select
                cur.execute(query, params)
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]  # pyright: ignore
                result = [dict(zip(column_names, row)) for row in rows]
            else:
                # it's not a select
                cur.execute(query, params)
                _conn().commit()
    except psycopg.errors.SyntaxError as e:
        logger.info(f"Syntax error: {str(e)} {query}")
    except psycopg.Error as e:
        logger.info(f"Database error: {e} when executing {query}")

    return result
