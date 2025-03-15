# type: ignore

import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from sleuth.datastore import execute_query
from sleuth.llm.algo import (
    gather_chunk_distances,
    relevance_by_distance,
    top_adjacent_chunks,
)
from sleuth.trustee import _find_relevant_text


def test_selected_chunks():
    print("\n")
    df_phrases = get_search_phrases_embeddings()
    for row in get_sample_filings():
        new_chunks, old_chunks = [], []

        cik = row["cik"]  # "1314414"
        accession_number = row["accession_number"]  # "0001580642-24-006321"
        # print(f"testing cik={cik}, accession_number={accession_number}")

        new_chunks = get_relevant_chunk_nums_new(df_phrases, cik, accession_number)
        assert new_chunks

        old_chunks, _ = _find_relevant_text(
            cik=cik,
            accession_number=accession_number,
            text_table_name="filing_text_chunks",
            embedding_table_name="filing_chunks_embeddings",
            search_phrase_table_name="search_phrase_embeddings",
            search_phrase_tag="group1",
            method="distance",
        )
        assert old_chunks

        if new_chunks != old_chunks:
            print(
                f"testing cik={cik}, accession_number={accession_number},old,new={old_chunks} || {new_chunks}"  # noqa E501
            )


def get_relevant_chunk_nums_new(df_phrases, cik, accession_number):
    relevance_result = calculate_distance_with_scipy(df_phrases, cik, accession_number)
    chunk_distances = gather_chunk_distances(relevance_result)
    relevance_scores = relevance_by_distance(chunk_distances)
    selected_chunks = [int(s) for s in top_adjacent_chunks(relevance_scores)]
    return selected_chunks


def calculate_distance_with_scipy(df_phrases, cik, accession_number, limit=20):
    df_chunks = get_filing_embeddings(cik, accession_number)
    assert df_chunks is not None and df_phrases is not None

    distances = []
    for idx, row in df_chunks.iterrows():
        for phrase_idx, phrase_row in df_phrases.iterrows():
            distance = cosine(phrase_row["embedding"], row["embedding"])
            distances.append({"chunk_num": row["chunk_num"], "distance": distance})
            # print(f"{idx},{phrase_idx}->{distance:.6f}")

    df_distances = pd.DataFrame(distances)
    df_distances.sort_values(by="distance", inplace=True)
    return df_distances.head(limit).to_dict(orient="records")


def get_filing_embeddings(cik, accession_number) -> pd.DataFrame | None:
    query = """
        SELECT chunk_num, embedding
        FROM filing_chunks_embeddings
        WHERE cik = %s AND accession_number = %s
        ORDER BY chunk_num
    """
    params = (cik, accession_number)
    rows = execute_query(query, params)
    if rows:
        df_result = pd.DataFrame(rows)
        df_result["embedding"] = df_result["embedding"].apply(
            lambda x: np.array(json.loads(x))
        )
        return df_result
    return None


def get_search_phrases_embeddings() -> pd.DataFrame | None:
    query = """
        SELECT phrase_embedding
        FROM search_phrase_embeddings
        WHERE %s = ANY(tags)
    """
    rows = execute_query(query, ("group1",))
    if rows:
        df_result = pd.DataFrame(rows)
        df_result["embedding"] = df_result["phrase_embedding"].apply(
            lambda x: np.array(json.loads(x))
        )
        return df_result
    return None


def xget_sample_filings() -> list[dict[str, str]]:
    query = """
        SELECT * FROM
        (
        SELECT distinct cik, accession_number
        FROM filing_chunks_embeddings
        ) CIKS
        ORDER BY random()
        LIMIT 100
    """
    query = """
        SELECT distinct cik, accession_number
        FROM filing_chunks_embeddings
    """
    result = execute_query(query)
    print(f"Total rows={len(result)}")
    return result


def get_sample_filings() -> list[dict[str, str]]:
    return [
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        {"cik": "275309", "accession_number": "0000275309-24-000546"},
        # {"cik": "819118", "accession_number": "0000819118-24-000143"},
        # {"cik": "837276", "accession_number": "0001683863-24-002893"},
        # {"cik": "916053", "accession_number": "0001104659-24-009563"},
        # {"cik": "872625", "accession_number": "0001741773-24-003614"},
    ]
