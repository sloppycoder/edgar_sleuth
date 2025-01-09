import os

import pytest

from edgar import SECFiling
from edgar_sleuth import chunk_filing, get_embeddings
from edgar_sleuth.trustee import (
    ask_model_about_trustee_comp,
    create_search_phrase_embeddings,
    find_relevant_text,
)
from llm.embedding import GEMINI_EMBEDDING_MODEL

run_models = os.environ.get("PYTEST_RUN_MODELS", "0") == "1"


@pytest.mark.skipif(
    not run_models or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="reduce runtime and cost for API calling",
)
def test_process_one_filing(clean_db):
    """take one simple filing and run through the entire flow"""

    # table names used for testing
    text_table_name = "filing_text_chunks"
    embedding_table_name = "filing_chunks_embeddings"
    search_phrase_table_name = "search_phrase_embeddings"

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)

    # arbitrary tags for testing
    tags = ["pytest"]

    # the entire process has 4 steps

    # step 1: chunk the filing
    n_chunks = chunk_filing(
        filing,
        form_type="485BPOS",
        tags=tags,
        table_name=text_table_name,
    )
    assert n_chunks == 272

    # step 2: get embedding
    n_embeddings = get_embeddings(
        text_table_name=text_table_name,
        cik=cik,
        accession_number=accession_number,
        tags=tags,
        embedding_table_name=embedding_table_name,
    )
    assert n_embeddings == n_chunks

    # step 3: using search phrases to run vector search
    # use scoring alborithm to determine the most relevant text chunks
    search_phrase_tag = "gemini_768"
    create_search_phrase_embeddings(
        search_phrase_table_name,
        model=GEMINI_EMBEDDING_MODEL,
        tags=tags + [search_phrase_tag],
    )
    relevant_text = find_relevant_text(
        cik=filing.cik,
        accession_number=filing.accession_number,
        text_table_name=text_table_name,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        embedding_tag=tags[0],
        search_phrase_tag=search_phrase_tag,
    )
    assert relevant_text and len(relevant_text) > 100

    # step 4: send the relevant text to the LLM model with designed prompt
    # TODO: implement this step
    response, comp_info = ask_model_about_trustee_comp(
        "gemini-1.5-flash-002", relevant_text
    )
    assert response and comp_info and "trustees" in comp_info
