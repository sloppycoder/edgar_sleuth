import os

import pytest

from sleuth import chunk_filing, get_embeddings
from sleuth.edgar import SECFiling
from sleuth.llm.embedding import GEMINI_EMBEDDING_MODEL
from sleuth.trustee import (
    create_search_phrase_embeddings,
    extract_trustee_comp,
)

run_models = os.environ.get("PYTEST_RUN_MODELS", "0") == "1"


@pytest.mark.skipif(
    not run_models or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="reduce runtime and cost for API calling",
)
def test_fully_process_one_filing(clean_db):
    """take one simple filing and run through the entire flow"""

    # table names used for testing
    text_table_name = "filing_text_chunks"
    embedding_table_name = "filing_chunks_embeddings"
    search_phrase_table_name = "search_phrase_embeddings"
    dimension = 256

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    filing = SECFiling(cik=cik, accession_number=accession_number)

    # arbitrary tags for testing
    tag = "pytest"
    search_tag = f"pytest_gemini_{dimension}"

    # the entire process has 4 steps

    # step 1: chunk the filing
    n_chunks = chunk_filing(
        filing,
        form_type="485BPOS",
        tags=[tag],
        table_name=text_table_name,
    )
    assert n_chunks == 271

    # step 2: get embedding
    n_embeddings = get_embeddings(
        text_table_name=text_table_name,
        cik=cik,
        accession_number=accession_number,
        tag=tag,
        embedding_table_name=embedding_table_name,
        dimension=dimension,
    )
    assert n_embeddings == n_chunks

    # step 3: using search phrases to run vector search
    # use scoring alborithm to determine the most relevant text chunks
    create_search_phrase_embeddings(
        search_phrase_table_name,
        model=GEMINI_EMBEDDING_MODEL,
        tag=search_tag,
        dimension=dimension,
    )
    response, comp_info = extract_trustee_comp(
        cik=filing.cik,
        accession_number=filing.accession_number,
        text_table_name=text_table_name,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        tag=tag,
        search_phrase_tag=search_tag,
        model="gemini-1.5-flash-002",
    )
    assert response and comp_info and "trustees" in comp_info
