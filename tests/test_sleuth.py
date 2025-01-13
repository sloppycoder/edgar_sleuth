import os

import pytest
from click.testing import CliRunner

from sleuth.__main__ import main, save_master_idx
from sleuth.llm.embedding import GEMINI_EMBEDDING_MODEL
from sleuth.processor import process_filing
from sleuth.trustee import (
    create_search_phrase_embeddings,
)

run_models = os.environ.get("PYTEST_RUN_MODELS", "0") == "1"


@pytest.mark.skipif(
    not run_models or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="reduce runtime and cost for API calling",
)
def test_process_filing(clean_db):
    """take one simple filing and run through the entire flow"""

    # table names used for testing
    dimension = 768
    tag = "pytest"

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    form_type = "485BPOS"

    create_search_phrase_embeddings(
        "search_phrase_embeddings",
        model=GEMINI_EMBEDDING_MODEL,
        tags=[tag],
        dimension=dimension,
    )

    result = process_filing(
        cik=cik,
        accession_number=accession_number,
        action="chunk",
        input_table="master_idx_sample",
        input_tag=tag,
        output_table="filing_text_chunks",
        output_tags=[tag],
        form_type=form_type,
        model="",
        text_table_name="",
        search_phrase_table_name="",
        dimension=dimension,
    )
    assert result

    result = process_filing(
        cik=cik,
        accession_number=accession_number,
        action="embedding",
        input_table="filing_text_chunks",
        input_tag=tag,
        output_table="filing_chunks_embeddings",
        output_tags=[tag],
        form_type=form_type,
        model=GEMINI_EMBEDDING_MODEL,
        text_table_name="filing_text_chunks",
        search_phrase_table_name="search_phrase_embeddings",
        dimension=dimension,
    )
    assert result

    result = process_filing(
        cik=cik,
        accession_number=accession_number,
        action="extract",
        input_table="filing_chunks_embeddings",
        input_tag=tag,
        output_table="trustee_comp_results",
        output_tags=[tag],
        form_type=form_type,
        model="gemini-1.5-flash-002",
        text_table_name="filing_text_chunks",
        search_phrase_table_name="search_phrase_embeddings",
        dimension=dimension,
    )
    assert result


def test_sleuth_cli():
    runner = CliRunner()
    runner.invoke(main, ["chunk"])


def test_save_master_idx(clean_db):
    assert save_master_idx(2020, 1, form_type_filter="485BPOS", tags=["pytest"]) == 1824
