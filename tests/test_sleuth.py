import os
import shlex
import unittest.mock

import pytest
from click.testing import CliRunner

from sleuth.__main__ import main, save_master_idx
from sleuth.datastore import execute_insertmany
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

    # default table names
    tables_map = {
        "full-idx": "master_idx",
        "idx": "master_idx_sample",
        "text": "filing_text_chunks",
        "embedding": "filing_chunks_embeddings",
        "result": "trustee_comp_results",
        "search": "search_phrase_embeddings",
    }

    # table names used for testing
    dimension = 768
    idx_tag = "pytest"
    result_tag = "result-pytest"

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    form_type = "485BPOS"

    create_search_phrase_embeddings(
        tables_map["search"],
        model=GEMINI_EMBEDDING_MODEL,
        tag=idx_tag,
        dimension=dimension,
    )

    result = process_filing(
        action="chunk",
        tables_map=tables_map,
        cik=cik,
        accession_number=accession_number,
        idx_tag=idx_tag,
        result_tag=result_tag,
        model="",
        dimension=dimension,
        form_type=form_type,
    )
    assert result

    result = process_filing(
        action="embedding",
        tables_map=tables_map,
        cik=cik,
        accession_number=accession_number,
        idx_tag=idx_tag,
        result_tag=result_tag,
        model=GEMINI_EMBEDDING_MODEL,
        dimension=dimension,
        form_type=form_type,
    )
    assert result

    result = process_filing(
        action="extract",
        tables_map=tables_map,
        cik=cik,
        accession_number=accession_number,
        idx_tag=idx_tag,
        result_tag=result_tag,
        model="gemini-1.5-flash-002",
        dimension=dimension,
        form_type=form_type,
    )
    assert result


@pytest.mark.skipif(
    not run_models or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None,
    reason="reduce runtime and cost for API calling",
)
def test_sleuth_cli(clean_db):
    idx_tag = "pytest-cli-test"
    test_filing = {
        "cik": "1002427",
        "accession_number": "0001133228-24-004879",
        "date_filed": "2024-01-29",  # not important
        "company_name": "VANGUARD GROUP INC",
        "form_type": "485BPOS",
        "idx_filename": "0001133228-24-004879.txt",  # not important
        "tags": [idx_tag],
    }
    execute_insertmany("master_idx_sample", [test_filing], create_table=True)

    # TODO: add more tests for CLI parameters
    with unittest.mock.patch("sleuth.__main__.process_filing", ret_val=True) as the_mock:
        runner = CliRunner()
        result = runner.invoke(
            main,
            shlex.split(f"chunk --tag={idx_tag}"),
        )

        assert result.exit_code == 0
        assert the_mock.call_count == 1

        _, kwargs = the_mock.call_args
        assert kwargs["actions"] == ["chunk"]
        assert kwargs["accession_number"] == "0001133228-24-004879"


def test_save_master_idx(clean_db):
    assert (
        save_master_idx(
            output_table_name="master_idx",
            year=2020,
            quarter=1,
            form_type_filter="485BPOS",
        )
        == 1824
    )
