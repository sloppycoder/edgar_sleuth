import json
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from sleuth.__main__ import main
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
    text_table_name = "filing_text_chunks"
    embedding_table_name = "filing_chunks_embeddings"
    search_phrase_table_name = "search_phrase_embeddings"
    dimension = 256
    tag = "pytest"
    search_tag = f"pytest_gemini_{dimension}"
    model = "gemini-1.5-flash-002"

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    form_type = "485BPOS"

    create_search_phrase_embeddings(
        search_phrase_table_name,
        model=GEMINI_EMBEDDING_MODEL,
        tag=search_tag,
        dimension=dimension,
    )

    result = process_filing(
        cik=cik,
        accession_number=accession_number,
        actions=["chunk", "embedding", "search_phrase", "extract"],
        tags=[tag],
        text_table_name=text_table_name,
        embedding_table_name=embedding_table_name,
        search_phrase_table_name=search_phrase_table_name,
        dimension=dimension,
        form_type=form_type,
        model=model,
        search_tag=search_tag,
    )

    assert result
    assert result["n_chunks"] == 271
    assert result["n_embeddings"] == result["n_chunks"]
    assert (
        result["response"] and result["comp_info"] and "trustees" in result["comp_info"]
    )


def test_sleuth_cli():
    with patch("sleuth.__main__.process_filing") as mock_process_filing:
        mock_process_filing.return_value = None

        key = json.dumps({"cik": "1002427", "accession_number": "0001133228-24-004879"})

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["chunk", f"--batch={key}", "--tags=pytest"],
        )

        assert result.exit_code == 0
        assert mock_process_filing.call_count == 1

        _, kwargs = mock_process_filing.call_args
        assert kwargs["actions"] == ["chunk"]
        assert kwargs["accession_number"] == "0001133228-24-004879"
