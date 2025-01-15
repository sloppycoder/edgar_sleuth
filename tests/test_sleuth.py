import os
import shlex

import pytest
from click.testing import CliRunner

from sleuth.__main__ import main
from sleuth.datastore import execute_insertmany, execute_query
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
    search_tag = "search-pytest-cli-test"
    result_tag = "result-pytest"

    # simple filing
    cik = "1002427"
    accession_number = "0001133228-24-004879"
    form_type = "485BPOS"

    create_search_phrase_embeddings(
        tables_map["search"],
        model=GEMINI_EMBEDDING_MODEL,
        search_tag=search_tag,
        dimension=dimension,
    )

    result = process_filing(
        action="chunk",
        tables_map=tables_map,
        cik=cik,
        accession_number=accession_number,
        idx_tag=idx_tag,
        search_tag=search_tag,
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
        search_tag=search_tag,
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
        search_tag=search_tag,
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
def test_sleuth_cli(tmp_path):
    output_path = tmp_path / "output.jsonl"
    idx_tag = "pytest-cli-test"
    search_tag = "search-pytest-cli-test"
    result_tag = "result-pytest-cli-test"
    test_filing = {
        "cik": "1002427",
        "accession_number": "0001133228-24-004879",
        "date_filed": "2024-01-29",  # not important
        "company_name": "VANGUARD GROUP INC",
        "form_type": "485BPOS",
        "idx_filename": "0001133228-24-004879.txt",  # not important
        "tags": [idx_tag],
    }
    execute_insertmany("master_idx_pytest", [test_filing], create_table=True)

    tables_map = {
        "idx": "master_idx_pytest",
        "text": "filing_text_chunks_pytest",
        "embedding": "filing_chunks_embeddings_pytest",
        "result": "trustee_comp_results_pytest",
        "search": "search_phrase_embeddings_pytest",
    }

    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split(f"""init-search-phrases --search-tag={search_tag} \
            --table search={tables_map["search"]} \
        """),
    )
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split(f"""chunk --tag={idx_tag} \
            --table idx={tables_map['idx']} \
            --table text={tables_map['text']} \
        """),
    )
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split(f"""embedding --tag={idx_tag} \
            --table idx={tables_map['idx']} \
            --table text={tables_map['text']} \
            --table embedding={tables_map['embedding']} \
        """),
    )
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split(f"""extract \
            --tag={idx_tag} --result-tag={result_tag} --search-tag={search_tag}\
            --table idx={tables_map['idx']}    \
            --table text={tables_map['text']}  \
            --table embedding={tables_map['embedding']} \
            --table result={tables_map['result']} \
            --table search={tables_map["search"]} \
            --model gemini \
        """),
    )
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split(f"""export --tag={idx_tag} --result-tag={result_tag} \
            --output={str(output_path)} \
            --table idx={tables_map['idx']}    \
            --table text={tables_map['text']}  \
            --table embedding={tables_map['embedding']} \
            --table result={tables_map['result']} \
        """),
    )
    assert result.exit_code == 0

    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1

    try:
        output_path.unlink()
    except FileNotFoundError:
        pass


def test_load_index(clean_db):
    runner = CliRunner()
    result = runner.invoke(
        main,
        shlex.split('load-index "2020/1" --table full-idx=master_idx_pytest_2'),
    )
    assert result.exit_code == 0

    result = execute_query("SELECT COUNT(*) FROM master_idx_pytest_2")
    assert result and result[0]["count"] == 1824
