import json
import logging
import logging.config
import multiprocessing
import sys
from fnmatch import fnmatch
from logging.handlers import QueueListener
from pathlib import Path
from typing import Iterator

import click
import yaml

from .datastore import execute_insertmany, execute_query
from .edgar import read_master_idx
from .llm.embedding import GEMINI_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL
from .processor import (
    gather_extractin_result,
    init_worker,
    process_filing,
    process_filing_wrapper,
)
from .trustee import create_search_phrase_embeddings

MAX_ERRORS = 5

logger = logging.getLogger(__name__)

logger_config_path = Path.cwd() / "logger_config.yaml"
if logger_config_path.exists():
    with open(logger_config_path, "r") as f:
        logging.config.dictConfig(yaml.safe_load(f))


def save_master_idx(
    year: int,
    quarter: int,
    form_type_filter: str,
    output_table_name: str,
) -> int | None:
    rows = read_master_idx(year, quarter, form_type_filter)
    if len(rows) == 0:
        logger.error(f"No records found for {year} Q{quarter} {form_type_filter}")
        return 0

    if execute_insertmany(output_table_name, rows, create_table=True):
        return len(rows)

    logger.error(f"Failed to save master idx for {year} Q{quarter}")
    return None


def enumerate_filings(
    tag: str,
    batch_limit: int,
    index_table_name: str,
) -> Iterator[tuple[str, str]]:
    rows = execute_query(
        f"""
        SELECT distinct cik, accession_number FROM {index_table_name}
        WHERE %s = ANY(tags)
        """,
        (tag,),
    )
    n_processed = 0
    for row in rows:
        if batch_limit and n_processed >= batch_limit:
            break

        n_processed += 1
        yield row["cik"], row["accession_number"]


@click.command()
@click.argument(
    "action",
    type=click.Choice(
        [
            "load-index",
            "init-search-phrases",
            "chunk",
            "embedding",
            "extract",
            "export",
        ],
        case_sensitive=False,
    ),
)
@click.argument(
    "index-range",
    required=False,
)
@click.option(
    "--output",
    required=False,
    help="Output file for export action",
)
@click.option(
    "--batch-limit",
    type=int,
    default=0,
    help="Number of records to process in batch mode, 0 means unlimited",
)
@click.option(
    "--model",
    type=click.Choice(["gpt", "gemini"]),
    default="gemini",
    help="Model to use for processing",
)
@click.option(
    "--table",
    "tables",
    required=False,
    multiple=True,
    help="Specify table names for overriding default ones. e.g. idx=master_idx_new",
)
@click.option(
    "--tag",
    required=False,
    help="tags used to query input",
)
@click.option(
    "--result-tag",
    required=False,
    help="tags used to save extraction result",
)
@click.option(
    "--dimension",
    type=int,
    default=768,
    help="Dimensionality of embeddings. Only applicable when using Gemini API",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    help="Number of works to use for processing. Each worker will process 1 filing at a time",  # noqa: E501
)
# ruff: noqa: C901
def main(
    action: str,
    tag: str,
    result_tag: str,
    tables: list[str],
    model: str,
    dimension: int,
    batch_limit: int,
    workers: int,
    index_range: str,
    output: str,
) -> None:
    if action not in ["load-index"] and not tag:
        raise click.UsageError(f"--tag is required for {action}")

    if action in ["extract", "export"] and not result_tag:
        raise click.UsageError(f"--result-tag is required for {action}")

    if action == "load-index" and not index_range:
        raise click.UsageError(f"index-range is required for {action}")

    form_type = "485BPOS"

    # default table names
    tables_map = {
        "full-idx": "master_idx",
        "idx": "master_idx_sample",
        "text": "filing_text_chunks",
        "embedding": "filing_chunks_embeddings",
        "result": "trustee_comp_results",
        "search": "search_phrase_embeddings",
    }

    # use command line options to override table names
    for table in tables:
        key, value = table.split("=")
        if key in tables_map:
            tables_map[key] = value

    if action != "extract":
        model = GEMINI_EMBEDDING_MODEL if model == "gemini" else OPENAI_EMBEDDING_MODEL
    else:
        model = "gemini-1.5-flash-002" if model == "gemini" else "gpt-4o-mini"

    if action == "load-index":
        for year in range(1995, 2025):
            for quarter in range(1, 5):
                if fnmatch(f"{year}/{quarter}", index_range):
                    n_count = save_master_idx(
                        output_table_name=tables_map["full-idx"],
                        year=year,
                        quarter=quarter,
                        form_type_filter=form_type,
                    )
                    if n_count:
                        print(f"Saved {n_count} records for {year}-QTR{quarter}")
                    else:
                        print(f"No records found for {year}-QTR{quarter}")
        return

    if action == "init-search-phrases":
        print("Initializing search phrase embeddings...")
        create_search_phrase_embeddings(
            tables_map["search"],
            model=GEMINI_EMBEDDING_MODEL,
            tag=tag,
            dimension=dimension,
        )
        return

    if action == "export":
        # TODO: remove hard coded table names
        result = gather_extractin_result(
            idx_table_name=tables_map["idx"],
            extraction_result_table_name=tables_map["result"],
            idx_tag=tag,
            result_tag=result_tag,
        )
        with open(output, "w") as f:
            for row in result:
                jsonl = json.dumps(row)
                f.write(jsonl)
                f.write("\n")
        print(f"Exported {len(result)} records to {output}")
        return

    print(f"Running {action} with tag {tag}")

    # list of arguments to pass to process_filing
    args = [
        {
            "action": action,
            "tables_map": tables_map,
            "cik": cik,
            "accession_number": accession_number,
            "idx_tag": tag,
            "result_tag": result_tag,
            "model": model,
            "dimension": dimension,
            "form_type": form_type,
        }
        for cik, accession_number in list(
            enumerate_filings(
                tag=tag, batch_limit=batch_limit, index_table_name=tables_map["idx"]
            )
        )
    ]

    if workers == 1:
        for arg in args:
            process_filing(**arg)
    else:
        # create a queue to receive log messages from worker processes
        # https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
        logging_q = multiprocessing.Queue()
        handler = logging.getHandlerByName("console")  # defined in logger_config.yaml
        q_listener = QueueListener(logging_q, handler)  # pyright: ignore
        q_listener.start()

        try:
            with multiprocessing.Pool(
                workers,
                init_worker,
                (
                    logging_q,
                    logging.DEBUG,
                ),
            ) as pool:
                pool.map(process_filing_wrapper, args)

            q_listener.stop()
        finally:
            if logging_q:
                logging_q.close()
                logging_q.join_thread()


if __name__ == "__main__":
    # the database connection is a singleton within this application.
    # on linux, multiprocessing uses fork() to create new processes
    # so the same connection is shared across all processes.
    # this will lead to funny error message like
    # "Error prepared statement "_pg3_0" already exists ...
    # the statement below force the use of spawn() method to create
    # new processes on Linux in order to avoid this issue
    # spawn() is default on macOS and Windows anyways.
    multiprocessing.set_start_method("spawn")
    main(sys.argv[1:])
