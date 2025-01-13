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
    tags: list[str] = [],
) -> int:
    rows = read_master_idx(year, quarter, form_type_filter)
    if len(rows) == 0:
        logger.error(f"No records found for {year} Q{quarter} {form_type_filter}")
        return 0

    for row in rows:
        row["tags"] = tags

    if execute_insertmany("master_idx_sample", rows, create_table=True):
        return len(rows)
    else:
        logger.error(f"Failed to save master idx for {year} Q{quarter}")
        return 0


def enumerate_filings(
    tag: str,
    batch_limit: int,
    index_table_name: str = "master_idx_sample",
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
    "-i",
    "--input",
    "input_",
    required=False,
    help="Name of the table or file get read input from",
)
@click.option(
    "--input-tag",
    "input_tag",
    required=False,
    help="tags used to query input",
)
@click.option(
    "-o",
    "--output",
    "output",
    required=False,
    help="Name of the table or file to write processing output to",
)
@click.option(
    "--tags",
    "output_tags_str",
    required=False,
    help="tags associated with output",
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
    model: str,
    batch_limit: int,
    input_: str,
    input_tag: str,
    output: str,
    output_tags_str: str,
    dimension: int,
    workers: int,
) -> None:
    # checking and setting default for parameters
    tags = output_tags_str.split(",") if output_tags_str else []

    if action == "chunk":
        input_ = input_ or "master_idx_sample"
        output = output or "filing_text_chunks"
    elif action == "embedding":
        input_ = input_ or "filing_text_chunks"
        output = output or "filing_chunks_embeddings"
        # always carry the tag on text chunks to embedding
        # this helps to correllate the embeddings back to the original text
        if input_tag and input_tag not in tags:
            tags += [input_tag]
    elif action == "init-search-phrases":
        output = output or "search_phrase_embeddings"
    elif action == "extract":
        input_ = input_ or "filing_chunks_embeddings"
        output = output or "trustee_comp_results"
    elif action == "load-index":
        output = output or "master_idx"

    if action != "extract":
        model = GEMINI_EMBEDDING_MODEL if model == "gemini" else OPENAI_EMBEDDING_MODEL
    else:
        model = "gemini-1.5-flash-002" if model == "gemini" else "gpt-4o-mini"

    if action not in ["load-index", "init-search-phrases"] and not input_tag:
        raise click.UsageError("--input-tag is required")

    if action not in ["export"] and not tags:
        raise click.UsageError("output tags is required")

    if action == "load-index":
        for year in range(1995, 2025):
            for quarter in range(1, 5):
                if fnmatch(f"{year}/{quarter}", input_):
                    n_count = save_master_idx(year, quarter, "485BPOS", tags=tags)
                    if n_count:
                        print(f"Saved {n_count} records for {year}-QTR{quarter}")
                    else:
                        print(f"No records found for {year}-QTR{quarter}")
        return

    if action == "init-search-phrases":
        print("Initializing search phrase embeddings...")
        create_search_phrase_embeddings(
            output,
            model=GEMINI_EMBEDDING_MODEL,
            tags=tags,
            dimension=dimension,
        )
        return

    if action == "export":
        result = gather_extractin_result(
            idx_table_name="master_idx_sample",
            extraction_result_table_name="trustee_comp_results",
            tag=input_tag,
        )
        with open(output, "w") as f:
            for row in result:
                jsonl = json.dumps(row)
                f.write(jsonl)
                f.write("\n")
        print(f"Exported {len(result)} records to {output}")
        return

    print(f"Running {action} with tags {tags} and output to {output}")

    form_type = "485BPOS"
    # TODO: remove hard coded table names
    text_table_name = "filing_text_chunks"
    search_phrase_table_name = "search_phrase_embeddings"

    if workers == 1:
        for cik, accession_number in enumerate_filings(input_tag, batch_limit):
            process_filing(
                action=action,
                dimension=dimension,
                cik=cik,
                accession_number=accession_number,
                input_tag=input_tag,
                output_tags=tags,
                model=model,
                input_table=input_,
                output_table=output,
                form_type=form_type,
                search_phrase_table_name=search_phrase_table_name,
                text_table_name=text_table_name,
            )
    else:
        all_filings = list(enumerate_filings(input_tag, batch_limit))
        args = [
            {
                "action": action,
                "dimension": dimension,
                "cik": cik,
                "accession_number": accession_number,
                "input_table": input_,
                "input_tag": input_tag,
                "output_tags": tags,
                "model": model,
                "output_table": output,
                "form_type": form_type,
                "search_phrase_table_name": search_phrase_table_name,
                "text_table_name": text_table_name,
            }
            for cik, accession_number in all_filings
        ]

        # create a queue to receive log messages from worker processes
        # https://stackoverflow.com/questions/641420/how-should-i-log-while-using-multiprocessing-in-python
        logging_q = multiprocessing.Queue()
        handler = logging.getHandlerByName("console")  # defined in logger_config.yaml
        q_listener = QueueListener(logging_q, handler)  # pyright: ignore
        q_listener.start()

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


if __name__ == "__main__":
    main(sys.argv[1:])
