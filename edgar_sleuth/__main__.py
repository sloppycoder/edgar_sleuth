import json
import logging
import logging.config
import sys
from pathlib import Path

import click

from edgar import SECFiling

from . import chunk_filing, get_embeddings

logger = logging.getLogger(__name__)

logger_config_path = Path.cwd() / "logger_config.json"
if logger_config_path.exists():
    with open(logger_config_path, "r") as f:
        logging.config.dictConfig(json.load(f))


def create_filing(
    cik: str = "", accession_number: str = "", idx_filename: str = ""
) -> SECFiling:
    if idx_filename:
        return SECFiling(idx_filename=idx_filename)
    else:
        return SECFiling(cik=cik, accession_number=accession_number)


@click.command()
@click.argument("action")
@click.option("--dryrun", is_flag=True, default=False)
@click.option("--filename", required=False)
@click.option("--cik", required=False)
@click.option("--accession_number", required=False)
@click.option("--tags", required=False, dest="tag_str")
@click.option("--form", required=False, default="485BPOS", dest="form_type")
def main(
    action: str,
    dryrun: bool,
    filename: str,
    cik: str,
    accession_number: str,
    tag_str: str,
    form_type: str,
) -> None:
    tags = tag_str.split(",") if tag_str else []

    if action == "chunk":
        filing = create_filing(
            cik=cik,
            accession_number=accession_number,
            idx_filename=filename,
        )
        n_chunks = chunk_filing(
            filing=filing,
            form_type=form_type,
            dryrun=dryrun,
            tags=tags,
        )
        if n_chunks > 1:
            print(
                f"{filing} {form_type} splitted into {n_chunks} chunks"  # noqa E501
            )
        else:
            print(f"Error when splitting {filing} {form_type}")

    elif action == "embedding":
        filing = create_filing(
            cik=cik,
            accession_number=accession_number,
            idx_filename=filename,
        )
        n_chunks = get_embeddings(
            cik=filing.cik,
            accession_number=filing.accession_number,
            dryrun=dryrun,
            tags=tags,
        )
        if n_chunks > 1:
            print(f"Saved {n_chunks} embeddings for {filing} {form_type}")
        else:
            print(f"Error when get embeddings for {filing} {form_type}")
    else:
        print("not yet implemented")


if __name__ == "__main__":
    main(sys.argv[1:])
