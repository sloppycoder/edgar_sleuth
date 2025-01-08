import json
import logging
import logging.config
import sys
from pathlib import Path

import click

from .splitter import chunk_text

logger = logging.getLogger(__name__)

logger_config_path = Path.cwd() / "logger_config.json"
if logger_config_path.exists():
    with open(logger_config_path, "r") as f:
        logging.config.dictConfig(json.load(f))


@click.command()
@click.argument("action")
@click.option("--dryrun", is_flag=True)
@click.option("--filename", required=False)
@click.option("--cik", required=False)
@click.option("--accession_number", required=False)
def main(
    action: str, dryrun: bool, filename: str, cik: str, accession_number: str
) -> None:
    logger.debug(f"Running action {action} with dryrun={dryrun}")

    if action == "chunk":
        chunks = chunk_text(cik, filename)
        print(f"{len(chunks)} chunks created")
    else:
        print("not yet implemented")


if __name__ == "__main__":
    main(sys.argv[1:])
