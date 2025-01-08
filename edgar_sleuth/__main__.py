import json
import logging
import logging.config
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

logger_config_path = Path.cwd() / "logger_config.json"
if logger_config_path.exists():
    with open(logger_config_path, "r") as f:
        logging.config.dictConfig(json.load(f))


@click.command()
@click.argument("action")
@click.option("--dryrun", is_flag=True)
def main(action: str, dryrun: bool) -> None:
    logger.info(f"Running action {action} with dryrun={dryrun}")


if __name__ == "__main__":
    main(sys.argv[1:])
