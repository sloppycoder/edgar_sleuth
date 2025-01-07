import json
import logging
import logging.config
from pathlib import Path

import config

with open(Path(__file__).parent / "logger_config.json", "r") as f:
    logging.config.dictConfig(json.load(f))

if __name__ == "__main__":
    print(f"cache dir is {config.cache_dir}")
