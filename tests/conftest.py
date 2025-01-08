from pathlib import Path

import config

cache_path = str(Path(__file__).parent / "data/cache")
config.setv("cache_path", cache_path)
