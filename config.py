"""
config

This module provides a configuration holder for the application. Each key can be
overriden by setting an environment variable with the same name in uppercase.

    import config
    config.database_id

Use a key before initializing will get a warning message

    import config
    config.database_id

    WARN: Config key database_id used before being set

"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConfigHolder:
    _env_prefix: str = ""
    cache_path: str = "cache"
    log_level: str = "DEBUG"
    database_url: str = ""

    def __init__(self):
        """Initialize the configuration holder with environment variables"""
        for attr_ in dir(self):
            if not attr_.startswith("__"):
                val = os.environ.get(f"{self._env_prefix}{attr_}".upper())
                if val:
                    setattr(self, attr_, val)


_config_ = ConfigHolder()


def __getattr__(key: str) -> str:
    val = getattr(_config_, key, "")
    if val:
        return val
    else:
        print(f"WARN: Config key {key} used before being set")
        return ""


# use a funciton because overriding __setattr__ is not allowed
def setv(key: str, value: str) -> None:
    attrs = [f for f in dir(_config_) if not f.startswith("__")]
    if key in attrs:
        setattr(_config_, key, str(value))
    else:
        raise RuntimeError(f"Config key {key} is not allowed")
