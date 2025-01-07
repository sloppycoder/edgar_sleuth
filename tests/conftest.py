import asyncio
import logging
import os

import pytest

# suppress INFO logs to reduce noise in test output
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARN)

