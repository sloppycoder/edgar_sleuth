import logging

import config

logger = logging.getLogger(__name__)


def test_logger():
    logger.info("running test_logger")
    assert config.cache_dir is not None
