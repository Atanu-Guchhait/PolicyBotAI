import logging

import sys

def setup_logger(name: str = "policybot"):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logger(__name__)