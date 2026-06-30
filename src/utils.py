"""Utilities set."""

import logging


def init_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """The logger initialization function."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger