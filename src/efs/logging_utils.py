"""Logging: one console handler plus one file handler per run.

All project modules log through children of the ``efs`` logger. Inside a
module, use ``logging.getLogger(__name__)`` (its name starts with ``efs.``);
inside a script, use ``get_logger("scripts.<name>")``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER_NAME = "efs"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    console_level: str = "INFO",
    log_file: Path | None = None,
    file_level: str = "DEBUG",
) -> logging.Logger:
    """Configure the project logger and return it."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # The logger itself passes everything; each handler decides what it shows.
    # This lets the file keep DEBUG detail while the console stays readable.
    logger.setLevel(logging.DEBUG)

    # Remove old handlers so calling this twice (e.g. in a notebook) does not
    # print every message twice.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # Do not also pass messages to Python's root logger (avoids duplicates
    # when libraries configure the root logger themselves).
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level.upper())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the project logger, e.g. ``efs.scripts.train``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
