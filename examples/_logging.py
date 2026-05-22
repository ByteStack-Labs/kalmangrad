"""Shared logging helper for the worked examples.

The examples instrument their runs with the standard library logging module
rather than scattered print statements. This mirrors how estimation code is
monitored in practice: progress and results are both shown in the terminal and
recorded to disk for later inspection.

The setup_logger function configures a logger with two handlers. A StreamHandler
writes to stdout so a reader running the example sees output immediately. A
FileHandler writes to logs/<name>.log so the run is recorded. The logs directory
is gitignored, so these records are never committed. It is created at runtime if
it does not exist.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# The logs directory lives at the repository root, one level above examples.
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Timestamp, level, and message. The same format is used by both handlers.
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both stdout and logs/<name>.log.

    The logger is set to INFO level. It carries a StreamHandler for the console
    and a FileHandler for the log file, both using LOG_FORMAT. Existing handlers
    are cleared first so that repeated calls within one process do not duplicate
    output.

    Args:
        name: Identifier for the logger and the log file stem.

    Returns:
        A configured logging.Logger.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
