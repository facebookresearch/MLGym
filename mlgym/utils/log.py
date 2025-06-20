"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Logging utility for MLGym.

Adapted from SWE-agent/sweagent/utils/log.py
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from rich.logging import RichHandler

if TYPE_CHECKING:
    from pathlib import PurePath

_SET_UP_LOGGERS: set[str] = set()
_ADDITIONAL_HANDLERS: list[logging.Handler] = []

# FIXME: This is a hack to add a TRACE level to the logging module. We need to come up with a better logging system in general.
logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore

# FIXME: The whole logging system is a mess and needs to be refactored.


def _interpret_level_from_env(level: str | None, *, default: int = logging.DEBUG) -> int:
    if not level:
        return default
    if level.isnumeric():
        return int(level)
    return int(getattr(logging, level.upper()))


_STREAM_LEVEL = _interpret_level_from_env(os.environ.get("MLGYM_LOG_STREAM_LEVEL"))
_FILE_LEVEL = _interpret_level_from_env(os.environ.get("MLGYM_LOG_FILE_LEVEL"), default=logging.TRACE)  # type: ignore


def get_logger(name: str) -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    print(f"Setting up logger for {name=}")
    logger = logging.getLogger(name)
    if name in _SET_UP_LOGGERS:
        # Already set up
        return logger
    handler = RichHandler(
        show_time=bool(os.environ.get("MLGYM_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    logger.setLevel(min(_STREAM_LEVEL, _FILE_LEVEL))
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    for handler in _ADDITIONAL_HANDLERS:  # type: ignore
        print(f"Registering {handler.baseFilename} to logger {name=}")  # type: ignore
        logger.addHandler(handler)
    return logger


def add_file_handler(path: PurePath | str, logger_names: list[str] | None = None) -> None:
    """Adds a file handler to all loggers that we have set up
    and all future loggers that will be set up with `get_logger`.
    """
    print(f"Adding file_handler for {path=}")
    handler = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(_FILE_LEVEL)
    if logger_names is None:
        for name in _SET_UP_LOGGERS:
            logger = logging.getLogger(name)
            print(f"Registering {path=} to logger {name=}")
            logger.addHandler(handler)
    else:
        for name in logger_names:
            logger = logging.getLogger(name)
            print(f"Registering {path=} to logger {name=}")
            logger.addHandler(handler)
    _ADDITIONAL_HANDLERS.append(handler)


default_logger = get_logger("MLGym")
