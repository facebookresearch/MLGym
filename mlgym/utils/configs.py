"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Utility functions for configuration management.

Adapted from SWE-agent/sweagent/utils/config.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from mlgym import REPO_ROOT
from mlgym.utils.log import get_logger

logger = get_logger("config")


def convert_path_to_abspath(path: Path | str) -> Path:
    """If path is not absolute, convert it to an absolute path
    using the MLGYM_CONFIG_ROOT environment variable (if set) or
    CONFIG_DIR as base.
    """
    path = Path(path)
    root = REPO_ROOT
    assert root.is_dir()
    if not path.is_absolute():
        path = root / path
    assert path.is_absolute()
    return path.resolve()


def convert_paths_to_abspath(paths: list[Path | str]) -> list[Path]:
    return [convert_path_to_abspath(p) for p in paths]


def load_environment_variables(path: Path | None = None) -> None:
    """Load environment variables from a .env file.
    If path is not provided, we first look for a .env file in the current working
    directory and then in the repository root.
    """
    if path is None:
        cwd_path = Path.cwd() / ".env"
        repo_path = REPO_ROOT / ".env"
        if cwd_path.exists():
            path = cwd_path
        elif repo_path.exists():
            path = repo_path
        else:
            logger.debug("No .env file found")
            return
    if not path.is_file():
        msg = f"No .env file found at {path}"
        raise FileNotFoundError(msg)
    loaded = load_dotenv(dotenv_path=path)
    if loaded:
        logger.info(f"Loaded environment variables from {path}")
