from __future__ import annotations

from pathlib import Path

__version__ = "0.1.1"

PACKAGE_DIR = Path(__file__).resolve().parent
assert PACKAGE_DIR.is_dir()
REPO_ROOT = PACKAGE_DIR.parent
assert REPO_ROOT.is_dir()
CONFIG_DIR = REPO_ROOT / "configs"
assert CONFIG_DIR.is_dir()


__all__ = [
    "CONFIG_DIR",
    "PACKAGE_DIR",
]
