"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Config module for task module.

This module provides the configuration interface for the task module.
"""

from __future__ import annotations

from dataclasses import dataclass

from simple_parsing.helpers.serialization.serializable import (
    FrozenSerializable,
)


@dataclass(frozen=True)
class BaseSplitConfig(FrozenSerializable):
    """
    Configuration for a dataset split.
    """

    name: str  # split name
    file_regex: str  # regex to match files in the split


@dataclass(frozen=True)
class BaseDatasetConfig(FrozenSerializable):
    """
    Configuration for a dataset.
    """

    name: str  # name of the datasets
    description: str  # description of the dataset format
    # local path or huggingface repo id of the dataset
    # local path should be relative to the REPO_ROOT
    data_path: str
    # indicates if dataset files are stored locally. If true, data_path must point to a valid filesystem directory
    is_local: bool = False
    train_split: BaseSplitConfig | None = None
    valid_split: BaseSplitConfig | None = None
    test_split: BaseSplitConfig | None = None
