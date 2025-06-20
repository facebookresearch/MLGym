"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Command handling module for MLGym tools.

This module provides classes and utilities for managing and executing commands
in the MLGym environment. It handles command registration, validation, and
execution with proper argument handling.

Adapted from SWE-agent/sweagent/tools/commands.py
"""

from __future__ import annotations

from dataclasses import dataclass

from simple_parsing.helpers.serialization.serializable import FrozenSerializable


@dataclass(frozen=True)
class Command(FrozenSerializable):
    code: str
    name: str
    docstring: str | None = None
    end_name: str | None = None  # if there is an end_name, then it is a multi-line command
    arguments: dict | None = None
    signature: str | None = None
