"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Base model implementation for the MLGym framework.

This module provides the core model functionality including configuration,
API interaction, and cost tracking. It defines the base classes and interfaces
for different model types (OpenAI, Azure, Meta, etc.) and handles common
operations like cost calculation and limit enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.serialization.serializable import (
    FrozenSerializable,
)

@dataclass(frozen=True)
class ModelConfig(FrozenSerializable):
    """Arguments configuring the model and it's behavior."""

    # Name of the model to use
    model_name: str
    # Cost limit for every task
    per_instance_cost_limit: float = 0.0
    # Total cost limit
    total_cost_limit: float = 0.0
    # Sampling temperature
    temperature: float = 1.0
    # Sampling top_p
    top_p: float = 1.0
    # Path to replay file when using the replay model
    replay_path: str | None = None
    # api base url
    host_url: str | None = None
    # api version - specific to azure
    api_version: str | None = None
    # api key
    api_key: str | None = None
    # custom stop sequences
    stop: list[str] = field(default_factory=list)  # noqa: RUF009
    # additional kwargs to pass to litellm.completion
    completion_kwargs: dict[str, Any] = field(default_factory=dict)  # noqa: RUF009
