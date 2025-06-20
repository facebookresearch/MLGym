"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Base model implementation for the MLGym framework.

This module provides the core model functionality including configuration,
API interaction, and cost tracking. It defines the base classes and interfaces
for different model types (OpenAI, Azure, Meta, etc.) and handles common
operations like cost calculation and limit enforcement.

Adapted from SWE-agent/sweagent/agent/models.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.serialization.serializable import (
    FrozenSerializable,
    Serializable,
)

from mlgym.exceptions import CostLimitExceededError
from mlgym.utils.log import get_logger

if TYPE_CHECKING:
    from mlgym.types import HistoryItem


@dataclass(frozen=True)
class ModelArguments(FrozenSerializable):
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


@dataclass
class APIStats(Serializable):
    total_cost: float = 0.0
    task_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0

    def __add__(self, other: APIStats) -> APIStats:
        """
        Add two APIStats objects together.

        Args:
            other (APIStats): Another APIStats object to add

        Returns:
            APIStats: New APIStats with summed values

        Raises:
            TypeError: If other is not an APIStats object
        """

        return APIStats(
            **{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)},
        )

    def replace(self, other: APIStats) -> APIStats:
        """
        Replace current stats with values from another APIStats object.

        Args:
            other (APIStats): APIStats object to copy values from

        Returns:
            APIStats: New APIStats with values from other

        Raises:
            TypeError: If other is not an APIStats object
        """

        return APIStats(**{field.name: getattr(other, field.name) for field in fields(self)})


class BaseModel(ABC):
    """
    Base class for all model implementations.

    Provides common functionality for model interaction, cost tracking,
    and limit enforcement. Specific model implementations (OpenAI, Azure, etc.)
    should inherit from this class.

    Attributes:
        MODELS (dict): Registry of supported models and their metadata
        SHORTCUTS (dict): Mapping of model name aliases to actual names
        args (ModelArguments): Configuration arguments for the model
        model_metadata (dict): Metadata for the current model
        logger: Logger instance for the model
        stats (APIStats): Statistics tracking for API usage
        model_provider (str): Name of the model provider
        api_model (str): API-compatible name for the model
    """

    MODELS: ClassVar = {}
    SHORTCUTS: ClassVar = {}

    def __init__(self, args: ModelArguments) -> None:
        """
        Initialize the model with configuration arguments.

        Args:
            args (ModelArguments): Configuration for the model
        """
        self.args = args
        self.model_metadata = {}
        self.logger = get_logger("lm-model")
        self.stats = APIStats()
        self.model_provider = "Meta"

        # Map `model_name` to API-compatible name `api_model`
        self.api_model = self.SHORTCUTS.get(self.args.model_name, self.args.model_name)

        # Map model name to metadata (cost, context info)
        # FIXME: Find a better way to handle custom models and shortcuts.
        MODELS = {  # noqa: N806
            **{dest: self.MODELS[src] for dest, src in self.SHORTCUTS.items()},
            **self.MODELS,
        }
        if args.model_name in MODELS:
            self.model_metadata = MODELS[args.model_name]
        elif args.model_name.startswith("meta:"):
            self.api_model = args.model_name.split("meta:", 1)[1]
            self.model_metadata = MODELS[self.api_model]
        elif args.model_name.startswith("litellm:"):
            # do nothing if it's a litellm model
            self.model_provider = "litellm"
        elif args.model_name.startswith("avior:"):
            self.api_model = args.model_name.split("avior:", 1)[1]
            self.model_metadata = MODELS.get(self.api_model)  # type: ignore
            self.model_provider = "avior"
        else:
            msg = f"Unregistered model ({args.model_name}). Add model name to MODELS metadata to {self.__class__}"
            raise ValueError(msg)

    def reset_stats(self, other: APIStats | None = None) -> None:
        """
        Reset or replace the current API statistics.

        Args:
            other (APIStats | None): If provided, replace stats with these values.
                If None, reset to initial state keeping total_cost. Defaults to None
        """
        if other is None:
            self.stats = APIStats(total_cost=self.stats.total_cost)
            self.logger.info("Resetting model stats")
        else:
            self.stats = other

    @abstractmethod
    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0.0) -> float:
        """
        Update API statistics with new usage information.

        Args:
            input_tokens (int): Number of tokens in the prompt
            output_tokens (int): Number of tokens in the response
            cost (float): Cost of the API call. Defaults to 0.0

        Returns:
            float: The calculated cost of the API call

        Raises:
            CostLimitExceededError: If total_cost_limit or per_instance_cost_limit is exceeded
        """
        # Calculate cost and update cost related fields
        if self.model_provider == "Meta" and self.model_metadata is not None:
            cost = (
                self.model_metadata["cost_per_input_token"] * input_tokens
                + self.model_metadata["cost_per_output_token"] * output_tokens
            )

        self.stats.total_cost += cost
        self.stats.task_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        # Log updated cost values to std. out.
        self.logger.info(
            f"input_tokens={input_tokens:,}, "
            f"output_tokens={output_tokens:,}, "
            f"instance_cost={self.stats.task_cost:.2f}, "
            f"cost={cost:.2f}",
        )
        self.logger.info(
            f"total_tokens_sent={self.stats.tokens_sent:,}, "
            f"total_tokens_received={self.stats.tokens_received:,}, "
            f"total_cost={self.stats.total_cost:.2f}, "
            f"total_api_calls={self.stats.api_calls:,}",
        )

        # Check whether total cost or instance cost limits have been exceeded
        if 0 < self.args.total_cost_limit <= self.stats.total_cost:
            self.logger.warning(f"Cost {self.stats.total_cost:.2f} exceeds limit {self.args.total_cost_limit:.2f}")
            msg = "Total cost limit exceeded"
            raise CostLimitExceededError(msg)

        if 0 < self.args.per_instance_cost_limit <= self.stats.task_cost:
            self.logger.warning(
                f"Cost {self.stats.task_cost:.2f} exceeds limit {self.args.per_instance_cost_limit:.2f}"
            )
            msg = "Instance cost limit exceeded"
            raise CostLimitExceededError(msg)
        return cost

    @abstractmethod
    def query(self, history: list[HistoryItem]) -> str:
        """
        Query the model with a conversation history.

        Args:
            history (list[dict[str, str]]): List of conversation turns

        Returns:
            str: Model's response

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        msg = "Use a subclass of BaseModel"
        raise NotImplementedError(msg)

    @abstractmethod
    def history_to_messages(
        self, history: list[HistoryItem], is_demonstration: bool = False
    ) -> str | list[dict[str, str]]:
        """
        Convert history to messages.

        Args:
            history (list[HistoryItem]): History to convert
            is_demonstration (bool): Whether the history is a demonstration

        Returns:
            str | list[dict[str, str]]: Messages
        """

        # Remove system messages if it is a demonstration
        if is_demonstration:
            history = [entry for entry in history if entry["role"] != "system"]
            return "\n".join([entry.get("content") or "" for entry in history])

        # Return history components with just role, content fields
        messages = []
        for entry in history:
            message = {}
            for key in ["role", "content"]:
                message[key] = str(entry.get(key) or "")
            messages.append(message)

        return messages
