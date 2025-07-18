"""
Copyright (c) Meta Platforms, Inc. and affiliates.

CLI Arguments for creating agent.

This module defines the necessary arguments that must be
defined when running MLGym as a CLI tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym.agent.config import AgentConfig

if TYPE_CHECKING:
    from mlgym.backend.config import ModelConfig


@dataclass(frozen=True)
class AgentArguments(FlattenedAccess, FrozenSerializable):
    """Configure the agent's behaviour (templates, parse functions, ...)."""

    model: ModelConfig

    # Policy can only be set via config yaml file from command line
    agent_config_path: Path | str | None = None
    # FIXME: Migrate to pydantic under issue #19
    config: AgentConfig | None = field(default=None, cmd=False)  # noqa: RUF009
    log_verbose_to_console: bool = False

    def __post_init__(self) -> None:
        if self.config is None and self.agent_config_path is not None:
            # If unassigned, we load the config from the file to store its contents with the overall arguments
            config = AgentConfig.load_yaml(self.agent_config_path)
            object.__setattr__(self, "config", config)
        assert self.config is not None
