"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Config module for agent module.

This module provides the configuration interface for the agent module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym.agent.history_processors import HistoryProcessor
from mlgym.agent.parsing import ParseFunction
from mlgym.tools.tools import ToolHandler, ToolsConfig
from mlgym.utils.config import convert_paths_to_abspath

# agent/base.py
@dataclass(frozen=True)
class AgentConfig(FlattenedAccess, FrozenSerializable):
    system_template: str
    task_template: str
    next_step_template: str | None = None  # defaults to task_template
    next_step_no_output_template: str | None = None  # default to next_step template
    strategy_template: str | None = None
    demonstration_template: str | None = None
    # Paths to demonstrations. If path is not absolute, it is assumed to be
    # relaive to the MLGym repository root.
    # FIXME: Migrate to pydantic under issue #19
    demonstrations: list[str | Path] = field(default_factory=list)  # noqa: RUF009
    # if True, add demonstration to history instead of as a single message
    put_demos_in_history: bool = False
    # defaults to format_error_template in ParseFunction
    format_error_template: str | None = None
    # Commands configuration with blocklist, env variables and util functions
    # FIXME: Migrate to pydantic under issue #19
    tools: ToolsConfig = field(default_factory=ToolsConfig)  # noqa: RUF009
    output_parser: str | ParseFunction = "ThoughtActionParser"
    history_processor: str = "DefaultHistoryProcessor"
    # FIXME: Migrate to pydantic under issue #19
    history_processor_args: dict[str, Any] = field(default_factory=dict)  # noqa: RUF009

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools_handler", ToolHandler(self.tools))

        object.__setattr__(self, "demonstrations", convert_paths_to_abspath(self.demonstrations))

        if self.next_step_template is None:
            object.__setattr__(self, "next_step_template", self.task_template)
        if self.next_step_no_output_template is None:
            object.__setattr__(self, "next_step_no_output_template", self.next_step_template)

        if isinstance(self.output_parser, str):
            object.__setattr__(self, "output_parser", ParseFunction.get(self.output_parser))
        assert isinstance(self.output_parser, ParseFunction)

        if self.format_error_template is None:
            object.__setattr__(
                self,
                "format_error_template",
                self.output_parser.format_error_template,
            )
        else:
            object.__setattr__(
                self,
                "format_error_template",
                self.format_error_template.format(**self.__dict__),
            )

        object.__setattr__(
            self,
            "history_processor",
            HistoryProcessor.get(self.history_processor, **self.history_processor_args),
        )

