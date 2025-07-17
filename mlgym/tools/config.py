"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Config module for tools module.

This module provides the configuration interface for the tools module.
"""

from __future__ import annotations

from dataclasses import dataclass

# Simple Parsing needs this import during runtime
from pathlib import Path  # noqa: TC003

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym.tools.commands import Command
from mlgym.tools.parsing import ParseCommand
from mlgym.utils.config import convert_paths_to_abspath

if TYPE_CHECKING:
    from pathlib import Path

# FIXME: Fix as part of issue #19.
@dataclass(frozen=True)
class BaseToolsConfig(FlattenedAccess, FrozenSerializable):
    command_files: list[str | Path] = field(default_factory=list)  # noqa: RUF009
    env_variables: dict[str, str] = field(default_factory=dict)  # noqa: RUF009
    util_functions: list[str] = field(default_factory=list)  # noqa: RUF009
    submit_command: str = "submit"
    parser: str | ParseCommand = "ParseCommandBash"
    state_command: Command = Command(  # noqa: RUF009
        name="state",
        code="""state() {
            echo '{"working_dir": "'$(realpath --relative-to=$ROOT/.. $PWD)'"}';
        };""",
    )
    blocklist_error_template: str = "Interactive operation '{name}' is not supported by this environment"
    blocklist: tuple[str, ...] = (
        "vim",
        "vi",
        "emacs",
        "nano",
        "nohup",
        "git",
    )
    blocklist_standalone: tuple[str, ...] = (
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "exit",
        "/bin/bash",
        "/bin/sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
        "su",
    )
    commands: list[Command] = field(default_factory=list)  # noqa: RUF009

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_files", convert_paths_to_abspath(self.command_files))
        if isinstance(self.parser, str):
            object.__setattr__(self, "parser", ParseCommand.get(self.parser))
        assert isinstance(self.parser, ParseCommand)

        for file in self.command_files:
            commands = self.parser.parse_command_file(str(file))

            util_functions = [command for command in commands if command.name.startswith("_")]
            commands = [command for command in commands if not command.name.startswith("_")]

            object.__setattr__(self, "util_functions", self.util_functions + util_functions)
            object.__setattr__(self, "commands", self.commands + commands)

        multi_line_command_endings = {
            command.name: command.end_name for command in self.commands if command.end_name is not None
        }
        object.__setattr__(self, "multi_line_command_endings", multi_line_command_endings)
        command_docs = self.parser.generate_command_docs(
            self.commands,
            **self.env_variables,
        )
        object.__setattr__(self, "command_docs", command_docs)
