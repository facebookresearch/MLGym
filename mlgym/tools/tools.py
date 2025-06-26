"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Core tools module for MLGym framework.

This module provides tools for managing commands, environment variables, and
command execution in the MLGym environment. It handles command parsing,
blocklist enforcement, and multi-line command processing.

Adapted from SWE-agent/sweagent/tools/tools.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# Simple Parsing needs this import during runtime
from pathlib import Path  # noqa: TC003

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym.tools.commands import Command
from mlgym.tools.parsing import ParseCommand
from mlgym.utils.config import convert_paths_to_abspath


# FIXME: Fix as part of issue #19.
@dataclass(frozen=True)
class ToolsConfig(FlattenedAccess, FrozenSerializable):
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


class ToolHandler:
    """
    Handles command parsing and management in the MLGym environment.

    Manages command parsing, validation, and documentation. Handles special cases
    like multi-line commands and blocked commands. Maintains environment
    variables and command patterns.
    """

    def __init__(self, tools: ToolsConfig) -> None:
        """
        Initialize the tool handler.

        Args:
            tools (ToolsConfig): Configuration for tools and commands
        """
        self.config = tools
        assert isinstance(self.config, ToolsConfig)

        self.command_docs = self.config.command_docs

        self.submit_command = self.config.submit_command
        self.commands = self.config.commands
        self.util_functions = self.config.util_functions

        for command in self.commands:
            if command.name == self.submit_command:
                self.submit_command_end_name = command.end_name
                break
        self.env_variables = self.config.env_variables
        self.command_patterns = self._parse_command_patterns()

    @property
    def state_command(self) -> Command:
        """
        Get the state command.

        Returns:
            Command: Command for retrieving environment state

        Raises:
            AssertionError: If state command is not configured
        """
        assert self.config.state_command is not None
        return self.config.state_command

    def _parse_command_patterns(self) -> dict[str, re.Pattern[str]]:
        """
        Parse command patterns for all registered commands.

        Creates regex patterns for matching both single-line and multi-line
        commands. Handles special case for submit command.

        Returns:
            dict: Dictionary mapping command names to compiled regex patterns
        """
        command_patterns: dict[str, re.Pattern[str]] = defaultdict(re.Pattern[str])
        for command in self.commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$",
                    re.DOTALL | re.MULTILINE,
                )
                command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                command_patterns[command.name] = pat
        if hasattr(self, "submit_command_end_name"):
            submit_pat = re.compile(
                rf"^\s*({self.submit_command})\s*(.*?)^({self.submit_command_end_name})\s*$",
                re.DOTALL | re.MULTILINE,
            )
        else:
            submit_pat = re.compile(rf"^\s*({self.submit_command})(\s*)$", re.MULTILINE)  # group 2 is nothing
        command_patterns[self.submit_command] = submit_pat
        return command_patterns

    def _get_first_match(self, action: str) -> re.Match | None:
        """
        Find first matching command pattern in action string.

        Args:
            action (str): Action string to search

        Returns:
            re.Match | None: First matching pattern or None if no match found
        """
        assert self.config is not None
        patterns = {
            k: v
            for k, v in self.command_patterns.items()
            if k in self.config.multi_line_command_endings or k == self.config.submit_command
        }

        matches = []
        for _, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]

    def guard_multiline_input(self, action: str) -> str:
        """
        Process multi-line command input.

        Adds heredoc syntax to multi-line commands that need it. Handles
        commands that span multiple lines and are terminated by specific
        end markers.

        Args:
            action (str): Command string to process

        Returns:
            str: Processed command string with proper heredoc syntax
        """
        parsed_action = []
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action)
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start() : first_match.end()]
                rem_action = rem_action[first_match.end() :]
                if pre_action.strip():
                    parsed_action.append(pre_action)
                if match_action.strip():
                    eof = first_match.group(3).strip()
                    if not match_action.split("\n")[0].strip().endswith(f"<< '{eof}'"):
                        guarded_command = match_action[first_match.start() :]
                        first_line = guarded_command.split("\n")[0]
                        # add a space before the << for insert command bad file descriptor error
                        guarded_command = guarded_command.replace(first_line, first_line + f" << '{eof}'", 1)
                        parsed_action.append(guarded_command)
                    else:
                        parsed_action.append(match_action)
            else:
                parsed_action.append(rem_action)
                rem_action = ""
        return "\n".join(parsed_action)

    def should_block_action(self, action: str) -> bool:
        """
        Check if an action should be blocked.

        Checks action against blocklist and standalone blocklist.
        Prevents execution of disallowed commands.

        Args:
            action (str): Action to check

        Returns:
            bool: True if action should be blocked, False otherwise
        """
        assert self.config is not None
        names = action.strip().split()[0]
        if len(names) == 0:
            return False
        name = names[0]
        if name in self.config.blocklist:
            return True
        return bool(name in self.config.blocklist_standalone and name == action.strip())
