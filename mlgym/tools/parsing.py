"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Command parsing module for MLGym tools.

This module provides classes for parsing different command formats and generating
their documentation. It supports parsing bash functions, scripts with YAML docstrings,
and detailed command specifications.

Adapted from SWE-agent/sweagent/tools/parsing.py
"""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from mlgym.tools.commands import Command


# TODO: Move to new meta class functionality
class ParseCommandMeta(type):
    """
    Metaclass for command parsers that maintains a registry of parser types.

    Provides automatic registration of parser classes to enable lookup by name.
    All parser classes except the base ParseCommand are added to the registry.

    Attributes:
        _registry (dict): Maps parser names to their implementations
    """

    _registry: ClassVar = {}

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "ParseCommand":
            cls._registry[name] = new_cls
        return new_cls


@dataclass
class ParseCommand(metaclass=ParseCommandMeta):
    """
    Base class for command parsers.

    Defines interface for parsing command files and generating documentation.
    Specific parser implementations should inherit from this class.
    """

    @classmethod
    def get(cls, name: str) -> ParseCommand:
        """
        Get a parser instance by name from the registry.

        Args:
            name (str): Name of parser to retrieve

        Returns:
            ParseCommand: Instance of requested parser

        Raises:
            ValueError: If parser name not found in registry
        """
        try:
            return cls._registry[name]()  # type: ignore
        except KeyError as e:
            msg = f"Command parser ({name}) not found."
            raise ValueError(msg) from e

    @abstractmethod
    def parse_command_file(self, path: str) -> list[Command]:
        """
        Define how to parse a file into a list of commands.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_command_docs(self, commands: list[Command], **kwargs: str) -> str:
        """
        Generate a string of documentation for the given commands and subroutine types.
        """
        raise NotImplementedError


# DEFINE NEW COMMAND PARSER FUNCTIONS BELOW THIS LINE


class ParseCommandBash(ParseCommand):
    """
    Parser for bash command files.

    Handles parsing of bash function definitions and script files with YAML
    docstrings. Supports both function-based and script-based commands.
    """

    def parse_command_file(self, path: str) -> list[Command]:
        """
        Parse a file into a list of commands.

        Handles both script files (with shebang) and bash function files.
        Validates file extensions and docstring formats.

        Args:
            path (str): Path to file to parse

        Returns:
            list[Command]: List of parsed commands

        Raises:
            ValueError: If file format or docstring is invalid
        """
        with open(path) as file:
            contents = file.read()
        if contents.strip().startswith("#!"):
            commands = self.parse_script(path, contents)
        else:
            if Path(path).suffix != ".sh" and not Path(path).name.startswith("_"):
                msg = (
                    f"Source file {path} does not have a .sh extension.\n"
                    "Only .sh files are supported for bash function parsing.\n"
                    "If you want to use a non-shell file as a command (script), "
                    "it should use a shebang (e.g. #!/usr/bin/env python)."
                )
                raise ValueError(msg)
            return self.parse_bash_functions(path, contents)
        if len(commands) == 0 and not Path(path).name.startswith("_"):
            msg = (
                f"Non-shell file {path} does not contain any commands.\n"
                "If you want to use a non-shell file as a command (script), "
                "it should contain exactly one @yaml docstring. "
                "If you want to use a file as a utility script, "
                "it should start with an underscore (e.g. _utils.py)."
            )
            raise ValueError(msg)
        else:
            return commands

    def parse_bash_functions(self, path: str, contents: str) -> list[Command]:
        """
        Parse bash file into function-based commands.

        Extracts function definitions and their YAML docstrings.
        Assumes functions have opening bracket on same line and
        closing bracket on separate line.

        Args:
            path: Path to bash file
            contents (str): File contents

        Returns:
            list[Command]: List of parsed function commands
        """
        lines = contents.split("\n")
        commands = []
        idx = 0
        docs = []
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            if line.startswith("# "):
                docs.append(line[2:])
            elif line.strip().endswith("() {"):
                name = line.split()[0][:-2]
                code = line
                while lines[idx].strip() != "}":
                    code += lines[idx]
                    idx += 1
                code += lines[idx]
                docstring, end_name, arguments, signature = None, None, None, name
                docs_dict = yaml.safe_load("\n".join(docs).replace("@yaml", ""))
                if docs_dict is not None:
                    docstring = docs_dict["docstring"]
                    end_name = docs_dict.get("end_name", None)
                    arguments = docs_dict.get("arguments", None)
                    if "signature" in docs_dict:
                        signature = docs_dict["signature"]
                    elif arguments is not None:
                        for param, settings in arguments.items():
                            if settings["required"]:
                                signature += f" <{param}>"
                            else:
                                signature += f" [<{param}>]"
                command = Command.from_dict(
                    {
                        "code": code,
                        "docstring": docstring,
                        "end_name": end_name,
                        "name": name,
                        "arguments": arguments,
                        "signature": signature,
                    },
                )
                commands.append(command)
                docs = []
        return commands

    def parse_script(self, path: str, contents: str) -> list[Command]:
        """
        Parse script file into commands.

        Extracts YAML docstrings from script files and creates
        corresponding commands.

        Args:
            path: Path to script file
            contents: File contents

        Returns:
            list[Command]: List of parsed script commands

        Raises:
            ValueError: If multiple YAML tags found in script
        """
        pattern = re.compile(r"^#\s*@yaml\s*\n^#.*(?:\n#.*)*", re.MULTILINE)
        matches = pattern.findall(contents)
        if len(matches) == 0:
            return []
        elif len(matches) > 1:
            msg = "Non-shell file contains multiple @yaml tags.\nOnly one @yaml tag is allowed per script."
            raise ValueError(msg)
        else:
            yaml_content = matches[0]
            yaml_content = re.sub(r"^#", "", yaml_content, flags=re.MULTILINE)
            docs_dict = yaml.safe_load(yaml_content.replace("@yaml", ""))
            assert docs_dict is not None
            docstring = docs_dict["docstring"]
            end_name = docs_dict.get("end_name", None)
            arguments = docs_dict.get("arguments", None)
            signature = docs_dict.get("signature", None)
            name = Path(path).name.rsplit(".", 1)[0]
            if signature is None and arguments is not None:
                signature = name
                for param, settings in arguments.items():
                    if settings["required"]:
                        signature += f" <{param}>"
                    else:
                        signature += f" [<{param}>]"
            code = contents
            return [
                Command.from_dict(
                    {
                        "code": code,
                        "docstring": docstring,
                        "end_name": end_name,
                        "name": name,
                        "arguments": arguments,
                        "signature": signature,
                    },
                ),
            ]

    def generate_command_docs(self, commands: list[Command], **kwargs: str) -> str:
        """
        Generate documentation for commands.

        Args:
            commands (list[Command]): Commands to document
            **kwargs: Format string parameters

        Returns:
            str: Formatted command documentation
        """
        docs = ""
        for cmd in commands:
            if cmd.docstring is not None:
                docs += f"{cmd.signature or cmd.name} - {cmd.docstring.format(**kwargs)}\n"
        return docs


class ParseCommandDetailed(ParseCommandBash):
    """
    Parser for detailed command specifications.

    Extends bash parser with more detailed documentation including
    argument types, requirements, and descriptions.

    Format:
    # command_name:
    #   "docstring"
    #   signature: "signature"
    #   arguments:
    #     arg1 (type) [required]: "description"
    #     arg2 (type) [optional]: "description"
    """

    @staticmethod
    def get_signature(cmd: Command) -> str:
        """
        Generate command signature from command object.

        Args:
            cmd: Command object to generate signature for

        Returns:
            str: Formatted command signature
        """
        signature = cmd.name
        if "arguments" in cmd.__dict__ and cmd.arguments is not None:
            if cmd.end_name is None:
                for param, settings in cmd.arguments.items():
                    if settings["required"]:
                        signature += f" <{param}>"
                    else:
                        signature += f" [<{param}>]"
            else:
                for param, settings in list(cmd.arguments.items())[:-1]:
                    if settings["required"]:
                        signature += f" <{param}>"
                    else:
                        signature += f" [<{param}>]"
                signature += f"\n{next(iter(cmd.arguments[-1].keys()))}\n{cmd.end_name}"
        return signature

    def generate_command_docs(
        self,
        commands: list[Command],
        **kwargs: str,
    ) -> str:
        """
        Generate detailed documentation for commands.

        Includes docstrings, signatures, and argument specifications.

        Args:
            commands (list[Command]): Commands to document
            **kwargs: Format string parameters

        Returns:
            str: Formatted detailed command documentation
        """
        docs = ""
        for cmd in commands:
            docs += f"{cmd.name}:\n"
            if cmd.docstring is not None:
                docs += f"  docstring: {cmd.docstring.format(**kwargs)}\n"
            if cmd.signature is not None:
                docs += f"  signature: {cmd.signature}\n"
            else:
                docs += f"  signature: {self.get_signature(cmd)}\n"
            if "arguments" in cmd.__dict__ and cmd.arguments is not None:
                docs += "  arguments:\n"
                for param, settings in cmd.arguments.items():
                    req_string = "required" if settings["required"] else "optional"
                    docs += f"    - {param} ({settings['type']}) [{req_string}]: {settings['description']}\n"
            docs += "\n"
        return docs
