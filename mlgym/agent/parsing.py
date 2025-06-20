"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Parser module for handling model outputs in the MLGym framework.

This module provides a collection of parser classes that handle different formats
of model outputs. Each parser is responsible for extracting thoughts and actions
from model responses in specific formats (JSON, XML, markdown code blocks, etc.).
The module supports extensible parsing through a registry pattern.

Adapted from SWE-agent/sweagent/tools/parsing.py
"""

from __future__ import annotations

import json
import re
import shlex
import string
import textwrap
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from mlgym.exceptions import FormatError

if TYPE_CHECKING:
    from mlgym.tools.commands import Command

# ABSTRACT BASE CLASSES


class ParseFunctionMeta(type):
    """
    Metaclass for parse functions that maintains a registry of parser types.

    Provides automatic registration of parser classes to enable lookup by name.

    Attributes:
        _registry (dict): Maps parser class names to their implementations
    """

    _registry: ClassVar = {}

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        """
        Creates new parser class and adds it to registry.

        Args:
            name (str): Name of the parser class
            bases (tuple): Base classes
            attrs (dict): Class attributes

        Returns:
            type: New parser class
        """
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "ParseFunction":
            cls._registry[name] = new_cls
        return new_cls


@dataclass
class ParseFunction(metaclass=ParseFunctionMeta):
    """
    Abstract base class for parsing model outputs.

    Defines the interface for all parser implementations and provides
    registry-based instantiation through the get() classmethod.

    Attributes:
        _error_message (str | None): Template for format error messages
    """

    _error_message: ClassVar[str | None] = None

    @abstractmethod
    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Parse the model response into thought and action components.

        Args:
            model_response (str): Raw response from the model
            commands (list[Command]): Available commands for validation
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple containing (thought, action)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @property
    def format_error_template(self) -> str:
        """
        Get the error message template for format errors.

        Returns:
            str: The error message template

        Raises:
            NotImplementedError: If _error_message is not defined
        """
        if self._error_message is None:
            msg = "You must define an error message for your parser."
            raise NotImplementedError(msg)
        return textwrap.dedent(self._error_message)

    @classmethod
    def get(cls, name: str, *args: str, **kwargs: int) -> ParseFunction:
        """
        Get a parser instance by name from the registry.

        Args:
            name (str): Name of the parser to retrieve

        Returns:
            ParseFunction: Instance of the requested parser

        Raises:
            ValueError: If parser name is not found in registry
        """
        try:
            return cls._registry[name](*args, **kwargs)  # type: ignore
        except KeyError as e:
            msg = f"Model output parser ({name}) not found."
            raise ValueError(msg) from e


# DEFINE NEW PARSING FUNCTIONS BELOW THIS LINE


class ActionParser(ParseFunction):
    """
    Parser for single command responses.

    Expects the model response to be a single command without additional text.
    Example: "ls -l"
    """

    _error_message = """\
    The command you provided was not recognized. Please specify one of the commands
    (+ any necessary arguments) from the following list in your response.
    Do not include any other text.

    COMMANDS:
    {command_docs}
    """

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Parse a single command response.

        Args:
            model_response (str): Raw model response
            commands (list[Command]): Available commands
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple of (model_response, model_response)

        Raises:
            FormatError: If first word is not a valid command
        """
        if model_response.split():
            action = model_response.strip().split()[0]
            if action in {command.name for command in commands}:
                return model_response, model_response
        msg = "First word in model response is not a valid command."
        raise FormatError(msg)


class MLGymThoughtActionParser(ParseFunction):
    """
    Parser for responses with discussion and single code block.

    Expects model response to contain discussion followed by a single command
    wrapped in backticks. Enforces single code block constraint.
    """

    _error_message = """\
    Your output was not formatted correctly. Your commands were not executed.
    You must always include one discussion and one command as part of your response.
    Make sure you do not have multiple discussion/command tags.

    Please make sure your output precisely matches the following format:
    DISCUSSION
    Discuss here with yourself about what you are planning and what you are going to do in this step.

    ```
    command(s) that you're going to run
    ```
    """

    def _extract_code_blocks(self, text: str) -> list[str]:
        """
        Extract all code blocks from text.

        Args:
            text (str): Text containing code blocks

        Returns:
            list[str]: List of extracted code block contents
        """
        pattern = r"```([\s\S]*?)```"

        # Find all matches in the text
        matches = re.findall(pattern, text, re.MULTILINE)
        return [block.strip() for block in matches]

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Parse response with discussion and code block.

        Args:
            model_response (str): Raw model response
            commands (list[Command]): Available commands
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple of (thought, action)

        Raises:
            FormatError: If multiple code blocks found or no action found
        """
        code_blocks = self._extract_code_blocks(model_response)
        if len(code_blocks) > 1:
            msg = "Found more than one code block in the model response."
            raise FormatError(msg)

        code_block_pat = re.compile(r"^```(\S*)\s*\n|^```\s*$", re.MULTILINE)
        stack: list[re.Match[str]] = []
        last_valid_block = None
        for match in code_block_pat.finditer(model_response):
            if stack and not match.group(1):  # Closing of a code block
                start = stack.pop()
                # Check if it's not nested within another block
                if not stack:
                    last_valid_block = (start, match)
            elif match.group(1) is not None:  # Opening of a code block
                stack.append(match)
        if last_valid_block:
            start, end = last_valid_block
            thought = model_response[: start.start()] + model_response[end.end() :]
            return thought, model_response[start.end() : end.start()]
        msg = "No action found in model response."
        raise FormatError(msg)


class ThoughtActionParser(ParseFunction):
    """
    Parser for responses with discussion and code block.

    Expects the model response to be a discussion followed by a command wrapped in backticks.
    Example:
    Let's look at the files in the current directory.
    ```
    ls -l
    ```
    """

    _error_message = """\
    Your output was not formatted correctly. You must always include one discussion and one command as part of your response. Make sure you do not have multiple discussion/command tags.
    Please make sure your output precisely matches the following format:
    DISCUSSION
    Discuss here with yourself about what you are planning and what you are going to do in this step.

    ```
    command(s) that you're going to run
    ```
    """

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Parse response with discussion and code block.

        Extracts the last code block as the action and everything else as thought.
        Handles nested code blocks by tracking opening/closing markers.

        Args:
            model_response (str): Raw model response
            commands (list[Command]): Available commands
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple of (thought, action)

        Raises:
            FormatError: If no action found in model response
        """
        code_block_pat = re.compile(r"^```(\S*)\s*\n|^```\s*$", re.MULTILINE)
        stack: list[re.Match[str]] = []
        last_valid_block = None
        for match in code_block_pat.finditer(model_response):
            if stack and not match.group(1):  # Closing of a code block
                start = stack.pop()
                # Check if it's not nested within another block
                if not stack:
                    last_valid_block = (start, match)
            elif match.group(1) is not None:  # Opening of a code block
                stack.append(match)
        if last_valid_block:
            start, end = last_valid_block
            thought = model_response[: start.start()] + model_response[end.end() :]
            return thought, model_response[start.end() : end.start()]
        msg = "No action found in model response."
        raise FormatError(msg)


class XMLThoughtActionParser(ParseFunction):
    """
    Parser for XML-formatted responses.

    Expects the model response to be a discussion followed by a command wrapped in XML tags.
    Example:
    Let's look at the files in the current directory.
    <command>
    ls -l
    </command>
    """

    _error_message = """\
    Your output was not formatted correctly. You must always include one discussion and one command as part of your response. Make sure you do not have multiple discussion/command tags.
    Please make sure your output precisely matches the following format:
    """

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Parse XML-formatted response.

        Extracts the last command block as the action and everything else as thought.
        Handles multiple command blocks by using the last one.

        Args:
            model_response (str): Raw model response
            commands (list[Command]): Available commands
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple of (thought, action)

        Raises:
            FormatError: If no command tags found in response
        """
        if "<command>" not in model_response or "</command>" not in model_response:
            msg = "No action found in model response."
            raise FormatError(msg)
        # `action` is everything between the last <command> and </command> tags
        start_action = model_response.rfind("<command>") + len("<command>")  # start after the last <command> tag
        end_thought = model_response.rfind("<command>")  # end before the last <command> tag
        end_action = model_response.rfind("</command>")  # end before the last </command> tag
        restart_thought = model_response.rfind("</command>") + len("</command>")  # start after the last </command> tag
        # `thought` is everything not in between <command> and </command> tags (includes after the last </command> tag)
        action = model_response[start_action:end_action]
        thought = model_response[:end_thought] + model_response[restart_thought:]

        return thought.strip(), action.strip()


class EditFormat(ThoughtActionParser):
    """
    Parser for edit operations.

    Expects the model response to be a discussion followed by replacement text in backticks.
    Used for editing file contents or making text replacements.

    Example:
    We'll replace the contents of the current window with the following:
    ```
    import os
    os.listdir()
    ```
    """

    _error_message = """\
    Your output was not formatted correctly. You must wrap the replacement text in backticks (```).
    Please make sure your output precisely matches the following format:
    COMMENTS
    You can write comments here about what you're going to do if you want.

    ```
    New window contents.
    Make sure you copy the entire contents of the window here, with the required indentation.
    Make the changes to the window above directly in this window.
    Remember that all of the window's contents will be replaced with the contents of this window.
    Don't include line numbers in your response.
    ```
    """


class Identity(ParseFunction):
    """
    Pass-through parser that returns input unchanged.

    This parser does not perform any parsing or validation. It simply returns
    the model response as both the thought and action components.
    """

    _error_message = """\
    It seems like something went wrong with your output. Please try again.
    """

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:
        """
        Return model response unchanged.

        Args:
            model_response (str): Raw model response
            commands (list[Command]): Available commands (unused)
            strict (bool, optional): Whether to enforce strict parsing (unused)

        Returns:
            tuple[str, str]: Tuple of (model_response, model_response)
        """
        return model_response, model_response


class JsonParser(ParseFunction):
    """
    Parser for JSON-formatted model responses.

    Expects the model response to be a JSON object with a specific structure containing
    thought and command information. The command can include arguments and must reference
    valid command names from the available commands list.

    Expected JSON format:
    {
        "thought": "discussion text here.",
        "command": {
            "arguments": {
                "arg1": "value1",
                "arg2": "value2",
                ...
            },
            "name": "command_name"
        }
    }
    """

    _error_message = """\
    Your output could not be parsed as JSON. Please make sure your output 1) is valid JSON and
    2) Includes the "thought" and "command" fields.

    """

    def __call__(self, model_response: str, commands: list[Command], strict: bool = False) -> tuple[str, str]:  # noqa: C901
        """
        Parse JSON-formatted model response into thought and action components.

        Validates JSON structure and extracts thought and command information.
        Handles command arguments and formatting based on command signatures.

        Args:
            model_response (str): Raw JSON response from model
            commands (list[Command]): Available commands for validation
            strict (bool, optional): Whether to enforce strict parsing. Defaults to False

        Returns:
            tuple[str, str]: Tuple containing:
                - thought: Extracted thought text
                - action: Formatted command string with arguments

        Raises:
            FormatError: If JSON is invalid or missing required fields
            FormatError: If command structure is invalid
            json.JSONDecodeError: If input is not valid JSON
        """
        try:
            data = json.loads(model_response)
            if not isinstance(data, dict):
                msg = "Model output is not a JSON object."
                raise FormatError(msg)

            # Check if required keys are present
            required_keys = ["thought", "command"]
            for key in required_keys:
                if key not in data:
                    msg = f"Key '{key}' is missing from model output."
                    raise FormatError(msg)

            # Check structure of 'command' key
            data_command = data["command"]
            if not isinstance(data_command, dict):
                msg = "Value of 'command' key is not a JSON object."
                raise FormatError(msg)

            # Check if required keys are present in 'command' object
            command_keys = ["name"]
            for key in command_keys:
                if key not in data_command:
                    msg = f"Key '{key}' is missing from 'command' object."
                    raise FormatError(msg)

            thought = data["thought"]

            # Generate action
            commands_dict = {c.name: c for c in commands}
            command = commands_dict.get(data_command["name"])
            if command is None:
                action = data_command["name"]
                if "arguments" in data_command:
                    action += " " + " ".join(data_command["arguments"].values())
            else:
                signature = command.signature
                if signature is None:
                    msg = f"Command signature is not defined for command: {data_command['name']}"
                    raise RuntimeError(msg)
                signature = signature.replace("[", "").replace("]", "").replace("<", "{").replace(">", "}")
                signature_args = extract_keys(signature)
                command_args = dict.fromkeys(signature_args, "")

                if "arguments" in data_command:
                    for arg in signature_args:
                        if arg in data_command["arguments"]:
                            value = data_command["arguments"][arg]
                            if should_quote(value, command):
                                value = shlex.quote(value)
                            command_args[arg] = value
                action = signature.format(**command_args)
            action = action.strip()
        except json.JSONDecodeError as e:
            msg = "Model output is not valid JSON."
            raise FormatError(msg) from e
        else:
            return thought, action


def extract_keys(format_string: str) -> set[str]:
    """
    Extract all format string keys from a string.

    Args:
        format_string (str): String containing format placeholders

    Returns:
        set[str]: Set of unique format keys found in string

    Example:
        >>> extract_keys("Hello {name}, you are {age} years old")
        {'name', 'age'}
    """
    formatter = string.Formatter()
    keys = set()
    for _, field_name, _, _ in formatter.parse(format_string):
        if field_name is not None:
            keys.add(field_name)
    return keys


def should_quote(value: str | Any, command: Command) -> bool:  # noqa: ANN401
    """
    Determine if a command argument value should be quoted.

    Args:
        value (str | Any): Value to check for quoting
        command (Command): Command context for the value

    Returns:
        bool: True if value should be quoted, False otherwise

    Note:
        Values are quoted if they are strings and the command doesn't
        have an end_name specified.
    """
    return isinstance(value, str) and command.end_name is None
