"""
Copyright (c) Meta Platforms, Inc. and affiliates.

History processing utilities for managing conversation history in ML agents.

This module provides classes and functions to process and manage conversation history,
including filtering, truncating, and formatting conversation entries. It helps prevent
context window overflow and maintains relevant conversation context.

Adapted from SWE-agent/sweagent/agent/history_processors.py
"""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from mlgym.types import HistoryItem


class FormatError(Exception):
    """Exception raised when history format is invalid."""

    pass


# ABSTRACT BASE CLASSES


class HistoryProcessorMeta(type):
    """
    Metaclass for history processors that maintains a registry of processor types.

    Attributes:
        _registry (dict): Dictionary mapping processor names to their classes
    """

    # TODO: Move to new meta class functionality
    _registry: ClassVar = {}

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        """
        Creates new history processor class and adds it to registry.

        Args:
            name (str): Name of the class
            bases (tuple): Base classes
            attrs (dict): Class attributes

        Returns:
            type: New history processor class
        """
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "HistoryProcessor":
            cls._registry[name] = new_cls
        return new_cls


@dataclass
class HistoryProcessor(metaclass=HistoryProcessorMeta):
    """
    Base class for history processors that modify conversation history.

    All history processors should inherit from this class and implement
    the __call__ method to process history items.
    """

    def __init__(self, *args: str, **kwargs: int) -> None:
        """Initialize the history processor."""
        pass

    @abstractmethod
    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process the conversation history.

        Args:
            history (list[HistoryItem]): List of conversation history items

        Returns:
            list[HistoryItem]: Processed history items

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @classmethod
    # NOTE: Acceptable standard for args and kwargs type annotation: https://stackoverflow.com/questions/37031928/type-annotations-for-args-and-kwargs
    def get(cls, name: str, *args: str, **kwargs: int) -> HistoryProcessor:
        """
        Get a history processor instance by name.

        Args:
            name (str): Name of the processor to retrieve
            *args: Positional arguments for processor initialization
            **kwargs: Keyword arguments for processor initialization

        Returns:
            HistoryProcessor: Instance of the requested processor

        Raises:
            ValueError: If processor name is not found in registry
        """
        try:
            # FIXME: Fix as part of new meta class functionality
            return cls._registry[name](*args, **kwargs)  # type: ignore
        except KeyError as e:
            msg = f"Model output parser ({name}) not found."
            raise ValueError(msg) from e


# DEFINE NEW PARSING FUNCTIONS BELOW THIS LINE
class DefaultHistoryProcessor(HistoryProcessor):
    """History processor that returns history unchanged."""

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Return history without modifications.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Same history unchanged
        """
        return history


def last_n_history(history: list[HistoryItem], n: int) -> list[HistoryItem]:
    """
    Keep the first message and last n user messages in history.

    For messages not in the last n, replaces content with a line count summary.
    Demo messages are always preserved.

    Args:
        history (list[HistoryItem]): List of history items
        n (int): Number of recent messages to keep

    Returns:
        list[HistoryItem]: Processed history with only recent messages

    Raises:
        ValueError: If n is not positive
    """
    if n <= 0:
        msg = "n must be a positive integer"
        raise ValueError(msg)
    new_history = []
    user_messages = len([entry for entry in history if (entry["role"] == "user" and not entry.get("is_demo", False))])
    user_msg_idx = 0
    for entry in history:
        data = entry.copy()
        if data["role"] != "user":
            new_history.append(entry)
            continue
        if data.get("is_demo", False):
            new_history.append(entry)
            continue
        else:
            user_msg_idx += 1
        if user_msg_idx == 1 or user_msg_idx in range(user_messages - n + 1, user_messages + 1):
            new_history.append(entry)
        else:
            data["content"] = f"Old output omitted ({len(entry['content'].splitlines())} lines)"  # type: ignore
            new_history.append(data)
    return new_history


class LastNObservations(HistoryProcessor):
    """
    Processor that keeps the first and last N user messages.

    Args:
        n (int): Number of recent messages to keep
    """

    def __init__(self, n: int) -> None:
        """
        Initialize with number of messages to keep.

        Args:
            n (int): Number of recent messages to keep
        """
        self.n = n

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process history to keep only recent messages.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Processed history with only recent messages
        """
        return last_n_history(history, self.n)


class Last2Observations(HistoryProcessor):
    """Processor that keeps the first and last 2 user messages."""

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process history to keep last 2 messages.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Processed history with last 2 messages
        """
        return last_n_history(history, 2)


class Last5Observations(HistoryProcessor):
    """Processor that keeps the first and last 5 user messages."""

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process history to keep last 5 messages.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Processed history with last 5 messages
        """
        return last_n_history(history, 5)


class Last100Observations(HistoryProcessor):
    """Processor that keeps the first and last 100 user messages."""

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process history to keep last 100 messages.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Processed history with last 100 messages
        """
        return last_n_history(history, 100)


class ClosedWindowHistoryProcessor(HistoryProcessor):
    """
    Processor that manages window-based history by replacing outdated windows.

    Identifies code windows in the history and replaces outdated windows
    (older windows for the same file) with a summary line count.
    """

    pattern = re.compile(r"^(\d+)\:.*?(\n|$)", re.MULTILINE)
    file_pattern = re.compile(r"\[File:\s+(.*)\s+\(\d+\s+lines\ total\)\]")

    def __call__(self, history: list[HistoryItem]) -> list[HistoryItem]:
        """
        Process history to manage code windows.

        Keeps the most recent window for each file and replaces older
        windows with line count summaries.

        Args:
            history (list[HistoryItem]): Input history

        Returns:
            list[HistoryItem]: Processed history with managed windows
        """
        new_history = []
        # For each value in history, keep track of which windows have been shown.
        # We want to mark windows that should stay open (they're the last window for a particular file)
        # Then we'll replace all other windows with a simple summary of the window (i.e. number of lines)
        windows = set()
        for entry in reversed(history):
            data = entry.copy()
            if data["role"] != "user":
                new_history.append(entry)
                continue
            if data.get("is_demo", False):
                new_history.append(entry)
                continue
            matches = list(self.pattern.finditer(str(entry.get("content", ""))))
            if len(matches) >= 1:
                file_match = self.file_pattern.search(str(entry.get("content", "")))
                if file_match:
                    file = file_match.group(1)
                else:
                    continue
                if file in windows:
                    start = matches[0].start()
                    end = matches[-1].end()
                    content = entry.get("content") or ""
                    data["content"] = (
                        content[:start] + f"Outdated window with {len(matches)} lines omitted...\n" + content[end:]
                    )
                windows.add(file)
            new_history.append(data)
        return list(reversed(new_history))
