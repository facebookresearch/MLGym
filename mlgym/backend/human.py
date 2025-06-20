"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Human model implementation for the MLGym framework.
This module provides a human-in-the-loop model that allows for interactive

Adapted from SWE-agent/sweagent/agent/models.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mlgym.backend.base import BaseModel, ModelArguments
from mlgym.types import HistoryItem

if TYPE_CHECKING:
    from mlgym.tools.commands import Command
    from mlgym.types import HistoryItem


class HumanModel(BaseModel):
    """
    Human model implementation for the MLGym framework.
    """

    MODELS: ClassVar = {"human": {}}

    def __init__(self, args: ModelArguments, commands: list[Command]) -> None:
        super().__init__(args)

        # Determine which commands require multi-line input
        self.multi_line_command_endings = {
            command.name: command.end_name for command in commands if command.end_name is not None
        }

    def query(self, history: list[HistoryItem], action_prompt: str = "> ") -> str:
        """
        Logic for handling user input to pass to MLGym
        """
        action = input(action_prompt)
        command_name = action.split()[0] if action.strip() else ""

        # Special handling for multi-line input actions (i.e. edit)
        if command_name in self.multi_line_command_endings:
            buffer = [action]
            end_keyword = self.multi_line_command_endings[command_name]
            while True:
                action = input("... ")
                buffer.append(action)
                if action.rstrip() == end_keyword:
                    # Continue reading input until terminating keyword inputted
                    break
            action = "\n".join(buffer)
        elif action.strip() == "start_multiline_command":
            # do arbitrary multi-line input
            buffer = []
            while True:
                action = input("... ")
                if action.rstrip() == "end_multiline_command":
                    break
                buffer.append(action)
            action = "\n".join(buffer)
        return action

    # FIXME: Bad Pattern I guess
    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0) -> float:
        return super().update_stats(input_tokens, output_tokens, cost)

    # FIXME: Bad Pattern I guess
    def history_to_messages(
        self, history: list[HistoryItem], is_demonstration: bool = False
    ) -> str | list[dict[str, str]]:
        return super().history_to_messages(history, is_demonstration)


class HumanThoughtModel(HumanModel):
    """
    Human model implementation for the MLGym framework.
    """

    MODELS: ClassVar = {"human_thought": {}}

    def query(self, history: list[HistoryItem], action_prompt: str = "> ") -> str:
        """
        Logic for handling user input for both thought and action to pass to MLGym
        """
        thought_all = ""
        thought = input("Thought (end with END_THOUGHT): ")
        while True:
            if "END_THOUGHT" in thought:
                thought = thought.split("END_THOUGHT")[0]
                thought_all += thought
                break
            thought_all += thought
            thought = input("... ")

        action = super().query(history, action_prompt=action_prompt)

        return f"{thought_all}\n```\n{action}\n```"
