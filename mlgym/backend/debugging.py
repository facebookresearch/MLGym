"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Debugging models for the MLGym framework.

This module provides a collection of debugging models that can be used to test
and debug the MLGym framework. These models are useful for simulating different
scenarios and validating the framework's functionality.

Adapted from SWE-agent/sweagent/agent/models.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mlgym.backend.base import BaseModel, ModelArguments
from mlgym.types import HistoryItem

if TYPE_CHECKING:
    from mlgym.types import HistoryItem


class SubmitBaselineModel(BaseModel):
    """
    Model that immediately submits. Useful for testing
    """

    MODELS: ClassVar = {"submit_baseline": {}}

    def __init__(self, args: ModelArguments) -> None:
        """
        This model immediately submits. Useful for testing
        """
        super().__init__(args)
        self._action_idx = 0

    def query(self, history: list[HistoryItem]) -> str:
        # Need to run baseline before submitting
        if self._action_idx == 0:
            self._action_idx += 1
            action = "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython baseline.py\n```\n"
        elif self._action_idx == 1:
            self._action_idx = 0
            action = (
                "DISCUSSION\nWe have reproduced the baseline method, so let's submit the results.\n\n```\nsubmit\n```\n"
            )
        else:
            action = "```\nsubmit\n```"
        return action

    # FIXME: Bad Pattern I guess
    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0) -> float:
        return super().update_stats(input_tokens, output_tokens, cost)

    # FIXME: Bad Pattern I guess
    def history_to_messages(
        self, history: list[HistoryItem], is_demonstration: bool = False
    ) -> str | list[dict[str, str]]:
        return super().history_to_messages(history, is_demonstration)


class SubmitBaselineRLModel(SubmitBaselineModel):
    """
    Model that immediately submits. Useful for testing
    """

    MODELS: ClassVar = {"submit_baseline_rl": {}}

    def query(self, history: list[HistoryItem]) -> str:
        # Need to run baseline before submitting
        if self._action_idx == 0:
            self._action_idx += 1
            action = "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython src/train.py\n```\n"
        elif self._action_idx == 1:
            self._action_idx = 0
            action = (
                "DISCUSSION\nWe have reproduced the baseline method, so let's submit the results.\n\n```\nsubmit\n```\n"
            )
        else:
            action = "```\nsubmit\n```"
        return action


class SubmitBaselineWrongModel(BaseModel):
    """
    Model that immediately submits the wrong artefact. Useful for testing
    """

    MODELS: ClassVar = {"submit_baseline_wrong": {}}

    def __init__(self, args: ModelArguments) -> None:
        """
        This model immediately submits. Useful for testing
        """
        super().__init__(args)
        self._action_idx = 0

    def query(self, history: list[HistoryItem]) -> str:
        # Need to run baseline before submitting
        if self._action_idx == 0:
            self._action_idx += 1
            action = "DISCUSSION\nLet's reproduce the baseline method using the baseline script.\n\n```\npython baseline.py\n```\n\nNow let's look at the files in the curren directory.\n\n```\nls -a\n```\n"
        elif self._action_idx == 1:
            self._action_idx = 0
            action = (
                "DISCUSSION\nWe have reproduced the baseline method, so let's submit the results.\n\n```\nsubmit\n```\n"
            )
        else:
            action = "```\nsubmit\n```"
        return action


class ReplayModel(BaseModel):
    """
    Model that replays a sequence of actions from a file. Useful for testing
    """

    MODELS: ClassVar = {"replay": {}}

    def __init__(self, args: ModelArguments) -> None:
        super().__init__(args)

        if self.args.replay_path is None or not Path(self.args.replay_path).exists():
            msg = "--replay_path must point to a file that exists to run a replay policy"
            raise ValueError(msg)

        self.replays = [
            next(iter(json.loads(x).values()))
            for x in Path(self.args.replay_path).read_text().splitlines(keepends=True)
        ]
        self.replay_idx = 0
        self.action_idx = 0

    def _next_replay(self) -> None:
        """Called after last action"""
        self.replay_idx += 1
        self.action_idx = 0

    def query(self, history: list[HistoryItem]) -> str:
        """
        Logic for tracking which replay action to pass to MLGym
        """
        actions = self.replays[self.replay_idx]
        try:
            action = actions[self.action_idx]
        except IndexError:
            msg = (
                "This seems to be an incomplete trajectory. "
                "We reached the end of it, but `submit` was not called. "
                "Calling it now."
            )
            self.logger.warning(msg)
            action = "```\nsubmit\n```"

        self.action_idx += 1

        # Assuming `submit` is always last action of replay trajectory
        if action == "submit":
            self._next_replay()

        return str(action)

    # FIXME: Bad Pattern I guess
    def update_stats(self, input_tokens: int, output_tokens: int, cost: float = 0) -> float:
        return super().update_stats(input_tokens, output_tokens, cost)

    # FIXME: Bad Pattern I guess
    def history_to_messages(
        self, history: list[HistoryItem], is_demonstration: bool = False
    ) -> str | list[dict[str, str]]:
        return super().history_to_messages(history, is_demonstration)
