"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from SWE-agent/sweagent/types.py
This file has types/dataclass definitions that are used in MLGym for exchanging information between modules/functions/classes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict


class TrajectoryStep(TypedDict):
    """A single step in an agent's trajectory through the environment.

    Represents the sequence of agent-environment interaction:
    1. state: Current environment state
    2. response: Raw model output
    3. thought: Extracted reasoning
    4. action: Selected action
    5. execution_time: Action execution duration
    6. observation: Environment response
    """

    state: str | None
    response: str
    thought: str
    action: str
    execution_time: float
    observation: str | None


class _HistoryItem(TypedDict):
    role: str


class HistoryItem(_HistoryItem, total=False):
    content: str | None
    agent: str
    is_demo: bool
    thought: str
    action: str | None


History = list[HistoryItem]
Trajectory = list[TrajectoryStep]


# FIXME: The types need some love from developers.
class AgentInfo(defaultdict):
    def __init__(self) -> None:
        super().__init__(lambda: None)
        self.model_stats: dict[str, float] = {}
        self.exit_status: str = ""
        self.submission: str | None = None
        self.score: list[dict[str, float]] = []
        self.summarizer: dict = {}

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return self[name]

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        self[name] = value

    def update(self, other: dict[str, Any]) -> None:  # type: ignore
        for key, value in other.items():
            if key == "score" and isinstance(value, list):
                self.score.extend(value)
            else:
                self[key] = value
