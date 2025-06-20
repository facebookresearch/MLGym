"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Utility functions for the MLGym framework.

This module provides utility functions for the MLGym framework.

Adapted from SWE-agent/sweagent/agent/models.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlgym.backend.debugging import ReplayModel, SubmitBaselineModel
from mlgym.backend.human import HumanModel, HumanThoughtModel
from mlgym.backend.litellm import LiteLLMModel

if TYPE_CHECKING:
    from mlgym.backend.base import BaseModel, ModelArguments
    from mlgym.tools.commands import Command


# ! TODO: Add a meta model class so that we can register custom model classes on the fly.
def get_model(args: ModelArguments, commands: list[Command] | None = None) -> BaseModel:
    """
    Returns correct model object given arguments and commands
    """
    if commands is None:
        commands = []
    if args.model_name == "submit_baseline":
        return SubmitBaselineModel(args)
    elif args.model_name == "human":
        return HumanModel(args, commands)
    elif args.model_name == "human_thought":
        return HumanThoughtModel(args, commands)
    elif args.model_name == "replay":
        return ReplayModel(args)
    elif args.model_name.startswith("litellm"):
        return LiteLLMModel(args)
    else:
        msg = f"Invalid model name: {args.model_name}. Please see models.py for valid model names or for adding a new model."
        raise ValueError(msg)
