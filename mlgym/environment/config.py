"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Core environment implementation for the MLGym framework.

This module provides the main environment class and supporting functionality for
running machine learning tasks. It handles container management, task execution,
file operations, and agent interactions. The environment supports various task types
and manages the workspace, evaluation, and resource limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from simple_parsing import choice
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable

from mlgym import CONFIG_DIR
from mlgym.environment.tasks import (
    AbstractMLTask,
    TaskConfig,
)
from mlgym.utils.extras import multiline_representer

yaml.add_representer(str, multiline_representer)
yaml.representer.SafeRepresenter.add_representer(str, multiline_representer)  # type: ignore


@dataclass(frozen=True)
class EnvironmentConfig(FlattenedAccess, FrozenSerializable):
    """
    Configuration for the MLGym environment.
    One env is always related to a single task.
    """

    # Name of the docker image to use for the environment. Defaults to mlgym/mlgym-agent:latest
    image_name: str = "aigym/mlgym-agent:latest"
    # maximum number of agent steps in the environment
    max_steps: int = 100
    # random seed
    seed: int = 42
    # ! TODO: Should be moved to ScriptArguments along with benchmark config
    # can be set by the user if evaluation should be done on a single task
    # accepts a TaskConfig object or a path to a yaml file relative to the CONFIG_DIR
    task_config_path: Path | str | None = None
    # task config object
    task: TaskConfig | None = None
    # container type to use. options are "docker"
    container_type: str = choice("docker", "podman", default="docker")
    # * CURRENTLY NOT USED
    # Only used for docker container. Use a persistent container with this name. After every task, the container will be paused, but not removed.
    container_name: str | None = None
    # Enable environment logger.
    verbose: bool = False
    # cache baseline scores for each dataset
    cache_baseline_scores: bool = True
    # * CURRENTLY NOT USED
    # Cache task images to speed up task initialization. This means that the environment will be saved as a docker image.
    cache_task_images: bool = False
    # * CURRENTLY NOT USED
    # Custom environment setup. Currently only used when running on a single task.
    # This needs to be either a string pointing to a yaml file (with yaml, yml file extension)
    # or a shell script (with sh extension). Generally this can be used to setup custom conda environments or docker containers for a task.
    environment_setup: str | None = None
    # Enable memory
    memory_enabled: bool = False
    # * CURRENTLY NOT USED
    # Container mounts = addiitonal folders to mount into the environment (useful for caching)
    container_mounts: list[str] = field(default_factory=list)
    # Path to the bash aliases file to source in the container
    aliases_file: str | None = None

    def get_task_class(self) -> type[AbstractMLTask]:
        """
        Get the task class based on the task configuration.

        Returns:
            type[AbstractMLTask]: Task class to instantiate

        Raises:
            AssertionError: If task is not set
        """
        assert isinstance(self.task, TaskConfig)
        return AbstractMLTask.get(self.task.task_entrypoint)

    def __post_init__(self) -> None:
        """
        Validate and process configuration after initialization.

        Raises:
            ValueError: If task_config_path is not provided
            ValueError: If cache_task_images and container_name are both set
            ValueError: If container_name is empty string
        """
        if self.task_config_path is None:
            msg = "You must provide a Task Config to the environment. Please provide a "
            raise ValueError(msg)

        # always load the task config from the path
        object.__setattr__(self, "task", TaskConfig.load_yaml(CONFIG_DIR / self.task_config_path))

        if self.cache_task_images and self.container_name:
            msg = (
                "Not allowed to use persistent container with caching task images "
                "(probably doesn't make sense and takes excessive space)."
            )
            raise ValueError(msg)
        if self.container_name is not None and self.container_name.strip() == "":
            msg = "Set container_name to None if you don't want to use a persistent container."
            raise ValueError(msg)

