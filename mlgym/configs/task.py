"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Config module for task module.

This module provides the configuration interface for the task module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from simple_parsing.helpers.fields import field
from simple_parsing.helpers.serialization import encode
from simple_parsing.helpers.serialization.serializable import (
    Serializable,
)

from mlgym import CONFIG_DIR
from mlgym.configs.dataset import BaseDatasetConfig
from mlgym.utils.extras import multiline_representer

AGENT_LONG_ACTION_TIMEOUT = int(os.getenv("MLGYM_AGENT_LONG_TIMEOUT", "3600"))


# FIXME: All RUF009 should be resolved as part of issue #19.
@dataclass
class BaseTaskConfig(Serializable):
    """
    Configuration for a MLGym task. A task can be tied to a single dataset or multiple datasets.
    """

    id: str  # text identifier for the task. Will be used to register the task with the gym environment
    name: str  # name of the task
    description: str  # description/goal of the task
    # list of paths to the dataset config files
    # paths should be relative to the CONFIG_DIR
    dataset_configs: list[str] = field(default_factory=list)  # noqa: RUF009

    _datasets: list[BaseDatasetConfig] = field(default_factory=list, init=False)  # noqa: RUF009
    # task class to use to instantiate the task
    task_entrypoint: str = "CSVSubmissionTasks"
    # maximum time (in seconds) allowed for each training run
    training_timeout: int | None = None
    # path to a requirements.txt file for creating a task specific conda environment
    requirements_path: Path | str | None = None
    # benchmark name to associate with the task. If None, the task will not be registered with a benchmark
    benchmark_name: str | None = None
    # use the default mlgym conda environment
    use_generic_conda: bool = True
    # TODO: maybe these should be tied to the dataset name as a dictionary?
    # TODO: we can create a baseline_config class that contains the baseline_path, baseline_score and dataset_name. But at this point do we really want to create a new config class? This might overload the user.
    # ASSUMPTION: the baseline_paths and baseline_scores are provided in the same order as the datasets
    # path to the starter code files for this task. This can include baseline code, evaluation script, any other local libraries etc.
    starter_code: list[str] = field(default_factory=list)  # noqa: RUF009
    # path to a baseline script for this task. This path should be relative to the `data/<task_name>` directory. Eg: for rlMountainCarContinuous, the path should be `baseline/train.py`.
    baseline_paths: list[Path | str] = field(default_factory=list)  # noqa: RUF009
    # baseline scrores for each dataset. If None, the baseline scores will be computed using the baseline scripts
    baseline_scores: list[dict[str, float]] = field(default_factory=list)  # noqa: RUF009
    # path to a sample submission file for the task
    sample_submission: Path | str | None = None
    # path to an evaluation script for the task. This path should be relative to the `data/<task_name>` directory. Eg: for rlMountainCarContinuous, the path should be `evaluate.py`.
    evaluation_paths: list[Path | str] = field(default_factory=list)  # noqa: RUF009
    # read-only flag for the evaluation script. If True, the agent will not have write access to the evaluation script. NOTE: This can cause evaluation script to fail but makes the evaluation more robust to agent's actions. USE IT AT YOUR OWN RISK.
    evaluation_read_only: bool = False
    # ! TODO: NOT IMPLEMENTED YET.
    # list of files where the agent should not have any access to. This is useful for cases like battle of sexes where the agent should not be able to peak into the target strategy.
    secret_files: list[Path | str] = field(default_factory=list)  # noqa: RUF009
    # path to a memory file for the task. This file will be used to store the task's memory.
    memory_path: Path | str | None = None

    # FIXME: use dump_yaml from superclass with a dump_fn argument.
    def dump_yaml(self, stream, **kwargs: Any) -> str:  # type: ignore # noqa: ANN001, ANN401
        # add multiline representer for description
        yaml.add_representer(str, multiline_representer)
        yaml.representer.SafeRepresenter.add_representer(str, multiline_representer)  # type: ignore

        data = encode(self)
        # Remove None values
        data = {k: v for k, v in data.items() if (k != "_datasets" and v is not None and v != [] and v != {})}
        return yaml.safe_dump(data, stream, **kwargs)  # type: ignore

    def __post_init__(self) -> None:
        # load dataset configs from paths
        if len(self.dataset_configs) > 0:
            datasets = []
            for path in self.dataset_configs:
                dataset_config_path = CONFIG_DIR / path
                if not dataset_config_path.exists():
                    msg = f"Dataset config file not found at {dataset_config_path}"
                    raise FileNotFoundError(msg)
                datasets.append(BaseDatasetConfig.load_yaml(dataset_config_path))
            object.__setattr__(self, "_datasets", datasets)

        if self.sample_submission is not None and not Path(self.sample_submission).exists():
            msg = f"Sample submission path provided but file not found at {self.sample_submission}"
            raise FileNotFoundError(msg)

        # check if requirements_path is provided if use_generic_conda is False
        if not self.use_generic_conda and (self.requirements_path is None or not Path(self.requirements_path).exists()):
            msg = "Requirements path must be provided if use_generic_conda is False"
            raise ValueError(msg)
