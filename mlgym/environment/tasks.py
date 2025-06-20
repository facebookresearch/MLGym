"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Tasks for Gymnasium environments.

This module provides the base class for defining ML tasks and a set of concrete task classes
for different types of ML tasks. It handles task configuration, evaluation, baseline execution,
and submission processing for various ML task types including CSV submissions, model submissions,
and language model tasks.

"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import yaml
from simple_parsing.helpers.fields import field
from simple_parsing.helpers.serialization import encode
from simple_parsing.helpers.serialization.serializable import (
    FrozenSerializable,
    Serializable,
)

from mlgym import CONFIG_DIR
from mlgym.utils.extras import multiline_representer
from mlgym.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

AGENT_LONG_ACTION_TIMEOUT = int(os.getenv("MLGYM_AGENT_LONG_TIMEOUT", "3600"))


class SubmissionNotFoundError(Exception):
    """
    Exception raised when a submission file is not found.
    """

    pass


class EvaluationFormatError(Exception):
    """
    Exception raised when evaluation output is not in valid JSON format.
    """

    pass


@dataclass(frozen=True)
class SplitConfig(FrozenSerializable):
    """
    Configuration for a dataset split.
    """

    name: str  # split name
    file_regex: str  # regex to match files in the split


@dataclass(frozen=True)
class DatasetConfig(FrozenSerializable):
    """
    Configuration for a dataset.
    """

    name: str  # name of the datasets
    description: str  # description of the dataset format
    # local path or huggingface repo id of the dataset
    # local path should be relative to the REPO_ROOT
    data_path: str
    # indicates if dataset files are stored locally. If true, data_path must point to a valid filesystem directory
    is_local: bool = False
    train_split: SplitConfig | None = None
    valid_split: SplitConfig | None = None
    test_split: SplitConfig | None = None


# FIXME: All RUF009 should be resolved as part of issue #19.
@dataclass
class TaskConfig(Serializable):
    """
    Configuration for a MLGym task. A task can be tied to a single dataset or multiple datasets.
    """

    id: str  # text identifier for the task. Will be used to register the task with the gym environment
    name: str  # name of the task
    description: str  # description/goal of the task
    # list of paths to the dataset config files
    # paths should be relative to the CONFIG_DIR
    dataset_configs: list[str] = field(default_factory=list)  # noqa: RUF009

    _datasets: list[DatasetConfig] = field(default_factory=list, init=False)  # noqa: RUF009
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
                datasets.append(DatasetConfig.load_yaml(dataset_config_path))
            object.__setattr__(self, "_datasets", datasets)

        if self.sample_submission is not None and not Path(self.sample_submission).exists():
            msg = f"Sample submission path provided but file not found at {self.sample_submission}"
            raise FileNotFoundError(msg)

        # check if requirements_path is provided if use_generic_conda is False
        if not self.use_generic_conda and (self.requirements_path is None or not Path(self.requirements_path).exists()):
            msg = "Requirements path must be provided if use_generic_conda is False"
            raise ValueError(msg)


class AbstractMLTaskMeta(type):
    """
    Metaclass for ML tasks that maintains a registry of task types.

    Provides automatic registration of task classes to enable lookup by name.
    All task classes except the base AbstractMLTask are added to the registry.

    Attributes:
        _registry (dict): Maps task class names to their implementations
    """

    _registry: ClassVar = {}

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        """
        Create new task class and add to registry.

        Args:
            name (str): Name of the task class
            bases (tuple): Base classes
            attrs (dict): Class attributes

        Returns:
            type: New task class
        """
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != "AbstractMLTask":
            cls._registry[name] = new_cls
        return new_cls


class AbstractMLTask(metaclass=AbstractMLTaskMeta):
    """
    Abstract base class for defining ML tasks.

    Provides core functionality for task initialization, evaluation, and baseline
    execution. Specific task types should inherit from this class and implement
    the evaluate method.
    """

    def __init__(
        self,
        seed: int,
        args: TaskConfig,
        task_workspace: str,
        _communicate: Callable,
        _communicate_with_handling: Callable,
    ) -> None:
        """
        Initialize the task.

        Args:
            seed (int): Random seed for reproducibility
            args (TaskConfig): Task configuration
            task_workspace (str): Path to task workspace
            _communicate (Callable): Function for container communication
            _communicate_with_handling (Callable): Function for error-handled communication
        """
        self.seed = seed
        self.args = args
        self.task_workspace = task_workspace
        self._communicate = _communicate
        self._communicate_with_handling = _communicate_with_handling

        self.random = np.random.default_rng(self.seed)
        self.logger = get_logger("MLGymTask")

    @classmethod
    def get(cls, name: str) -> type[AbstractMLTask]:
        """
        Get a task class by name from the registry.

        Args:
            name (str): Name of task class to retrieve

        Returns:
            type[AbstractMLTask]: Task class

        Raises:
            ValueError: If task class not found in registry
        """
        try:
            return cls._registry[name]  # type: ignore
        except KeyError as e:
            msg = f"Task class {name} not found. Please check the task_entrypoint property in the TaskConfig."
            raise ValueError(msg) from e

    def update_baseline_scores(self) -> bool:
        """
        Update baseline scores by executing baseline scripts.

        Returns:
            bool: True if baseline scores were updated, False otherwise
        """
        if not len(self.args.baseline_scores):
            metrics = self._execute_baseline()
            if metrics is not None:
                self.args.baseline_scores.append(metrics)
                return True
        return False

    @abstractmethod
    def evaluate(self) -> tuple[dict[str, Any], str]:
        """
        Evaluate the submission artifact and return scores.

        Returns:
            tuple[dict[str, Any], str]: Tuple containing:
                - Evaluation metrics dictionary
                - Path to submission artifact

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def _execute_baseline(self) -> dict[str, Any] | None:
        """
        Execute baseline scripts to get baseline scores.

        Returns:
            dict[str, Any] | None: Baseline metrics if successful, None otherwise
        """
        # FIXME: accept none as a valid baseline paths option. Issue #19.
        if self.args.baseline_paths is None:
            return None  # type: ignore
        baseline_paths = self._get_baseline_paths()
        if baseline_paths:
            self.logger.info(
                "Running baseline scripts to get baseline scores. This might take a while, sit back and drink some water 🧘"
            )

            # ! We only support one baseline path for now.
            path = baseline_paths[0]
            self._communicate_with_handling(
                f"python {path}",
                timeout_duration=self.args.training_timeout,
                error_msg=f"Failed to run baseline script {path}",
            )

            metrics, _ = self.evaluate()
            return metrics
        else:
            self.logger.info("No baseline scripts provided. Skipping baseline execution.")
            return None

    def setup(self) -> None:
        """
        Set up the task environment.

        Initializes task description and timeout settings.
        """
        self.args.description = self.args.description.format(dataset_docs=self._generate_dataset_docs())
        self.args.training_timeout = (
            AGENT_LONG_ACTION_TIMEOUT if self.args.training_timeout is None else self.args.training_timeout
        )

    def _generate_dataset_docs(self) -> str:
        """
        Generate documentation for task datasets.

        Returns:
            str: Formatted dataset documentation string
        """
        docs = ""
        for dataset in self.args._datasets:
            data_name = dataset.name.upper()
            docs += f"{data_name}:\n{dataset.description}\n\n"

        return docs

    def _get_baseline_paths(self) -> list[str]:
        """
        Get paths to baseline scripts in container.

        Returns:
            list[str]: List of baseline script paths
        """
        return [str(Path(self.task_workspace) / path) for path in self.args.baseline_paths]

    def _get_evaluation_paths(self) -> list[str]:
        """
        Get paths to evaluation scripts in container.

        Returns:
            list[str]: List of evaluation script paths
        """
        return [str(Path(self.task_workspace) / path) for path in self.args.evaluation_paths]


# DEFINE NEW TASK CLASSES BELOW THIS LINE


class CSVSubmissionTasks(AbstractMLTask):
    """
    Task class for submissions in CSV format.

    Handles tasks where the agent submits results in a CSV file.
    Includes validation against sample submission format if provided.
    """

    def evaluate(self) -> tuple[dict[str, Any], str]:
        """
        Evaluate a CSV submission file.

        Returns:
            tuple[dict[str, Any], str]: Tuple containing:
                - Evaluation metrics dictionary
                - Path to submission file

        Raises:
            SubmissionNotFoundError: If no submission file found
            EvaluationFormatError: If evaluation output not in valid JSON format
        """
        submission = self._get_submission_file()
        if submission is None:
            msg = "No submission file found. Please make sure that your code produces a submission file."
            raise SubmissionNotFoundError(msg)

        evaluation_paths = self._get_evaluation_paths()

        # ! We only support one evaluation path for now.
        eval_script = evaluation_paths[0]
        output = self._communicate(
            f"python {eval_script} --submission_file {submission}", timeout_duration=self.args.training_timeout
        )

        # Parse the output as json
        try:
            metrics = json.loads(output)
        except json.JSONDecodeError as e:
            msg = f"Failed to decode metrics from evaluation script. Output:\n---\n{output}\n---\nPlease make sure the evaluation script prints the metrics in json format."
            raise EvaluationFormatError(msg) from e

        return metrics, submission

    ######## PRIVATE METHODS ########
    def _get_submission_file(self) -> str | None:
        """
        Find the submission file in workspace directory.

        Returns:
            str | None: Path to submission file if found, None otherwise
        """
        files = self._communicate(f"ls {self.task_workspace}").strip().split("\n")

        if "submission.csv" in files:
            return f"{self.task_workspace}/submission.csv"
        else:
            return None


class ModelSubmissionTasks(CSVSubmissionTasks):
    """
    Task class for model artifact submissions.

    Handles tasks where the agent submits a trained model.
    Supports model checkpoints and configuration files.
    """

    def evaluate(self) -> tuple[dict[str, Any], str]:
        """
        Evaluate a model submission.

        Returns:
            tuple[dict[str, Any], str]: Tuple containing:
                - Evaluation metrics dictionary
                - Path to submission file

        Raises:
            SubmissionNotFoundError: If no valid submission found
            EvaluationFormatError: If evaluation output not in valid JSON format
        """
        submission = self._get_submission_file()
        if submission is None:
            msg = "No valid submission artefacts found. Please make sure that your code produces a checkpoints folder."
            raise SubmissionNotFoundError(msg)

        evaluation_paths = self._get_evaluation_paths()

        # ! We only support one evaluation path for now.
        eval_script = evaluation_paths[0]
        output = self._communicate(
            f"python {eval_script} --config_fname {submission}", timeout_duration=self.args.training_timeout
        )

        # Parse the output as json
        try:
            json_line = next(line for line in output.split("\n") if line.strip().startswith("{"))
            metrics = json.loads(json_line)
        except (StopIteration, json.JSONDecodeError) as e:
            msg = f"Failed to decode metrics from evaluation script. Output:\n---\n{output}\n---\nPlease make sure the evaluation script prints the metrics in json format."
            raise EvaluationFormatError(msg) from e

        return metrics, submission

    def _get_submission_file(self) -> str | None:
        """
        Get the model submission file/folder.

        Returns:
            str | None: Path to submission if found, None otherwise
        """
        submission = None
        config_files = self._communicate(f"find {self.task_workspace} -type f -name '*.yaml'").strip().split("\n")
        if len(config_files):
            submission = config_files[0]

        return submission


class LMSubmissionTasks(CSVSubmissionTasks):
    """
    Task class for language model submissions.

    Handles tasks specific to language models, including GPU detection
    and distributed training support.
    """

    def setup(self) -> None:
        """
        Set up the language model task environment.

        Initializes task description, timeout settings, and detects GPU availability.
        """
        self.args.description = self.args.description.format(dataset_docs=self._generate_dataset_docs())
        self.args.training_timeout = (
            AGENT_LONG_ACTION_TIMEOUT if self.args.training_timeout is None else self.args.training_timeout
        )
        try:
            gpu_count = self._communicate("nvidia-smi --list-gpus | wc -l").strip()
            self.num_gpus = int(gpu_count)
        except:
            self.num_gpus = 0

    def _execute_baseline(self) -> dict[str, Any] | None:
        """
        Execute baseline scripts with distributed training support.

        Returns:
            dict[str, Any] | None: Baseline metrics if successful, None otherwise
        """
        # FIXME: accept none as a valid baseline paths option. Issue #19.
        if self.args.baseline_paths is None:
            return None  # type: ignore
        baseline_paths = self._get_baseline_paths()
        if baseline_paths:
            self.logger.info(
                "Running baseline scripts to get baseline scores. This might take a while, sit back and drink some water 🧘"
            )
            # ! We only support one baseline path for now.
            path = baseline_paths[0]
            print(f"Running baseline script {path} with {self.num_gpus} GPUs")
            self._communicate_with_handling(
                f"torchrun --nproc_per_node={self.num_gpus} --standalone {path}",
                timeout_duration=self.args.training_timeout,
                error_msg=f"Failed to run baseline script {path}",
            )

            metrics, _ = self.evaluate()
            return metrics
        else:
            self.logger.info("No baseline scripts provided. Skipping baseline execution.")
            return None

    def evaluate(self) -> tuple[dict[str, Any], str]:
        """
        Evaluate a language model submission.

        Returns:
            tuple[dict[str, Any], str]: Tuple containing:
                - Evaluation metrics dictionary
                - Empty string as submission path

        Raises:
            EvaluationFormatError: If evaluation output not in valid JSON format
        """

        evaluation_paths = self._get_evaluation_paths()

        # ! We only support one evaluation path for now.
        eval_script = evaluation_paths[0]
        output = self._communicate(
            f"torchrun --nproc_per_node={self.num_gpus} --standalone {eval_script}",
            timeout_duration=self.args.training_timeout,
        )

        # Parse the output as json
        try:
            json_line = next(line for line in output.split("\n") if line.strip().startswith("{"))
            metrics = json.loads(json_line)
        except (StopIteration, json.JSONDecodeError) as e:
            msg = f"Failed to decode metrics from evaluation script. Output:\n---\n{output}\n---\nPlease make sure the evaluation script prints the metrics in json format."
            raise EvaluationFormatError(msg) from e

        return metrics, ""


class PythonSubmissionTasks(AbstractMLTask):
    """
    Task class for Python code submissions.

    Handles tasks where the agent submits Python code files.
    Supports direct execution and evaluation of Python scripts.
    """

    def evaluate(self) -> tuple[dict[str, Any], str]:
        """
        Evaluate a Python code submission.

        Returns:
            tuple[dict[str, Any], str]: Tuple containing:
                - Evaluation metrics dictionary
                - Path to submission file

        Raises:
            EvaluationFormatError: If evaluation output not in valid JSON format
        """
        evaluation_paths = self._get_evaluation_paths()
        submission = f"{self.task_workspace}/target.py"

        # ! We only support one evaluation path for now.
        eval_script = evaluation_paths[0]
        output = self._communicate(f"python {eval_script}", timeout_duration=self.args.training_timeout)

        # Parse the output as json
        try:
            metrics = json.loads(output)
        except json.JSONDecodeError as e:
            msg = f"Failed to decode metrics from evaluation script. Output:\n---\n{output}\n---\nPlease make sure the evaluation script prints the metrics in json format."
            raise EvaluationFormatError(msg) from e

        return metrics, submission

    ######## PRIVATE METHODS ########
    def _get_submission_file(self) -> str | None:
        """
        Find the Python submission file.

        Returns:
            str | None: Path to submission file if found, None otherwise
        """
        files = self._communicate(f"ls {self.task_workspace}").strip().split("\n")

        if "target.csv" in files:
            return f"{self.task_workspace}/submission.csv"
        else:
            return None
