"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Core environment implementation for the MLGym framework.

This module provides the main environment class and supporting functionality for
running machine learning tasks. It handles container management, task execution,
file operations, and agent interactions. The environment supports various task types
and manages the workspace, evaluation, and resource limits.

Adapted from SWE-Agent/sweagent/environment/swe_env.py
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import docker
import docker.models.containers
import gymnasium as gym
import yaml
from docker.errors import DockerException, NotFound

from mlgym import CONFIG_DIR, REPO_ROOT
from mlgym.configs.dataset import BaseDatasetConfig
from mlgym.configs.environment import BaseEnvironmentConfig
from mlgym.configs.task import BaseTaskConfig
from mlgym.environment.spaces import Unicode
from mlgym.environment.tasks import (
    AbstractMLTask,
    EvaluationFormatError,
    SubmissionNotFoundError,
)
from mlgym.environment.utils import (
    PROCESS_DONE_MARKER_END,
    PROCESS_DONE_MARKER_START,
    NoOutputTimeoutError,
    copy_anything_from_container,
    copy_anything_to_container,
    copy_file_to_container,
    get_container,
    image_exists,
    read_with_timeout,
    read_with_timeout_pid,
)
from mlgym.types import AgentInfo
from mlgym.utils.extras import multiline_representer
from mlgym.utils.log import get_logger

if TYPE_CHECKING:
    import subprocess


# ? I DON'T THINK WE NEED THIS
ENV_LONG_TIMEOUT = int(os.getenv("MLGYM_ENV_TIMEOUT", "500"))
# for normal agent calls such as editing, ls, cd etc.
AGENT_SHORT_ACTION_TIMEOUT = int(os.getenv("MLGYM_AGENT_SHORT_TIMEOUT", "25"))
# for training calls
AGENT_LONG_ACTION_TIMEOUT = int(os.getenv("MLGYM_AGENT_LONG_TIMEOUT", "3600"))
AGENT_ACTION_NO_OUTPUT_TIMEOUT = int(os.getenv("MLGYM_AGENT_ACTION_NO_OUTPUT_TIMEOUT", str(AGENT_LONG_ACTION_TIMEOUT)))
# global workspace directory
GLOBAL_WORKSPACE_DIR = Path(os.getenv("MLGYM_WORKSPACE_PATH", str(REPO_ROOT / "workspace")))

EDIT_PATTERN = re.compile(r"^edit\s+(\d+):(\d+)(?:\n|$|\s)", re.MULTILINE)
TRAIN_COMMANDS = ["torchrun", "python", "python3", "accelerate", "deepspeed"]

yaml.add_representer(str, multiline_representer)
yaml.representer.SafeRepresenter.add_representer(str, multiline_representer)  # type: ignore


class MLGymEnv(gym.Env):
    """
    Main environment class for MLGym framework.

    Implements the OpenAI Gym interface for machine learning tasks. Manages the
    container lifecycle, task execution, and agent interactions.

    Attributes:
        metadata (dict): Environment metadata including render modes
        name (str): Environment identifier
        cached_image_prefix (str): Prefix for cached task images
    """

    # NOTE: cannot fix the classvar ruff issue as it is gym.env variable
    metadata: dict[str, Any] = {"render_modes": ["human"]}  # noqa: RUF012
    name = "mlgym_main"
    # ! TODO: Caching should be handled at the benchmark level and not task level.
    cached_image_prefix = "mlgym-task-env-"

    def __init__(
        self,
        args: BaseEnvironmentConfig,
        devices: list[str],
        render_mode: str | None = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            args (EnvironmentArguments): Environment configuration
            devices (list[str]): List of devices to use (e.g., GPUs)
            render_mode (str | None): Rendering mode. Defaults to None

        Raises:
            AssertionError: If render_mode is invalid
        """
        super().__init__()
        t0 = time.perf_counter()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.args: BaseEnvironmentConfig = args
        self.task_args: BaseTaskConfig = args.task  # type: ignore
        self.communicate_output: str | None = None
        self.container_name: str | None = args.container_name
        self.logger = get_logger("MLGymEnv")
        self.persistent = args.container_name is not None
        self.container_mounts = args.container_mounts
        self.returncode: None | int = None
        self.devices: list[str] = devices

        # * set bash path to homebrew bash if running in local mode
        self.bash_path = "/bin/bash"

        if not self.args.verbose:
            self.logger.disabled = True

        self.task: AbstractMLTask | None = None
        self.current_step: int = 0
        assert isinstance(self.task_args, BaseTaskConfig)
        self.task_entrypoint: type[AbstractMLTask] = args.get_task_class()
        self.render_mode = render_mode

        self.seed: int = args.seed
        self.max_steps: int = args.max_steps
        self.workspace_dir: Path = Path("/home/agent")
        self.task_workspace: Path = Path(f"{self.workspace_dir}/workspace")
        self.memory_path: Path = Path(f"{self.workspace_dir}/memory.json")
        self.logger.info(f"Global Workspace directory set to: {self.workspace_dir}")
        self.logger.info(f"Task workspace directory set to: {self.task_workspace}")

        # TODO: this action space is a placeholder maybe.
        self.action_space = Unicode(min_length=0, max_length=1000)  # TODO: read max_length from a global variable

        # TODO: this observation space is a placeholder. We need to define the actual observation space.
        self.observation_space = Unicode(min_length=0, max_length=10000)

        # this should be replaced b a docker container object if it is active
        self.image_name = args.image_name
        self.container_type = args.container_type
        self.container: subprocess.Popen | None = None
        self.container_obj: docker.models.containers.Container | None = None
        self._reset_container()

        self.clean_multi_line_functions = lambda x: x

        #! TODO: environment initialization is not complete yet. Move this line to appropriate location.
        self.logger.debug("Environment initialization took %.2f seconds", time.perf_counter() - t0)

    def _get_cached_task_image_name(self) -> str:
        """
        Generate name for cached task image.

        Returns:
            str: Unique image name based on task ID and setup
        """
        assert self.task_args is not None
        inputs: list[str] = [
            self.task_args.id,
            self.args.environment_setup or "no_setup",
        ]
        tag = hashlib.sha256("".join(inputs).encode()).hexdigest()[:50]
        return f"{self.cached_image_prefix}{tag}"

    def reset_container(self) -> None:
        """
        Reset the container to initial state.

        Closes existing container and initializes a new one.
        """
        self.close()
        self.container = None
        self.container_obj = None
        self._reset_container()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to initial state. seed and options are optional but can be used to control the randomness depending on the environment.

        Resets container, workspace, and task state. Installs required dependencies
        and sets up the task environment.

        Args:
            seed (int | None): Seed for the environment
            options (dict[str, Any] | None): Optional arguments for the environment

        Returns:
            dict[str, Any]: Initial observation

        Raises:
            RuntimeError: If container setup fails
        """
        self.logger.info("Resetting the environment")
        super().reset(seed=seed, options=options)
        self.np_random = None  # type: ignore

        self.current_step = 0
        self.task = self.task_entrypoint(
            seed=self.seed,
            args=copy.deepcopy(self.task_args),
            task_workspace=str(self.task_workspace),
            _communicate=self.communicate,
            _communicate_with_handling=self.communicate_with_handling,
        )

        ### Reset Container ###

        if self.container_obj is not None and self.args.cache_task_images:
            cached_image = self._get_cached_task_image_name()
            if image_exists(cached_image):
                self.logger.info(f"Restore environment from cached image {cached_image}")
                self.close()  # stop current container
                self._init_container(cached_image=cached_image)
                self.communicate("export $(xargs </.env)")
                envs = self.communicate("env")
                self.logger.debug(f"Environment variables restored from the image:\n{envs}\n")
                return {"observation": None}, {}
            else:
                self.logger.info(f"Cached image {cached_image} not found, rebuilding task environment...")

        self._setup_workspace()
        self._reset_environment_variables()

        # setup the task
        self.task.setup()

        # install build essnetials on linux
        assert self.container_obj is not None  # mypy
        system = self.communicate("uname -s").strip().lower()
        arch = self.communicate("uname -m").strip().lower()
        if system == "linux" and arch == "x86_64":
            self.container_obj.exec_run("apt update; apt install build-essential -y", user="root")

        # Install mypy for linting purposes
        self.communicate_with_handling("pip install flake8", error_msg="Failed to install flake8 (lint library)")
        # self.container_obj.exec_run("pip install flake8")

        # Activate correct conda environment
        if self.task.args.use_generic_conda:
            self.communicate_with_handling(
                "conda activate mlgym_generic",
                error_msg="Failed to activate generic conda environment",
            )
        else:
            self.install_and_activate_env()

        if self.args.cache_task_images:
            envs = self.communicate("env")
            self.logger.debug(f"Environment variables to save:\n{envs}\n")
            self.communicate("env >> /.env")
            assert self.container_obj is not None  # mypy
            self.container_obj.commit(cached_image)
            self.logger.info(f"Container with environment {self.container_obj.id} cached as image {cached_image}")

        # change current directory to the task workspace
        self.communicate(f"cd {self.task_workspace}")

        # this executes the baseline for the first time
        updated = self.task.update_baseline_scores()

        # We need to clean up the workspace after baseline is done here. Remove everything in the workspace except starter code.
        workspace_files = self.communicate(f"ls {self.task_workspace}").split("\n")
        workspace_files = [file.strip("*/") for file in workspace_files if file]
        starter_files = [Path(self.task_workspace / file).name for file in self.task.args.starter_code]
        for file in workspace_files:
            if file not in starter_files and file != "data":
                self.communicate(f"rm -rf {self.task_workspace / file}")

        # cache baseline scores to the task config file
        if self.args.cache_baseline_scores and updated:
            assert self.args.task_config_path is not None
            self.task_args.baseline_scores = self.task.args.baseline_scores
            with open(CONFIG_DIR / self.args.task_config_path, "w") as f:
                self.task_args.dump_yaml(f, sort_keys=False)

        # TODO: what info should we return here?
        return {"observation": None}, {}

    # FIXME: Fix the step method signature and return type to match the Gymnasium class and reduce complexity.
    def step(self, action: str) -> tuple[str | None, float, bool, dict[str, Any]]:  # type: ignore # noqa: C901
        """
        Execute one step of the environment.

        Processes the agent's action and returns the result.

        Args:
            action (str): Command to execute in the environment

        Returns:
            tuple[str | None, float, bool, AgentInfo]: Tuple containing:
                - observation: Output from action or None
                - reward: Always 0 in current implementation
                - done: Whether episode is complete
                - info: Additional information about the step

        Raises:
            TimeoutError: If action execution times out
            RuntimeError: If command execution fails
            BrokenPipeError: If container communication fails
        """
        # ? MARK: Maybe we don't need this.
        info: AgentInfo = AgentInfo()
        self.current_step += 1

        observation = ""
        if self.current_step > self.max_steps:
            self.logger.warning(f"Max Steps reached: {self.max_steps}")
            return self._evaluate_with_error_handling(info, "max_steps")

        action = action.strip()
        if action == "skip":
            observation = "Skipped"
            info["exit_status"] = "skipped"
            return observation, 0, True, info
        if action == "exit_forfeit":
            observation = "Exited (exit_forfeit)"
            info["exit_status"] = action
            return observation, 0, True, info
        if action in {
            "exit_context",
            "exit_cost",
            "exit_error",
            "exit_format",
            "exit_api",
        }:
            return self._evaluate_with_error_handling(info, action)

        # check if edit action is malformed
        if re.match(EDIT_PATTERN, action) and "end_of_edit" not in action:
            observation = (
                "Malformed edit command. Please ensure you use the `end_of_edit` keyword to mark the end of your edit."
            )
            return observation, 0, False, info

        # Attempt to run action in container
        observation = ""
        assert self.task is not None
        # FIXME: Remove set timeout code from task and include it in the environment. The task accepts an int timeout.
        timeout = (
            self.task.args.training_timeout
            if any(command in action for command in TRAIN_COMMANDS)
            else AGENT_SHORT_ACTION_TIMEOUT
        )

        try:
            observation = self.communicate(
                input=action,
                timeout_duration=timeout,  # type: ignore
                no_output_timeout_duration=AGENT_ACTION_NO_OUTPUT_TIMEOUT,
                set_last_action=True,
            )
        except TimeoutError as e:
            try:
                observation += e.args[1] if len(e.args) > 1 else ""
                observation += self.interrupt()
                observation += "\nEXECUTION TIMED OUT"
                observation += (
                    f" BECAUSE NO OUTPUT WAS PRODUCED FOR MORE THAN {AGENT_ACTION_NO_OUTPUT_TIMEOUT} SECONDS.\nPLEASE REFINE YOUR RUNNING COMMAND SO IT WILL PRODUCE OUTPUT IN THE SPECIFIED TIME FRAME."
                    if isinstance(e, NoOutputTimeoutError)
                    else f" BECAUSE THE COMMAND WAS RUNNING FOR MORE THAN {timeout} SECONDS."
                )
            except RuntimeError as er:
                observation += er.args[1] if len(er.args) > 1 else ""
                observation += "\nEXECUTION TIMED OUT AND INTERRUPT FAILED. RESTARTING PROCESS."
                info["exit_status"] = "early_exit"
                self.logger.warning(f"Failed to interrupt container: {e}\nRESTARTING PROCESS.")
                self.reset_container()
                return observation, 0, True, info
        except RuntimeError as e:
            observation += e.args[1] if len(e.args) > 1 else ""
            # ? why do we need to restart the process if commands fail to execute? This seems more related to enigma.
            observation += "\nCOMMAND FAILED TO EXECUTE. RESTARTING PROCESS."
            info["exit_status"] = "early_exit"
            self.logger.warning(f"Failed to execute command: {e}\nRESTARTING PROCESS.")
            self.reset_container()
            return observation, 0, True, info
        except BrokenPipeError:
            observation += "\nBROKEN PIPE ERROR. RESTARTING PROCESS."
            info["exit_status"] = "early_exit"
            self.logger.exception("Broken pipe error: \nRESTARTING PROCESS.")
            self.reset_container()
            return observation, 0, True, info
        except Exception:
            observation += "\nEXECUTION FAILED OR COMMAND MALFORMED"
            self.logger.exception("Unknown exception")

        # Check for submission file and end episode if `submit` keyword found
        # If submission file not found, return to agent to make sure it wants to submit.
        validation_check = self.get_validation(observation)
        submission_check = self.get_submission(observation)
        if validation_check is not None:
            self.logger.info("Validating current solution")
            return self._evaluate_with_error_handling(info, "validate")
        if submission_check is not None:
            return self._evaluate_with_error_handling(info, "submit")

        return observation, 0, False, info

    # FIXME: Reduce complexity
    def close(self) -> None:  # noqa: C901
        """
        Clean up environment resources.

        Saves memory if enabled, stops and removes container unless persistent.
        Handles various cleanup scenarios based on container state.

        Raises:
            KeyboardInterrupt: If interrupt received during cleanup
        """
        # TODO: copy the contents of the workspace_dir on the container in the logs directory on host.
        # TODO: remove the commands directory from the workspace_dir when working in local mode.

        self.logger.info("Beginning environment shutdown...")
        try:
            if self.args.memory_enabled:
                self.logger.info("Attempting to save memory before exiting container")
                assert self.container_obj is not None
                copy_anything_from_container(
                    container=self.container_obj,
                    container_type=self.container_type,
                    host_path=str(self.task_args.memory_path),
                    container_path=str(self.memory_path),
                )

            self.communicate(input="exit")

        except KeyboardInterrupt:
            raise
        except Exception:
            self.logger.warning("Errors when exiting container", exc_info=True)
        assert self.container is not None

        self.container.terminate()
        if self.container_obj is None:
            pass
        elif self.persistent:
            # stopping is Podman specific but doesn't hurt to include
            # https://stackoverflow.com/a/32428199/
            # Want to avoid https://github.com/princeton-nlp/SWE-agent/issues/496
            # Note that container_obj.status might not be updated throughout the container
            # lifecycle, so let's get the container_obj again
            assert self.container_name
            try:
                self.container_obj = docker.from_env().containers.get(self.container_name)
            except Exception as e:
                self.logger.warning(f"Failed to get fresh container object: {e}", exc_info=True)
            if self.container_obj.status not in {
                "paused",
                "exited",
                "dead",
                "stopping",
            }:
                try:
                    self.container_obj.pause()
                except Exception:
                    self.logger.warning("Failed to pause container.", exc_info=True)
                except KeyboardInterrupt:
                    raise
                else:
                    self.logger.info("Agent container paused")
            else:
                self.logger.info(f"Docker container status: {self.container_obj.status}")
        else:
            try:
                self.container_obj.remove(force=True)
            except KeyboardInterrupt:
                raise
            except NotFound:
                # We already tried to exit the container, so it's actually good if
                # it's not found
                pass
            except Exception:
                self.logger.warning("Failed to remove container", exc_info=True)
            else:
                self.logger.info("Agent container stopped")

    def install_and_activate_env(self) -> None:
        """
        Set up conda environment for the task.

        Creates and activates a conda environment based on task requirements.
        Installs all dependencies specified in the requirements file.

        Raises:
            RuntimeError: If environment creation or package installation fails
        """
        assert self.task is not None
        assert self.container_obj is not None
        self.logger.info("Installing and activating custom conda environment.")

        env_name = self.task.args.id

        # Check if the environment already exists
        output = self.communicate(f"conda env list | grep {env_name}", timeout_duration=20)
        if output != "":
            self.logger.info(f"Conda environment {env_name} already exists. Activating it.")
            self.communicate_with_handling(
                f"conda activate {env_name}",
                error_msg=f"Failed to activate conda environment {env_name}",
            )
            return

        # if environment does not exist, create it
        copy_anything_to_container(
            self.container_obj,
            self.container_type,
            str(self.task_args.requirements_path),
            "/home/agent/task_conda_requirements.txt",
        )
        self.communicate_with_handling(
            f"conda create -n {env_name} python=3.11 -y",
            error_msg="Failed to create task conda environment",
            timeout_duration=600,
        )

        # activate the conda environment
        self.communicate_with_handling(
            f"conda activate {env_name}",
            error_msg=f"Failed to activate conda environment {env_name}",
        )

        self.communicate_with_handling(
            "python -m pip install -r /home/agent/task_conda_requirements.txt",
            error_msg=f"Failed to install requirements for conda env {env_name}",
            timeout_duration=600,
        )

    def get_submission(self, output: str) -> str | None:
        """
        Extract submission from output.

        Args:
            output (str): Output containing submission

        Returns:
            str | None: Extracted submission or None if not found
        """
        pattern = r"\<\<SUBMISSION\|\|(.*)\|\|SUBMISSION\>\>"
        match = re.search(pattern, output, re.DOTALL)
        if match is None:
            return None
        return match.group(1)

    def get_validation(self, output: str) -> str | None:
        """
        Extract validation from output.

        Args:
            output (str): Output containing validation

        Returns:
            str | None: Extracted validation or None if not found
        """
        pattern = r"\<\<VALIDATION\|\|(.*)\|\|VALIDATION\>\>"
        match = re.search(pattern, output, re.DOTALL)
        if match is None:
            return None
        return match.group(1)

    # ! TODO: for training/evaluation commands, we need to support showing the progress bars on the host machine.
    def communicate(
        self,
        input: str,
        timeout_duration: float = 25,
        no_output_timeout_duration: float | None = None,
        *,
        set_last_action: bool = False,
    ) -> str:
        """
        Send input to container and get output.

        Args:
            input (str): Command to execute
            timeout_duration (int | float): Maximum execution time. Defaults to 25
            no_output_timeout_duration (int | float | None): Maximum time without output
            set_last_action (bool): Whether to set LAST_ACTION env var. Defaults to False

        Returns:
            str: Command output

        Raises:
            RuntimeError: If command execution fails
        """
        if no_output_timeout_duration is None:
            no_output_timeout_duration = timeout_duration
        if input.strip() != "exit":
            # self.logger.log(logging.TRACE, "Input:\n%s", input)  # type: ignore
            output, valid = self._check_syntax(input)
            if not valid:
                return output  # shows syntax errors
            output = self._communicate(
                input,
                timeout_duration=timeout_duration,
                no_output_timeout_duration=no_output_timeout_duration,
            )
            self.logger.log(logging.TRACE, "Output:\n%s", output)  # type: ignore

            self.communicate_output = output
            if set_last_action:
                # Cannot merge this with last command, because of multiline command
                # handling.
                last_action_string = shlex.quote(input.strip())
                input = f"export LAST_ACTION={last_action_string}"
                self._communicate(input, timeout_duration=5, no_output_timeout_duration=5)
            return output
        else:
            self.container.terminate()  # type: ignore
            self.returncode = 0
            self.communicate_output = ""
            return ""

    def communicate_with_handling(self, input: str, error_msg: str, timeout_duration: float = 25) -> str:
        """
        Execute command with error handling.

        Args:
            input (str): Command to execute
            error_msg (str): Error message if command fails
            timeout_duration (int | float): Maximum execution time. Defaults to 25

        Returns:
            str: Command output

        Raises:
            RuntimeError: If command returns non-zero exit code
        """
        logs = self.communicate(input, timeout_duration=timeout_duration)
        if self.returncode != 0:
            # self.logger.error(f"{error_msg}: {logs}")
            self.close()
            msg = f"{error_msg}: {logs}"
            raise RuntimeError(msg)
        return logs

    # /borrowed from swe-agent

    def add_commands(self, commands: list[dict]) -> None:
        """
        Add custom commands to container.

        Args:
            commands (list[dict]): List of command definitions

        Raises:
            ValueError: If command type is invalid
            RuntimeError: If command installation fails
        """
        for command in commands:
            name = command["name"]
            contents = command["contents"]

            # used to set the correct path for commands directory depending on container type
            assert self.container_obj is not None
            commands_dir = str(self.workspace_dir / "commands")
            copy_file_to_container(
                self.container_obj,
                self.container_type,
                contents,
                f"{commands_dir}/{name}",
            )
            if command["type"] == "source_file":
                self.communicate_with_handling(
                    f"source {commands_dir}/{name}",
                    error_msg=f"Failed to source {name}. If you meant to make a script,"
                    " start the file with a shebang (e.g. #!/usr/bin/env python).",
                )
            elif command["type"] == "script":
                assert self.container_obj is not None
                self.container_obj.exec_run(f"chmod +x {commands_dir}/{name}", user="root")
            elif command["type"] == "utility":
                # nothing to do for utility scripts
                pass
            else:
                msg = f"Invalid command type: {command['type']}"
                raise ValueError(msg)

    def get_available_actions(self) -> list[str]:
        """
        Get list of available actions.

        Returns:
            list[str]: List of available commands
        """
        return []

    def interrupt(self) -> str:
        """
        Send interrupt signal to running processes.

        Returns:
            str: Output buffer contents

        Raises:
            RuntimeError: If interrupt fails
        """

        assert self.container_obj is not None
        assert self.container is not None
        pids = self.get_pids()
        for pid, name in pids:
            # Sending signal several times ensures the the process is dead
            self.logger.warning(f"Killing process {pid}: {name}")
            for _ in range(3):
                self.container_obj.exec_run(f"kill -9 {pid}")
        observation = ""
        try:
            observation += read_with_timeout_pid(self.container, self.get_pids, 60)
        except TimeoutError:
            self.logger.exception("Timeout error while reading PIDs")
        try:
            # This is a workaround because of bash behaviour
            # when sometimes we get the prints of Killed after we press some "Enter" in stdin
            self.communicate(input="echo 'interrupted'", timeout_duration=5)
            output = self.communicate(input="echo 'interrupted'", timeout_duration=5)
            assert output.strip().endswith("interrupted"), "container health check failed"
        except TimeoutError as e:
            msg = "Failed to interrupt container"
            raise RuntimeError(msg) from e
        return observation

    def get_pids(self, all_pids: bool = False) -> list[tuple[int, str]]:
        """
        Get list of process IDs in container.

        Args:
            all_pids (bool): Whether to include all PIDs. Defaults to False

        Returns:
            list[tuple[int, str]]: List of (pid, process_name) tuples
        """
        assert self.container_obj is not None
        pids = self.container_obj.exec_run("ps -eo pid,comm,ppid --no-headers").output.decode().split("\n")
        pids = [x.split() for x in pids if x]
        if not all_pids:
            # Get just the PIDs of processes that are descendants of parent_pids and not others
            pids = [
                (x[0], x[1]) for x in pids if x[1] != "ps" and x[0] not in self.parent_pids and x[2] in self.parent_pids
            ]
        return pids

    ### HELPER FUNCTIONS ###

    def _evaluate_with_error_handling(self, info: AgentInfo, action: str) -> tuple[str, float, bool, AgentInfo]:  # noqa: C901
        """
        Handle submission evaluation with comprehensive error handling.

        Processes submission artifacts, validates results, and handles various
        error conditions during evaluation.

        Args:
            info (AgentInfo): Object to store evaluation results and status
            action (str): The exit status/action from environment

        Returns:
            tuple[str, float, bool, AgentInfo]: Tuple containing:
                - observation: Output message
                - reward: Always 0 in current implementation
                - done: Whether episode is complete
                - info: Updated AgentInfo with evaluation results

        Raises:
            SubmissionNotFoundError: If submission artifact not found
            EvaluationFormatError: If submission format is invalid
        """
        assert self.task is not None
        try:
            metrics, submission = self.task.evaluate()
            if action == "submit":
                self.logger.info(f"Evaluation score: {metrics}")
                info["exit_status"] = "submitted"
                info["submission"] = submission
                info["score"].append(metrics)
                observation = "Exited (submit)"
                return observation, 0, True, info
            elif action == "validate":
                self.logger.info(f"Evaluation score: {metrics}")
                info["score"].append(metrics)
                try:
                    baseline_sc = self.task_args.baseline_scores[0]
                except (AttributeError, IndexError):
                    msg = "NA (no baseline was there)"
                    observation = f"Your code produced a valid submission artefact at {submission}.\nBaseline Score: {msg}\nEvaluation Score: {metrics}".strip()
                else:
                    observation = f"Your code produced a valid submission artefact at {submission}.\nBaseline Score: {baseline_sc}\nEvaluation Score: {metrics}".strip()
                return observation, 0, False, info
            else:
                self.logger.warning(f"Autosubmitting the last valid submission artefact: {submission}")
                self.logger.info(f"Evaluation score: {metrics}")
                # ! TODO: Add step information along with the score.
                info["score"].append(metrics)
                # ! TODO: Standardize the exit statuses.
                info["exit_status"] = f"autosubmission ({action})"
                info["submission"] = submission
                observation = f"Exited ({action})"
                return observation, 0, True, info
        except SubmissionNotFoundError:
            if action == "validate":
                self.logger.exception("Submission artefact not found.")
                observation = "Submission artefact not found. You have to produce a valid submission artefact as described in the task description to validate your results."
                return observation, 0, False, info
            else:
                self.logger.warning("No submission artefact found. Exiting with score 0.")
                observation = f"Exited ({action})"
                info["exit_status"] = f"submission_not_found ({action})"
                return observation, 0, True, info
        except EvaluationFormatError as ef:
            # ! MARK: If there is a evaluation format error, agent cannot do anything, so we should exit.
            if action == "validate":
                observation = f"{ef.args[0]}"
                self.logger.exception("Evaluation format error")
                return observation, 0, False, info
            else:
                self.logger.exception("Evaluation format error")
                observation = f"Exited ({action})"
                info["exit_status"] = f"evaluation_format_error ({action})"
                return observation, 0, True, info
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if action == "validate":
                observation = f"Error: {e}"
                self.logger.exception("Error")
                return observation, 0, False, info
            else:
                observation = f"Exited ({action})"
                info["exit_status"] = f"{e.__class__.__name__}, {e}, ({action})"
                return observation, 0, True, info

    def _reset_container(self) -> None:
        """
        Reset container to initial state.

        Terminates existing container if present and initializes a new one.
        Sets up required scripts and configurations.

        Raises:
            RuntimeError: If container initialization fails
        """
        if self.container is not None:
            try:
                self.container.terminate()
            except KeyboardInterrupt:
                raise
            except:
                print("Failed to terminate container")
            else:
                print("Terminated container")
        self._init_container()
        self._init_scripts()

    def _init_container(self, cached_image: str | None = None) -> None:
        """
        Initialize container environment.

        Sets up Docker container with specified image and configuration.
        Handles both new and cached container images.

        Args:
            cached_image (str | None): Name of cached image to use. Defaults to None

        Raises:
            RuntimeError: If Docker daemon not running or container setup fails
        """
        image_name = self.image_name
        if cached_image is not None:
            image_name = cached_image
            self.logger.info(f"Using cached image: {image_name}")
        if self.persistent:
            assert self.container_name is not None
        else:
            self.container_name = self._get_container_name(image_name)
        self.container, self.parent_pids = get_container(
            self.container_name,
            image_name,
            container_type=self.container_type,
            persistent=self.persistent,
            container_mounts=self.container_mounts,
            devices=self.devices,
        )
        try:
            client = docker.from_env(timeout=600)
        except DockerException as e:
            if "Error while fetching server API version" in str(e):
                msg = "Docker is not running. Please start Docker and try again."
            else:
                msg = "Unknown docker exception occurred. Are you sure docker is running?"
            raise RuntimeError(msg) from e
        t0 = time.time()
        self.container_obj = None
        # ? Why are we using a hardcoded timeout of 60 seconds here?
        while time.time() - t0 < 60:
            try:
                assert self.container_name is not None
                self.container_obj = client.containers.get(self.container_name)
            except NotFound:
                self.logger.debug("Couldn't find container. Let's wait and retry.")
                time.sleep(1)
            else:
                break
        else:
            print(f"{self.persistent=}")
            available_containers = client.containers.list(all=True)
            available_containers_info = json.dumps([str(c.attrs) for c in available_containers], indent=2)
            print(available_containers_info)
            msg = "Failed to get container object."
            raise RuntimeError(msg)
        self.logger.info("🌱 Environment Initialized")

    def _get_container_name(self, image_name: str) -> str:
        """
        Generate unique container name.

        Creates a deterministic but unique name based on process ID, timestamp,
        and device configuration.

        Args:
            image_name (str): Base image name to include in container name

        Returns:
            str: Unique container name
        """
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id + "_".join(self.devices)
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        image_name_sanitized = image_name_sanitized + "-d_" + "_".join(self.devices)
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _init_scripts(self) -> None:
        """
        Initialize custom scripts and aliases in container.

        Sets up container-specific configurations and command aliases.
        Handles different container runtimes (Docker/Podman).
        """
        # ! MARK: Specific alias setup for podman on devgpu
        assert self.container_obj is not None
        self.communicate_with_handling(
            f"mkdir -p {self.workspace_dir!s}/commands",
            error_msg="Failed to create commands directory",
        )
        self.communicate_with_handling(
            f"touch {self.workspace_dir!s}/commands/__init__.py",
            error_msg="Failed to create __init__.py",
        )
        self.communicate_with_handling(
            f"export PATH=$PATH:{self.workspace_dir!s}/commands",
            error_msg="Failed to add commands directory to PATH",
        )
        if self.args.aliases_file is not None:
            # expand aliases in the container
            self.communicate_with_handling(
                "shopt -s expand_aliases",
                error_msg="Failed to expand aliases",
            )
            copy_anything_to_container(
                self.container_obj,
                self.container_type,
                self.args.aliases_file,
                str(self.workspace_dir / "commands/bash_aliases.sh"),
            )
            self.communicate_with_handling(
                f"source {self.workspace_dir!s}/commands/bash_aliases.sh",
                error_msg="Failed to source bash_aliases.sh",
            )

    def _reset_environment_variables(self) -> None:
        """
        Reset environment variables to initial state.

        Clears and reinitializes environment variables in container.
        Sets up basic environment configuration.

        Raises:
            RuntimeError: If environment variable setup fails
        """
        cmd = [
            'export CURRENT_FILE=""',
            "export CURRENT_LINE=0",
            "export SEARCH_RESULTS=()",
            "export SEARCH_FILES=()",
            "export SEARCH_INDEX=0",
        ]
        self.communicate_with_handling(
            input=" && ".join(cmd),
            error_msg="Failed to reset environment variables",
        )

    # borrowed from swe-agent
    def _communicate(
        self,
        input: str,
        timeout_duration: float = 25,
        no_output_timeout_duration: float = 25,
    ) -> str:
        """
        Low-level communication with container.

        Handles direct interaction with container process, including
        timeouts and output buffering.

        Args:
            input (str): Command to execute
            timeout_duration (int | float): Maximum execution time. Defaults to 25
            no_output_timeout_duration (int | float): Maximum time without output

        Returns:
            str: Command output

        Raises:
            TimeoutError: If execution exceeds timeout
            NoOutputTimeoutError: If no output received within timeout
            RuntimeError: If command execution fails
        """
        assert self.container is not None
        # Sleep to ensure that the exit code is in the last line
        # See https://github.com/princeton-nlp/SWE-agent/issues/595
        command_suffix = (
            f'EXITSTATUS="$?"; sleep 0.01; echo {PROCESS_DONE_MARKER_START}$EXITSTATUS{PROCESS_DONE_MARKER_END}\n'
        )
        try:
            self.returncode = None
            cmd = input if input.endswith("\n") else input + "\n"
            cmd += command_suffix
            os.write(self.container.stdin.fileno(), cmd.encode())  # type: ignore
            time.sleep(0.1)
            self.container.stdin.flush()  # type: ignore
        except BrokenPipeError as e:
            # traceback.print_exc()
            # self.logger.error("Failed to communicate with container. Check docker logs for more information.")
            msg = "Failed to communicate with container"
            raise RuntimeError(msg) from e

        try:
            buffer, exit_code = read_with_timeout(self.container, timeout_duration, no_output_timeout_duration)
        except Exception:
            msg = f"Read with timeout failed on input:\n---\n{input}\n---"
            self.logger.exception(msg)
            raise
        if exit_code == "$EXITSTATUS":
            # this sometimes happens if the command badly fails
            # for example if you just try to run python with no arguments
            # in this case, the error message is usually also garbage, so let's set
            # something new.
            # See https://github.com/princeton-nlp/SWE-agent/issues/630
            buffer = (
                "Unkknown error occurred when running the command. Please double check syntax "
                "and that you're not running an interactive command."
            )
            # self.logger.warning("Couldn't get real exit code. Setting it to 999")
            exit_code = "999"
        elif not exit_code.isdigit():
            msg = f"Failed to get exit code. Output:\n---\n{buffer}\n---"
            raise RuntimeError(msg)
        self.returncode = int(exit_code)
        return buffer

    def _check_syntax(self, input: str) -> tuple[str, bool]:
        """
        Check syntax of input command.

        Validates command syntax and returns any error messages.

        Args:
            input (str): Command to validate

        Returns:
            tuple[str, bool]: Tuple containing:
                - error message or empty string
                - whether syntax is valid
        """
        output = self._communicate(f"{self.bash_path} -n <<'EOF'\n{input}\nEOF\n")
        return output, self.returncode == 0

    def _setup_workspace(self) -> None:
        """
        Set up workspace directory in container.

        Creates and configures workspace directory structure, sets permissions,
        and initializes required files.

        Raises:
            RuntimeError: If workspace setup fails
        """
        assert self.task is not None
        assert self.container is not None
        assert self.container_obj is not None
        self.communicate_with_handling(
            f"rm -rf {self.task_workspace!s}",
            error_msg="Failed to remove task workspace",
        )
        self.communicate_with_handling(
            f"mkdir {self.task_workspace!s}",
            error_msg="Failed to create task workspace",
        )

        # FIXME: ooo bad design. memory configuration is split across task and environment. A case can be made to integrate memory into the agent class.
        if self.args.memory_enabled:
            if self.task_args.memory_path is not None and Path(self.task_args.memory_path).exists():
                # copy existing memory file from local folder to container
                copy_anything_to_container(
                    container=self.container_obj,
                    container_type=self.args.container_type,
                    host_path=str(self.task_args.memory_path),
                    container_path=str(self.workspace_dir),
                )
            else:
                # create an empty memory file if doesn't exist
                self.communicate_with_handling(
                    f"touch {self.memory_path!s}",
                    error_msg="Failed to create an empty memory file",
                )

        for dataset in self.task.args._datasets:
            assert isinstance(dataset, BaseDatasetConfig)
            if dataset.is_local:
                # Get list of files in dataset.data_path
                data_path = Path(dataset.data_path)
                # Copy data folder to agent workspace directory.
                copy_anything_to_container(
                    container=self.container_obj,
                    container_type=self.args.container_type,
                    host_path=str(data_path),
                    container_path=f"{self.task_workspace!s}/data",
                )

                # set read-only flags for all files in the data dir
                output = self.communicate(f"ls {self.task_workspace!s}/data/")
                objects = output.strip().split("\n")
                for object in objects:
                    self.container_obj.exec_run(
                        f"chmod -R 555 {self.task_workspace!s}/data/{object}",
                        user="root",
                    )

        # copy all starter code files to workspace_dir
        if self.task.args.starter_code is not None:
            for path in self.task.args.starter_code:
                copy_anything_to_container(
                    container=self.container_obj,
                    container_type=self.args.container_type,
                    host_path=str(path),
                    container_path=f"{self.task_workspace!s}/",
                )

        if self.task.args.evaluation_read_only:
            # set read and execute permissions for evaluation script
            evaluation_paths = [Path(self.task_workspace) / path for path in self.task.args.evaluation_paths]
            for eval_path in evaluation_paths:
                self.container_obj.exec_run(f"chmod 555 {eval_path!s}", user="root")
