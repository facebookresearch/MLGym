"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Base agent implementation for the MLGym framework.

This module provides the core agent functionality including configuration,
history tracking, model interaction, and environment communication. The agent
is responsible for receiving observations from the environment, querying the
model for actions, and executing those actions.

Adapted from SWE-Agent/sweagent/agent/agents.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from simple_parsing.helpers.fields import field
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from tenacity import RetryError

from mlgym.agent.history_processors import HistoryProcessor
from mlgym.agent.parsing import ParseFunction
from mlgym.backend.debugging import ReplayModel
from mlgym.backend.utils import get_model
from mlgym.exceptions import (
    APIError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
)
from mlgym.tools.tools import ToolHandler, ToolsConfig
from mlgym.types import AgentInfo, History, HistoryItem, Trajectory, TrajectoryStep
from mlgym.utils.config import convert_paths_to_abspath
from mlgym.utils.log import get_logger, logging

if TYPE_CHECKING:
    from mlgym.backend.base import APIStats, ModelArguments
    from mlgym.environment.env import MLGymEnv
    from mlgym.environment.tasks import TaskConfig


@dataclass(frozen=True)
class AgentConfig(FlattenedAccess, FrozenSerializable):
    system_template: str
    task_template: str
    next_step_template: str | None = None  # defaults to task_template
    next_step_no_output_template: str | None = None  # default to next_step template
    strategy_template: str | None = None
    demonstration_template: str | None = None
    # Paths to demonstrations. If path is not absolute, it is assumed to be
    # relaive to the MLGym repository root.
    # FIXME: Migrate to pydantic under issue #19
    demonstrations: list[str | Path] = field(default_factory=list)  # noqa: RUF009
    # if True, add demonstration to history instead of as a single message
    put_demos_in_history: bool = False
    # defaults to format_error_template in ParseFunction
    format_error_template: str | None = None
    # Commands configuration with blocklist, env variables and util functions
    # FIXME: Migrate to pydantic under issue #19
    tools: ToolsConfig = field(default_factory=ToolsConfig)  # noqa: RUF009
    output_parser: str | ParseFunction = "ThoughtActionParser"
    history_processor: str = "DefaultHistoryProcessor"
    # FIXME: Migrate to pydantic under issue #19
    history_processor_args: dict[str, Any] = field(default_factory=dict)  # noqa: RUF009

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools_handler", ToolHandler(self.tools))

        object.__setattr__(self, "demonstrations", convert_paths_to_abspath(self.demonstrations))

        if self.next_step_template is None:
            object.__setattr__(self, "next_step_template", self.task_template)
        if self.next_step_no_output_template is None:
            object.__setattr__(self, "next_step_no_output_template", self.next_step_template)

        if isinstance(self.output_parser, str):
            object.__setattr__(self, "output_parser", ParseFunction.get(self.output_parser))
        assert isinstance(self.output_parser, ParseFunction)

        if self.format_error_template is None:
            object.__setattr__(
                self,
                "format_error_template",
                self.output_parser.format_error_template,
            )
        else:
            object.__setattr__(
                self,
                "format_error_template",
                self.format_error_template.format(**self.__dict__),
            )

        object.__setattr__(
            self,
            "history_processor",
            HistoryProcessor.get(self.history_processor, **self.history_processor_args),
        )


@dataclass(frozen=True)
class AgentArguments(FlattenedAccess, FrozenSerializable):
    """Configure the agent's behaviour (templates, parse functions, ...)."""

    model: ModelArguments

    # Policy can only be set via config yaml file from command line
    agent_config_path: Path | str | None = None
    # FIXME: Migrate to pydantic under issue #19
    config: AgentConfig | None = field(default=None, cmd=False)  # noqa: RUF009
    log_verbose_to_console: bool = False

    def __post_init__(self) -> None:
        if self.config is None and self.agent_config_path is not None:
            # If unassigned, we load the config from the file to store its contents with the overall arguments
            config = AgentConfig.load_yaml(self.agent_config_path)
            object.__setattr__(self, "config", config)
        assert self.config is not None


class BaseAgent:
    """
    Base agent class that handles model-environment interaction.

    The agent manages the conversation history, executes actions in the environment,
    and maintains trajectory information. It implements the core logic for:
    - Processing observations from the environment
    - Querying the model for actions
    - Executing actions and handling errors
    - Maintaining conversation history and trajectory
    - Saving results and trajectories

    Args:
        name (str): Name identifier for the agent
        args (AgentArguments): Configuration arguments for the agent
    """

    def __init__(self, name: str, args: AgentArguments) -> None:
        self.name: str = name
        # TODO: currently only used to get the model name, so might remove this later
        self._args: AgentArguments = args
        self.config: AgentConfig | None = args.config
        assert self.config is not None

        # Get tools handler
        self.tools: ToolHandler = self.config.tools_handler
        assert self.tools is not None

        # Get model details
        self.model = get_model(args.model, self.tools.commands)

        # Get system arguments
        self.system_args = {
            "command_docs": self.tools.command_docs,
            **self.tools.env_variables,
        }

        # FIXME: We pass this here to get the task definition and format the prompt template. But again, that is something that we can decouple from the agent.
        self.task_args: TaskConfig | None = None

        self.logger: logging.Logger = get_logger(name)

        # FIXME: Environment dependency
        self._env: MLGymEnv | None = None
        self.traj_dir: Path | None = None

        self._history: History = []
        self._trajectory: Trajectory = []
        self._info: AgentInfo
        # FIXME: Trace is not a custom logging level, and is thus not recognized by mypy.
        self._default_logging_level = logging.INFO if args.log_verbose_to_console else logging.TRACE  # type: ignore

    def set_log_verbose_to_console(self, flag: bool) -> None:
        """
        Sets the logging verbosity level for console output.

        Args:
            flag (bool): If True, sets logging to INFO level, otherwise TRACE
        """
        if flag:
            self._default_logging_level = logging.INFO

    @property
    def history(self) -> History:
        """
        History that is passed on to the model.
        Use `_append_history` to modify.
        """
        return self._history

    @history.setter
    def history(self, value: History) -> None:
        self._history = value

    @property
    def trajectory(self) -> Trajectory:
        """
        Trajectory of the agent for the current task. In contrast to `history`,
        this is mostly for the informational value of how the agent interacted with
        the environment and is also what is being used when replaying the trajectory
        """
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value: Trajectory) -> None:
        self._trajectory = value

    @property
    def info(self) -> AgentInfo:
        """Information about the agent's run"""
        return self._info

    @info.setter
    def info(self, value: AgentInfo) -> None:
        self._info = value

    @property
    def traj_path(self) -> Path | None:
        """
        Returns path to the trajectory.
        The path is reset for every new task.
        """
        # FIXME: Environment dependency
        if self.traj_dir and self._env is not None:
            assert self._env.task is not None
            assert self._env.task.args.id
            return self.traj_dir / (self._env.task.args.id + ".traj")
        return None

    def _append_history(self, item: HistoryItem) -> None:
        """
        Adds an item to the conversation history.

        Args:
            item (HistoryItem): History item to append
        """
        self.history.append(item)

    def setup(self, task_args: TaskConfig, init_model_stats: APIStats | None = None) -> None:
        """
        Initializes the agent for a new task.

        Sets up the system message, loads demonstrations if configured, and
        prepares the conversation history.

        Args:
            task_args (TaskConfig): Task configuration
            init_model_stats (APIStats | None): Initial model usage statistics
        """
        assert self.config is not None
        self.task_args = task_args
        self._history = []
        self._trajectory = []
        self._info = AgentInfo()

        # reset model stats
        self.model.reset_stats(init_model_stats)

        system_msg = self.config.system_template.format(**self.system_args, **asdict(self.task_args))
        self.logger.log(self._default_logging_level, f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(HistoryItem({"role": "system", "content": system_msg, "agent": self.name}))
        # FIXME: We don't need to add the demonstrations in setup. Or we should separate it out into a separate method based on the user agent definition.
        if "history_to_messages" in dir(self.model):
            for demonstration_path in self.config.demonstrations:
                if self.config.demonstration_template is None and not self.config.put_demos_in_history:
                    msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
                    raise ValueError(msg)

                # Load history
                self.logger.log(
                    self._default_logging_level,
                    f"DEMONSTRATION ({self.name}): {demonstration_path}",
                )
                demo_history = json.loads(Path(demonstration_path).read_text())["history"]
                demo_history = [
                    entry
                    for entry in demo_history
                    if ("agent" not in entry) or ("agent" in entry and entry["agent"] == self.name)
                ]

                if self.config.put_demos_in_history:
                    if self.config.demonstration_template is not None:
                        self.logger.warning("Demonstration template is ignored for put_demos_in_history=True")
                    # Add demonstrations to history directly as separate messages
                    for entry in demo_history:
                        if entry["role"] != "system":
                            entry["is_demo"] = True
                            self._append_history(entry)
                else:
                    # Add demonstration as single message to history if model is not a replay model
                    if not isinstance(self.model, ReplayModel):
                        demo_message = self.model.history_to_messages(
                            demo_history,
                            is_demonstration=True,
                        )
                        if self.config.demonstration_template is None:
                            msg = "Cannot use demonstrations without a demonstration template"
                            raise ValueError(msg)
                        demonstration = self.config.demonstration_template.format(demonstration=demo_message)
                        self._append_history(
                            {
                                "agent": self.name,
                                "content": demonstration,
                                "is_demo": True,
                                "role": "user",
                            },
                        )

    @property
    def local_history(self) -> list[HistoryItem]:
        """Return the history of the agent."""
        assert self.config is not None
        assert isinstance(self.config.history_processor, HistoryProcessor)
        truncated_history = self.config.history_processor(
            [entry for entry in self.history if entry.get("agent") == self.name]
        )
        return truncated_history

    def save_trajectory(self) -> None:
        """
        Save the trajectory to disk.
        This includes the history, the environment state, and the model stats
        """
        # FIXME: Environment dependency
        assert self._env is not None
        data = {
            "environment": self._env.name,
            "trajectory": self._trajectory,
            "history": self._history,
            "info": self._info,
        }
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def save_results(self) -> None:
        """
        Save the results to disk.
        This includes the model stats and the task info
        """
        # FIXME: Environment dependency
        results = {}
        if self._info.get("score", None) is not None:
            results["agent"] = self._info["score"]
        assert self._env is not None
        assert self._env.task is not None
        if self._env.task.args.baseline_scores:
            results["baseline"] = self._env.task.args.baseline_scores[0]
        assert self.traj_path is not None
        results_path = self.traj_path.parent / "results.json"
        results_path.write_text(json.dumps(results, indent=2))

    def forward(self, observation: str | None, available_actions: list[str], state: str) -> tuple[str, str, str]:
        """
        Processes an observation and generates the next action.

        Args:
            observation (str | None): Current observation from environment
            available_actions (list[str]): List of available actions
            state (str): Current environment state

        Returns:
            tuple[str, str, str]: Tuple containing:
                - thought: Model's reasoning
                - action: Selected action to execute
                - output: Raw model output
        """

        thought, action, output = self.forward_with_error_check(observation, state)

        self._append_history(
            {
                "role": "assistant",
                "content": output,
                "thought": thought,
                "action": action,
                "agent": self.name,
            }
        )

        self.logger.log(self._default_logging_level, f"💭 THOUGHT ({self.name})\n{thought}")
        self.logger.log(self._default_logging_level, f"🎬 ACTION ({self.name})\n{action}")

        return thought, action, output

    def forward_model(self, observation: str | None, state: str) -> str:
        """
        Queries the model with current state and observation using appropriate template.

        Selects and formats the appropriate template based on conversation context,
        then queries the model for a response.

        Args:
            observation (str | None): Current observation from environment
            state (str): Current environment state as JSON string

        Returns:
            str: Raw model output

        Raises:
            ValueError: If state is not valid JSON
        """
        assert self.config is not None
        try:
            state_vars = json.loads(state)
        except json.JSONDecodeError as e:
            msg = f"State {state!r} is not valid json. This is an internal error, please report it."
            raise ValueError(msg) from e
        # add step information to state_vars
        # FIXME: Environment dependency
        assert self._env is not None
        state_vars["current_step"] = self._env.current_step
        state_vars["remaining_steps"] = self._env.max_steps - self._env.current_step

        templates: list[str] = []

        # Determine observation template based on what prior observation was
        if self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False):
            # Show task template if prev. obs. was initial system message
            templates = [self.config.task_template]
            if self.config.strategy_template is not None:
                templates.append(self.config.strategy_template)
        elif observation is None or observation.strip() == "":
            # Show no output template if observation content was empty
            assert self.config.next_step_no_output_template is not None  # linting
            templates = [self.config.next_step_no_output_template]
        else:
            # Show standard output template if there is observation content
            assert self.config.next_step_template is not None  # linting
            templates = [self.config.next_step_template]

        # Format selected template(s) with information
        messages = []
        assert self.task_args is not None
        for template in templates:
            messages.append(
                template.format(
                    **asdict(self.task_args),
                    **self.system_args,
                    **state_vars,
                    observation=(observation if observation is not None else ""),
                ),
            )

        message = "\n".join(messages)

        self.logger.log(self._default_logging_level, f"🤖 MODEL INPUT ({self.name})\n{message}")
        self.logger.info(f"({self.name}) {state_vars=}")
        self._append_history({"role": "user", "content": message, "agent": self.name})

        # model query hooks here
        return self.model.query(self.local_history)

    def retry_after_format_fail(self, output: str) -> str:
        """
        Requests model to correct malformed output.

        Makes a new query to the model asking it to fix formatting issues,
        without saving the failed attempt to persistent history.

        Args:
            output (str): The malformed model output

        Returns:
            str: New model output after correction attempt
        """
        assert self.config is not None
        format_error_template = self.config.format_error_template

        self.logger.warning(f"MALFORMED OUTPUT ({self.name})\n{output}")
        self.logger.warning(f"FORMAT ERROR ({self.name})\n{format_error_template}")

        temp_history = [
            *self.local_history,
            HistoryItem({"role": "assistant", "content": output, "agent": self.name}),
            HistoryItem({"role": "user", "content": format_error_template, "agent": self.name}),
        ]

        return self.model.query(temp_history)

    def retry_after_blocklist_fail(self, output: str, action: str) -> str:
        """
        Requests model to correct blocked command usage.

        Makes a new query to the model after it attempted to use a blocked command,
        without saving the failed attempt to persistent history.

        Args:
            output (str): The original model output
            action (str): The blocked action that was attempted

        Returns:
            str: New model output after correction attempt
        """
        assert self.config is not None
        name = action.strip().split()[0]
        blocklist_error_message = self.tools.config.blocklist_error_template.format(name=name)

        self.logger.warning(f"BLOCKLISTED OUTPUT ({self.name})\n{output}")
        self.logger.warning(f"BLOCKLIST ERROR ({self.name})\n{blocklist_error_message}")

        temp_history = [
            *self.local_history,
            HistoryItem({"role": "assistant", "content": output, "agent": self.name}),
            HistoryItem({"role": "user", "content": blocklist_error_message, "agent": self.name}),
        ]

        return self.model.query(temp_history)

    def check_format_and_requery(self, output: str) -> tuple[str, str, str]:
        """
        Validates model output format and handles any necessary requerying.

        Attempts to parse model output and handles format errors or blocked actions
        by requerying up to 2 times.

        Args:
            output (str): Raw model output to validate

        Returns:
            tuple[str, str, str]: Tuple containing:
                - thought: Model's reasoning
                - action: Selected action
                - output: Raw model output

        Note:
            Special handling for 'human' and 'human_thought' model types.
        """
        # Condition for handling outputs with no thought (just action)
        assert self.config is not None
        if self.model.args.model_name == "human":
            return "", output, output
        elif self.model.args.model_name == "human_thought":
            thought, action = ParseFunction.get("ThoughtActionParser")(output, self.tools.commands, strict=False)
            return thought, action, output

        format_fails = blocklist_fails = 0

        while format_fails + blocklist_fails <= 2:
            try:
                thought, action = self.config.output_parser(output, self.tools.commands, strict=False)  # type: ignore
            except KeyboardInterrupt:
                raise
            except FormatError:
                format_fails += 1
                output = self.retry_after_format_fail(output)
                continue
            if self.tools.should_block_action(action):
                blocklist_fails += 1
                output = self.retry_after_blocklist_fail(output, action)
            else:
                return thought, action, output
        self.logger.warning(f"Malformat limit reached: \n{output}")
        return "Exit due to format error", "exit_format", output

    def forward_with_error_check(self, observation: str | None, state: str) -> tuple[str, str, str]:
        """
        Wraps forward_model with comprehensive error handling.

        Handles various error conditions including context window exceeded,
        cost limits, API errors, and other runtime errors.

        Args:
            observation (str | None): Current observation
            state (str): Current environment state

        Returns:
            tuple[str, str, str]: Tuple containing:
                - thought: Model's reasoning or error message
                - action: Selected action or error action
                - output: Raw output or error message
        """
        try:
            return self.check_format_and_requery(self.forward_model(observation, state))
        except KeyboardInterrupt:
            raise
        except RuntimeError as e:
            self.logger.warning(f"Runtime error: {e}")
            return (
                f"Exit due to runtime error: {e}",
                "exit_error",
                f"exit due to runtime error: {e}",
            )
        except ContextWindowExceededError:
            self.logger.warning("Context window exceeded")
            return (
                "Exit due to context window",
                "exit_context",
                "Exit due to context window",
            )
        except CostLimitExceededError:
            self.logger.warning("Cost limit exceeded")
            return "Exit due to cost limit", "exit_cost", "Exit due to cost limit"
        except RetryError as e:
            self.logger.warning(f"Retry error: {e}")
            return (
                f"Exit due to retry error: {e}",
                "exit_api",
                f"exit due to retry error: {e}",
            )
        except APIError as e:
            self.logger.warning(f"Unexpected error: {e}")
            return (
                f"Exit due to unexpected error: {e}",
                "exit_api",
                f"exit due to unexpected error: {e}",
            )

    # FIXME: move to environment api
    def init_environment_vars(self, env: MLGymEnv) -> None:
        """
        Initializes environment variables for a new environment.

        Args:
            env (MLGymEnv): Environment to initialize
        """
        assert self.config is not None
        self.set_environment_vars(env, self.config.env_variables)

    # FIXME: move to environment api
    def set_environment_vars(self, env: MLGymEnv, env_variables: dict[str, Any]) -> None:
        """
        Sets environment variables and configures command files.

        Sets up environment variables and handles different types of command files:
        - Shell scripts with shebang
        - Source files (.sh extension)
        - Utility files (prefixed with underscore)

        Args:
            env (MLGymEnv): Environment to configure
            env_variables (dict[str, Any]): Variables to set

        Raises:
            ValueError: If a non-shell script file lacks proper configuration
            RuntimeError: If setting environment variables fails
        """
        assert self.config is not None  # mypy
        commands_to_execute = [self.tools.state_command.code] + [f"{k}={v}" for k, v in env_variables.items()]
        commands = "\n".join(commands_to_execute)
        try:
            output = env.communicate(commands)
            # FIXME: remove dependency on environment
            if env.returncode != 0:
                msg = f"Nonzero return code: {env.returncode}\nOutput: {output}"
                raise RuntimeError(msg)  # noqa: TRY301
        except KeyboardInterrupt:
            raise
        except Exception:
            self.logger.warning("Failed to set environment variables")
            raise
        command_files = []
        for file in self.tools.config.command_files:
            datum = {}
            with open(file) as f:
                contents = f.read()
            datum["contents"] = contents
            filename = Path(file).name
            if not contents.strip().startswith("#!"):
                if filename.endswith(".sh"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "source_file"
                elif filename.startswith("_"):
                    # files are sourced, so they are not executable
                    datum["name"] = Path(file).name
                    datum["type"] = "utility"
                else:
                    msg = (
                        f"Non-shell script file {file} does not start with shebang.\n"
                        "Either add a shebang (#!) or change the file extension to .sh if you want to source it.\n"
                        "You can override this behavior by adding an underscore to the file name (e.g. _utils.py)."
                    )
                    raise ValueError(msg)
            else:
                # scripts are made executable
                datum["name"] = Path(file).name.rsplit(".", 1)[0]
                datum["type"] = "script"
            command_files.append(datum)
        env.add_commands(command_files)

    def get_environment_vars(self, env: MLGymEnv) -> dict[str, Any]:
        """
        Retrieves current environment variables from container.

        Args:
            env (MLGymEnv): Environment to query

        Returns:
            dict[str, Any]: Dictionary of environment variable names and values
        """
        assert self.config is not None
        env_vars = {}
        for var in self.tools.config.env_variables:
            env_vars[var] = env.communicate(f"echo ${var}").strip()
        return env_vars

    def _run_step(self, observation: str | None) -> tuple[str | None, bool]:
        """
        Executes a single step of the agent's decision-making loop.

        Gets environment state, queries model for action, executes action,
        and updates trajectory information.

        Args:
            observation (str | None): Current observation

        Returns:
            tuple[str | None, bool]: Tuple containing:
                - observation: New observation or None
                - done: Whether episode is complete
        """
        # FIXME: Environment dependency
        assert self._env is not None
        assert self.config is not None

        # NOTE: changed the state to be an empty string instead of None
        state = self._env.communicate(self.tools.state_command.name) if self.tools.state_command else ""
        thought, action, output = self.forward(observation, self._env.get_available_actions(), state)

        # TODO: actions generated hooks here

        run_action: str = self.tools.guard_multiline_input(action).strip()

        done = False
        observation = None
        execution_t0 = time.perf_counter()

        assert self._env is not None
        assert self.config is not None
        observation, _, done, _info = self._env.step(run_action)
        self.info.update(_info)

        if run_action.strip() == self.tools.submit_command:
            done = True
        execution_time = time.perf_counter() - execution_t0

        trajectory_step = TrajectoryStep(
            state=state,
            response=output,
            thought=thought,
            action=action,
            execution_time=execution_time,
            observation=observation,
        )
        self.trajectory.append(trajectory_step)
        model_stats: APIStats = self.model.stats
        self.info["model_stats"] = model_stats.to_dict()
        return observation, done

    def run(
        self,
        env: MLGymEnv,
        observation: str | None = None,
        traj_dir: Path | None = None,
        return_type: str = "info_trajectory",
        init_model_stats: APIStats | None = None,
    ) -> AgentInfo | tuple[AgentInfo, Trajectory]:
        """
        Runs the agent in the given environment until completion.

        Manages the main interaction loop between agent and environment,
        including initialization, step execution, and result saving.

        Args:
            env (MLGymEnv): Environment to run in
            observation (str | None): Initial observation
            traj_dir (Path | None): Directory to save trajectory files
            return_type (str): Type of return value to provide
            init_model_stats (APIStats | None): Initial model statistics

        Returns:
            Various types depending on return_type:
                - "info_trajectory": Tuple of (info dict, trajectory list)
                - "info": Just the info dictionary
                - others: Specific field from last trajectory step

        Note:
            Saves trajectory and results to disk if traj_dir is provided.
        """
        assert env.task is not None
        assert env.container is not None

        if env.container is not None:
            self.logger.info(f"Initializing commands for container pid {env.container.pid}")
            self.init_environment_vars(env)

        self.setup(env.task.args, init_model_stats)
        # TODO: configure summarizer here

        # Save/reset some attributes
        # FIXME: Environment dependency
        self.trajectory = Trajectory()
        self._env = env
        self.info = AgentInfo()
        self.traj_dir = traj_dir

        self.logger.info(f"Trajectory will be saved to {self.traj_path}")

        # Run action/observation loop
        # TODO: run start hooks here
        done = False
        while not done:
            try:
                observation, done = self._run_step(observation)
            except Exception as e:
                self.logger.warning(f"Error in run step: {e}\nExiting environment...")
                done = True

            self.save_trajectory()
            if done:
                done = True
                self.save_results()
        # TODO: run done hooks here

        if self.traj_path is not None:
            self.logger.info(f"Trajectory saved to {self.traj_path}")
            self.logger.info(f"Results saved to {self.traj_path.parent / 'results.json'}")

        if return_type == "info":
            return self.info
        if return_type == "info_trajectory":
            return self.info, self.trajectory
        # FIXME: this is a hack so that users can get any specific field from the last trajectory step
        return self.trajectory[-1][return_type]  # type: ignore
