# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import tensordict
tensordict.set_list_to_stack(True).set()

import dataclasses
import gymnasium as gym
import json
import re
import torch
import torchrl
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from tensordict import NonTensorData, TensorClass, TensorDict, TensorDictBase
from tensordict.utils import is_non_tensor
from torchrl._utils import logger as torchrl_logger
from torchrl.data import Choice, Composite, NonTensor
from torchrl.envs import ConditionalSkip, EnvBase, GymWrapper, Transform, TransformedEnv
from typing import Any, Dict, List, Literal, Union

from mlgym.agent.parsing import FormatError, ParseFunction, ThoughtActionParser
from torchrl.data import History

################ Inv transforms ################
# Transforms to apply prior to pass the model output to the env

class SubAction(TensorClass):
    agent: str
    action: str
    cmd_name: str | None
    args: str



class MLGymBaseTransform(Transform):
    """Base class for all MLGym transforms."""

    @property
    def config(self):
        return self.parent.base_env.config

    @property
    def system_args(self):
        return {
            "command_docs": self.config.tools_handler.command_docs, **self.config.tools_handler.env_variables,
        }

    @property
    def task_args(self):
        # Placeholder
        task_args = getattr(self, "_task_args", None)
        if task_args is None:
            return self.parent.base_env.task.args
        return task_args

    @task_args.setter
    def task_args(self, task_args):
        self._task_args = task_args

    @property
    def name(self):
        return "torchrl"

    @property
    def state_command(self):
        return self.config.state_command.name

    @property
    def agent_args(self):
        return self.parent.base_env.agent_args

    @property
    def model_name(self) -> Literal["human", "human_thought"]:
        return self.agent_args.model.model_name

#######################################################
##### Forward transforms: Format the env output #######


# Transform #0: Resets the env
class ResetModule(MLGymBaseTransform):
    """Runs setup pipeline and enables multi-resets.

    The reset method reads the 'system' initial input from the config and parses it to a History
    object.

    """

    def __init__(self):
        super().__init__(in_keys=[], out_keys=["history"])

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        env = self.parent.base_env._env
        env.reset_container()
        env.communicate(f"cd {Path(env.task_workspace).parent}")
        return tensordict

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: what to do with this?
        # reset model stats
        # self.model.reset_stats(init_model_stats)
        # env = self.parent.base_env._env

        env = self.parent.base_env._env
        self.set_environment_vars(env, self.config.env_variables)

        system_msg = self.config.system_template.format(**self.system_args, **asdict(self.task_args))
        # self.logger.log(self._default_logging_level, f"SYSTEM ({self.name})\n{system_msg}")
        history = History(
            role="system",
            content=system_msg,
            # agent=self.name,
            batch_size=(1,),
            device=self.parent.device
        )
        tensordict_reset["history"] = history

        return tensordict_reset

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        # Placeholder
        if "history" not in next_tensordict:
            if "local_history" in tensordict:
                torchrl_logger.debug("Getting the local history from prev action.")
                local_history = tensordict["local_history"]
                torchrl_logger.debug("The local history is {}".format(local_history))
            else:
                local_history = None
            history = tensordict["history"]
            if local_history is not None:
                history = torch.stack(
                    list(history.unbind(-1)) + [local_history], -1
                )
                tensordict["history"] = history
            next_tensordict["history"] = history
        return next_tensordict

    def set_environment_vars(self, env: "MLGymEnv", env_variables: Dict[str, Any]) -> None:
        commands_to_execute = ([self.config.state_command.code] +  # [code for code in self.config.util_functions] +
                               # [command.code for command in self.config._commands] +
                               [f"{k}={v}" for k, v in env_variables.items()])
        print("commands_to_execute", commands_to_execute)
        commands = "\n".join(commands_to_execute)
        try:
            output = env.communicate(commands)
            if env.returncode != 0:
                msg = f"Nonzero return code: {env.returncode}\nOutput: {output}"
                raise RuntimeError(msg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise e
        command_files = list()
        for file in self.config.command_files:
            datum = dict()
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
                    msg = (f"Non-shell script file {file} does not start with shebang.\n"
                           "Either add a shebang (#!) or change the file extension to .sh if you want to source it.\n"
                           "You can override this behavior by adding an underscore to the file name (e.g. _utils.py).")
                    raise ValueError(msg)
            else:
                # scripts are made executable
                datum["name"] = Path(file).name.rsplit(".", 1)[0]
                datum["type"] = "script"
            command_files.append(datum)
        # TODO: implement add commands method in environment
        env.add_commands(command_files)

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        observation_spec["history"] = History.default_spec()
        return observation_spec

    def transform_action_spec(self, action_spec: Composite) -> Composite:
        if isinstance(action_spec, Composite):
            action_spec["action"] = self.transform_action_spec(action_spec["action"])
            return action_spec
        # make the "random" action just a choice between innocuous bash commands
        return Choice(
            [NonTensor(example_data="ls -rtlh", shape=action_spec.shape),
             NonTensor(example_data="pwd", shape=action_spec.shape), ]
        )

    def transform_state_spec(self, state_spec: Composite) -> Composite:
        state_spec["history"] = History.default_spec()
        return state_spec


# Transform #1: env -> state
class ReadState(MLGymBaseTransform):
    """Reads current state and writes it as a parsable str in the tensordict."""

    # from mlgym/agent/base.py:BaseAgent:forward_model
    def _step(
            self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        base_mlgym_env = self.parent.base_env  # getattr is forwarded

        command = self.state_command
        state = base_mlgym_env.communicate(command) if self.state_command else None

        next_tensordict["state"] = state
        torchrl_logger.debug(f"state:\n{state}")
        torchrl_logger.debug(f"observation:\n{tensordict['observation']}")

        return next_tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # tensordict_reset.setdefault("message", NonTensorData(""))
        # tensordict_reset.setdefault("state", NonTensorData(""))
        return self._step(tensordict_reset, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec.set(
            "state", NonTensor(
                example_data="a string", device=observation_spec.device, shape=observation_spec.shape
            )
        )
        return observation_spec


# Transform #2: state -> message
class StateToMessage(MLGymBaseTransform):
    """Parses the string using json to a given template.

    Requires:
        - a 'state' key from the ReadState transform
        - an 'observation' key from the base environment

    """

    def _step(
            self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        base_mlgym_env = self.parent.base_env  # getattr is forwarded
        observation = tensordict["observation"]
        state = tensordict["state"]
        config = self.config

        current_step = base_mlgym_env.current_step
        max_steps = base_mlgym_env.max_steps
        try:
            state_vars = json.loads(state)
        except json.JSONDecodeError as e:
            msg = f"State {state!r} is not valid json. This is an internal error, please report it."
            raise ValueError(msg) from e
        # add step information to state_vars
        state_vars["current_step"] = current_step
        state_vars["remaining_steps"] = max_steps - current_step

        # FIXME: we don't need to do this, we have our own observation space
        # Determine observation template based on what prior observation was

        history: History = tensordict["history"]
        if history[..., -1].role == "system" or history[..., -1].is_demo:
            # Show task template if prev. obs. was initial system message
            templates = [config.task_template]
            if config.strategy_template is not None:
                templates.append(config.strategy_template)
        elif observation is None or observation.strip() == "":
            # Show no output template if observation content was empty
            assert config.next_step_no_output_template is not None  # linting
            templates = [config.next_step_no_output_template]
        else:
            # Show standard output template if there is observation content
            assert config.next_step_template is not None  # linting
            templates = [config.next_step_template]

        # Format selected template(s) with information
        messages = list()
        assert self.task_args is not None
        for template in templates:
            messages.append(
                template.format(
                    **asdict(self.task_args),
                    **self.system_args,
                    **state_vars,
                    observation=(observation if observation is not None else ""),
                    # missing forwarded_vars because no attempts
                ), )

        message = "\n".join(messages)
        next_tensordict["message"] = message
        # model query hooks here
        return next_tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        # tensordict_reset.setdefault("message", NonTensorData(""))
        # tensordict_reset.setdefault("state", NonTensorData(""))
        return self._step(tensordict_reset, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec.set(
            "message", NonTensor(
                example_data="a string", device=observation_spec.device, shape=observation_spec.shape
            )
        )
        return observation_spec


# Transform #3: Append message to history
class MessageToHistory(MLGymBaseTransform):
    """Parses the message string to a History object, then reparses the history to a complete message.

    .. seealso:: HistoryToMessage

    """

    def __init__(self):
        super().__init__(in_keys=["message", "history"], out_keys=["history", "chat"])

    # from mlgym/agent/base.py:BaseAgent:local_history
    # from mlgym/agent/base.py:BaseAgent:_append_history
    def _step(
            self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # From PrepareDataForModel
        message: str = next_tensordict["message"]
        # from mlgym/agent/base.py:BaseAgent:forward_model
        history = tensordict["history"]
        cur_history = History(role="user", content=message, batch_size=(), device=self.parent.device)
        torchrl_logger.debug(f"Current history:\n{cur_history}\nmessage:\n{message}")
        # This is the basic thing our transform does: append the history to the existing one.
        # (We should be able to extend the lazy stack directly)
        history = history.append(cur_history)
        torchrl_logger.debug(f"History length: {history.shape[-1]}")

        next_tensordict["history"] = history

        torchrl_logger.debug(f"parsed message from history:\n{TensorDict.from_any(message)}")

        return next_tensordict

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._step(tensordict_reset, tensordict_reset)

##### Inverse transforms: Format the action from the model for the env #######
class TemplateTransform(MLGymBaseTransform):
    # alternative to DummyFormat, wip
    def __init__(
            self,
            in_keys=None,
            out_keys=None,
            in_keys_inv=None,
            out_keys_inv=None,
            tokenizer=None,
            chat_template_name: Literal["chatml_format"] = "chatml_format",
            continue_final_message: bool = False,
            tokenize: bool = False,
            return_tensors: str = "pt",
            return_dict: bool = False,
            padding: bool | str = False,
            truncation: bool | str = False
    ):
        super().__init__(
            in_keys=["history"] if in_keys is None else in_keys,
            out_keys=["text"] if out_keys is None else out_keys,
            in_keys_inv=["text_response"] if in_keys_inv is None else in_keys_inv,
            out_keys_inv=["action"] if out_keys_inv is None else out_keys_inv, )
        self.chat_template_name = chat_template_name
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.continue_final_message = continue_final_message
        self.return_tensors = return_tensors
        self.return_dict = return_dict
        self.padding = padding
        self.truncation = truncation

    def transform_observation_spec(self, observation_spec: Composite):
        observation_spec["text"] = NonTensor(
            example_data="<some chat string>",
            shape=observation_spec.shape,
            device=observation_spec.device
            )
        return observation_spec

    def _apply_transform(self, history: History) -> NonTensorData:
        torchrl_logger.debug(f"Applying template:\n{history[-1]}")
        return history.apply_chat_template(
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            # chat_template=self.chat_template_name,
            continue_final_message=self.continue_final_message,
            tokenize=self.tokenize,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.in_keys_inv:
            history, action = self._inv_apply_transform(tensordict["text_response"])
            print('local history after action', history)
            tensordict["local_history"] = history
            tensordict["action"] = action
        return tensordict

    def _inv_apply_transform(self, action):
        if self.tokenize:
            action = self.tokenizer.decode(action)

        if not isinstance(action, (str, list)):
            action = action.data
            history, action = self._inv_apply_transform(action)
            action = NonTensorData(action, batch_size=action.batch_size, device=action.device)
            return history, action

        history = History.inv_chat_template(action, chat_template_name=self.chat_template_name)
        action = history.get("content")
        return history, action


class IsolateCodeBlock(MLGymBaseTransform):
    def __init__(self):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])
        self.parser = ThoughtActionParser()

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.in_keys_inv:
            try:
                action = self._inv_apply_transform(tensordict["action"])
                print("action block", action)
                tensordict["action"] = action
                tensordict["retry"] = False
            except FormatError:
                # Replace the local history with one that tells our LLM that things haven't worked out as expected
                tensordict["local_history"] = History(
                    role="user",
                    content="The command did not respect the instructions. Try again.",
                    device=self.parent.device
                )
                tensordict["retry"] = True

        return tensordict

    def _inv_apply_transform(self, action):
        if not isinstance(action, (str, list)):
            return NonTensorData(
                self._inv_apply_transform(action.data), batch_size=action.batch_size, device=action.device
            )
        thought, action = self.parser(action, None)
        torchrl_logger.debug(f'isolated action:\n{action}')
        return action

####################################################################
# TODO: These transforms are WIP and inspired by the agent class.

# Transform #1. Entry transform: checks the action, then converts in thought - action - output
class CheckActionFormat(MLGymBaseTransform):
    """This class checks the format, and write the results in thought, action and output.

    If the format is violated, a retry entry indicates that the env.step must be skipped and the model
    must be queried again.

    """

    # from mlgym.agents.base.py:BaseAgent:check_format_and_requery
    def _inv_call(
            self, tensordict: TensorDictBase, ) -> TensorDictBase:

        output = tensordict["action"]

        config = self.config
        assert config is not None

        retry_cond = False
        if self.model_name == "human":
            thought, action, output = "", output, output
        elif self.model_name == "human_thought":
            thought, action = ParseFunction.get("ThoughtActionParser")(
                output, config._commands + config.subroutine_types, strict=False
            )
        else:
            # TODO: implement a limit on how many trials can be done
            # format_fails = blocklist_fails = 0
            #
            # while format_fails + blocklist_fails <= 2:
            try:
                torchrl_logger.debug(f"action:\n{output}")
                thought, action = config.output_parser(
                    output, config._commands + config.subroutine_types, strict=False
                )  # type: ignore
                torchrl_logger.debug(f"processed action:\n{action}")
            except KeyboardInterrupt:
                raise
            except FormatError as err:
                torchrl_logger.debug(f"formatting failed:\n {err}")
                # format_fails += 1
                history = self.retry_after_format_fail(tensordict, output)
                thought, action, output = (None,) * 3
                tensordict["history"] = history
                retry_cond = True
            if action is not None and self.should_block_action(action):
                history = self.retry_after_blocklist_fail(tensordict, output, action)
                thought, action, output = (None,) * 3
                tensordict["history"] = history
                retry_cond = True

        tensordict["thought"] = thought
        tensordict["action"] = action
        tensordict["output"] = output
        tensordict["retry_cond"] = retry_cond
        return tensordict

    # FIXME: command blocking check should be in the environment
    def should_block_action(self, action: str) -> bool:
        """
        Check if the command should be blocked.
        """
        names = action.strip().split()[0]
        if len(names) == 0:
            return False
        name = names[0]
        if name in self.config.blocklist:
            return True
        if name in self.config.blocklist_standalone and name == action.strip():
            return True
        # DON'T BLOCK REGEX now
        return False

    def retry_after_format_fail(self, tensordict, output: str) -> str:
        """
        Ask the model to correct (without committing to persistent history) after a malformed model output
        """
        config = self.container.base_env.config
        assert config is not None
        format_error_template = config.format_error_template

        history = tensordict["history"]
        new_history = History(
            role=["assistant", "user"], content=[output, format_error_template], device=self.parent.device, batch_size=(2,)
        )
        history = history.extend(new_history)
        assert history.batch_size[-1] == tensordict["history"].batch_size[-1] + 2
        return history

    # FIXME: we should get the blocklist error message from the environment
    def retry_after_blocklist_fail(self, tensordict, output: str, action: str) -> str:
        """
        Ask the model to correct (without committing to persistent history) after a disallowed command
        """
        assert self.config is not None
        name = action.strip().split()[0]
        blocklist_error_message = self.config.blocklist_error_template.format(name=name)

        history = tensordict["history"]
        new_history = [History(
            role="assistant", content=output, agent=self.name, device=self.parent.device
        ), History(
            role="user", content=blocklist_error_message, agent=self.name, device=self.parent.device
        )]
        history = torch.stack(list(history.unbind(-1)) + new_history, dim=-1)
        assert history.batch_size[-1] == tensordict["history"].batch_size[-1] + 2
        return history


class GuardMultiline(MLGymBaseTransform):

    def __init__(self):
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])

    def set_container(self, container: Union[Transform, EnvBase]) -> None:
        try:
            self._parse_command_patterns()
        except AttributeError:
            pass

    def _parse_command_patterns(self) -> None:
        config = self.parent.base_env.config
        assert config is not None  # mypy
        self.command_patterns = dict()
        for command in config._commands:
            if command.end_name is not None:
                pat = re.compile(
                    rf"^\s*({command.name})\s*(.*?)^({command.end_name})\s*$", re.DOTALL | re.MULTILINE, )
                self.command_patterns[command.name] = pat
            else:
                pat = re.compile(rf"^\s*({command.name})\s*(.*?)$", re.MULTILINE)
                self.command_patterns[command.name] = pat
        self.subroutine_patterns = dict()
        for _, subroutine in config._subroutines.items():
            if subroutine.end_name is None:
                pat = re.compile(rf"^\s*({subroutine.name})\s*(.*?)$", re.MULTILINE)
                self.subroutine_patterns[subroutine.name,] = pat
            else:
                pat = re.compile(
                    rf"^\s*({subroutine.name})\s*(.*?)^({subroutine.end_name})\s*$", re.DOTALL | re.MULTILINE, )
                self.subroutine_patterns[subroutine.name] = pat
        if hasattr(config, "submit_command_end_name"):
            submit_pat = re.compile(
                rf"^\s*({config.submit_command})\s*(.*?)^({config.submit_command_end_name})\s*$",  # type: ignore
                re.DOTALL | re.MULTILINE, )
        else:
            submit_pat = re.compile(rf"^\s*({config.submit_command})(\s*)$", re.MULTILINE)  # group 2 is nothing
        self.subroutine_patterns[config.submit_command] = submit_pat
        self.command_patterns[config.submit_command] = submit_pat

    def _inv_apply_transform(self, action: str) -> str:
        """
        Split action by multiline commands, then append the first line in each multiline command with "<< '{end_name}'".
        Multiline commands (which are specified by an end_name) are commands that span multiple lines and are terminated by a specific end_name.

        Their multi-line argument is sent using a heredoc, which is a way to send a multi-line string to a command in bash.
        """
        if not isinstance(action, str):
            action = action.data
        parsed_action = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, "multi_line_no_subroutines")
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start(): first_match.end()]
                rem_action = rem_action[first_match.end():]
                if pre_action.strip():
                    parsed_action.append(pre_action)
                if match_action.strip():
                    eof = first_match.group(3).strip()
                    if not match_action.split("\n")[0].strip().endswith(f"<< '{eof}'"):
                        guarded_command = match_action[first_match.start():]
                        first_line = guarded_command.split("\n")[0]
                        # add a space before the << for insert command bad file descriptor error
                        guarded_command = guarded_command.replace(first_line, first_line + f" << '{eof}'", 1)
                        parsed_action.append(guarded_command)
                    else:
                        parsed_action.append(match_action)
            else:
                parsed_action.append(rem_action)
                rem_action = ""
        return "\n".join(parsed_action)

    def _get_first_match(self, action: str, pattern_type: str) -> re.Match | None:
        """
        Return the first match of a command pattern in the action string.
        """
        config = self.parent.base_env.config
        assert config is not None
        if pattern_type == "subroutine":
            patterns = {k: v for k, v in self.subroutine_patterns.items()}
        elif pattern_type == "multi_line":
            patterns = {k: v for k, v in self.command_patterns.items() if
                        k in config.multi_line_command_endings or k == config.submit_command}
            patterns += {k: v for k, v in self.subroutine_patterns.items() if k in config.multi_line_command_endings}
        elif pattern_type == "multi_line_no_subroutines":
            patterns = {k: v for k, v in self.command_patterns.items() if k in config.multi_line_command_endings}
        else:
            msg = f"Unknown pattern type: {pattern_type}"
            raise ValueError(msg)

        matches = list()
        for _, pat in patterns.items():
            match = pat.search(action)
            if match:
                matches.append(match)
        if len(matches) == 0:
            return None
        matches = sorted(matches, key=lambda x: x.start())
        return matches[0]


class SplitMultilineAction(MLGymBaseTransform):

    def _inv_call(
            self, tensordict: TensorDictBase, ) -> TensorDictBase:
        action = tensordict["action"]
        action = self.split_action(action)
        tensordict["action"] = action
        return tensordict

    def split_action(self, action: str, pattern_type="subroutine") -> str:
        """
        Split an action into a list of actions in a greedy manner, each of which is subroutine call or a single command.
        """
        config = self.parent.base_env.config
        parsed_action = list()
        rem_action = action
        while rem_action.strip():
            first_match = self._get_first_match(rem_action, pattern_type)
            if first_match:
                pre_action = rem_action[: first_match.start()]
                match_action = rem_action[first_match.start(): first_match.end()]
                rem_action = rem_action[first_match.end():]
                if pre_action.strip():
                    parsed_action.append({"agent": self.name, "action": pre_action, "cmd_name": None, "args": ""})
                if match_action.strip():
                    if match_action.split()[0] == config.submit_command:
                        parsed_action.append(
                            SubAction(
                                agent=self.name, action=match_action, cmd_name=first_match.group(1), args="", )
                        )

                    else:
                        parsed_action.append(
                            SubAction(
                                agent=first_match.group(1),
                                action=match_action,
                                cmd_name=first_match.group(1),
                                args=first_match.group(2), )
                        )
            else:
                parsed_action.append(
                    SubAction(agent=self.name, action=rem_action, cmd_name=None, args="")
                )
                rem_action = ""
        return torch.stack(parsed_action, -1)

    _get_first_match = GuardMultiline._get_first_match


class MultiThoughtActionParser(ParseFunction):
    """
    Expects the model response to be a discussion followed by one or more commands wrapped in backticks.
    Example:
    Let's look at the files in the current directory.
    ```
    ls -l
    ```
    We can also check the current working directory.
    ```
    pwd
    ```
    Multiple code blocks are stacked together into a single action.
    """
    _error_message = """\
    Your output was not formatted correctly. Your commands were not executed. You must always include one discussion and one or more commands as part of your response. 
    Please make sure your output precisely matches the following format:
    DISCUSSION
    Discuss here with yourself about what you are planning and what you are going to do in this step.
    ```
    command(s) that you're going to run
    ```
    """

    def _extract_code_blocks(self, text):
        # Define the pattern to match code blocks
        pattern = r'```([\s\S]*?)```'

        # Find all matches in the text
        matches = re.findall(pattern, text, re.MULTILINE)

        # Strip leading and trailing whitespace from each match
        code_blocks = [block.strip() for block in matches]

        return code_blocks

    def __call__(self, model_response, strict=False):
        """
        Parses the action from the output of the API call.
        We assume that the actions are the code blocks in the model_response.
        We stack multiple code blocks together into a single action.
        """

        # Extract the code blocks
        code_blocks = self._extract_code_blocks(model_response)
        # Check if there are any code blocks
        if not code_blocks:
            msg = "No actions found in model response."
            raise FormatError(msg)
        # Split the model response into thought and action
        parts = model_response.split('```')
        thought = ''
        for i in range(0, len(parts), 2):
            thought += parts[i].strip() + '\n'
        thought = thought.strip()
        action = '\n'.join(code_blocks)
        return thought, action


def make_env(args, tokenizer=None, device="cpu") -> TransformedEnv:
    """Wraps an MLGymEnv in a TorchRL Environment.

    The appended transforms will make sure that the data is formatted for the LLM during (for the outputs of `env.step`)
    and for the MLGym API (for inputs to `env.step`).

    """

    base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"]).unwrapped
    # we need the env to have access to the config
    base_env.config = args.agent.config
    env = TransformedEnv(GymWrapper(base_env, auto_reset=False, device=device), auto_unwrap=False)

    env.append_transform(ConditionalSkip(lambda td: td["retry"]))
    env.append_transform(IsolateCodeBlock())

    env.append_transform(ResetModule())
    env.append_transform(ReadState())
    env.append_transform(StateToMessage())
    env.append_transform(MessageToHistory())
    env.append_transform(TemplateTransform(tokenizer=tokenizer))
    return env
