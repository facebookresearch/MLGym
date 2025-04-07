# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import ast
import gymnasium as gym
import json
import pytest
import re
import sys
import torch
import yaml
from functools import partial
from torchrl.envs import ConditionalSkip, GymWrapper, TransformedEnv
from transformers import GenerationConfig

from mlgym.main import (
    AgentArguments, CONFIG_DIR, EnvironmentArguments, ModelArguments, RichHelpFormatter, multiline_representer, parse,
)
from mlgym.torchrl.models import MetagenModel
from mlgym.torchrl.transforms import (
    CheckActionFormat,
    IsolateCodeBlock,
    MessageToHistory,
    MessageToHistory,
    ReadState,
    ResetModule,
    StateToMessage,
    TemplateTransform,
    wrap_env,
)

DEBUG = False


def make_unsloth_model():
    from unsloth import FastLanguageModel
    from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

    max_seq_length = 10000
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct", max_seq_length=max_seq_length, dtype=None, load_in_4bit=True, )
    FastLanguageModel.for_inference(model)

    return Seq(
        lambda td: td.clear_device_(), Mod(
            lambda msg: tokenizer([msg], return_attention_mask=True, return_tensors="pt"),
            in_keys=["message"],
            out_keys=["tokens_in"]
        ), Mod(lambda tensor: tensor.cuda(), in_keys=["tokens_in"], out_keys=["tokens_in"]), Mod(
            lambda tokens_in: model.generate(
                **tokens_in, stop_strings=["<|im_end|>"], tokenizer=tokenizer, max_new_tokens=10000
            ), in_keys=["tokens_in"], out_keys=["tokens_out"]
        ), Mod(
            lambda tout, tin: tokenizer.decode(tout.squeeze().tolist()),
            in_keys=["tokens_out", ("tokens_in", "input_ids")],
            out_keys=["action"]
        ), partial(printtd, prefix="1.", field="action"), ), tokenizer


def get_args(args=None) -> "ScriptArguments":
    """Parse command line arguments and return a ScriptArguments object.

    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    from sys import path as sys_path
    import pathlib
    sys_path.append(pathlib.Path(__file__).parent.parent)
    from run import ScriptArguments
    defaults = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path="tasks/regressionKaggleHousePrice.yaml",
            max_steps=10,
            seed=42,
            container_type="docker",
            verbose=True, ), agent=AgentArguments(
            model=ModelArguments(
                model_name="metagen:gpt-4-turbo",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95, ), config_file=CONFIG_DIR / "agents" / "better_thought_action_parser_with_insert.yaml", ), )
    yaml.add_representer(str, multiline_representer)

    args = parse(
        ScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description="Run inference.", )

    return args


class TestForward:
    def test_mlgym_env(self):
        args = get_args()
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = GymWrapper(base_env)
        # test forward
        assert env.config is not None
        env.check_env_specs()

    def test_reset(self):
        args = get_args()
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = TransformedEnv(GymWrapper(base_env), ResetModule())
        # test forward
        assert env.config is not None
        env.check_env_specs()
        # Test 2 resets
        env.reset()
        env.reset()

    def test_read_state(self):
        args = get_args()
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = TransformedEnv(GymWrapper(base_env), ResetModule())
        # test forward
        assert env.config is not None
        # These two need each other
        env.append_transform(ReadState())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.check_env_specs()

        # Test another reset
        env.reset()

    def test_read_state_to_message(self):
        args = get_args()
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = TransformedEnv(GymWrapper(base_env), ResetModule())
        # test forward
        assert env.config is not None
        # These two need each other
        env.append_transform(ReadState())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.append_transform(StateToMessage())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.check_env_specs()

        # Test another reset
        env.reset()

    def test_read_msg2history(self):
        args = get_args()
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = TransformedEnv(GymWrapper(base_env), ResetModule())
        # test forward
        assert env.config is not None
        # These two need each other
        env.append_transform(ReadState())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.append_transform(StateToMessage())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.append_transform(MessageToHistory())
        env.transform.transform_output_spec(env.base_env.output_spec)
        env.check_env_specs()

        # Test another reset
        env.reset()


class TestInverse:

    def test_format_template(self):
        torch.manual_seed(0)
        args = get_args()
        model, tokenizer = make_unsloth_model()
        # model = MetagenModel(args.agent.model, access_token="504693128585594|Fj6YNKUmWGH1pC04Rio8bsOAXbc",
        #                      url_base="https://graph.intern.facebook.com/v20.0")
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        assert base_env.config is not None
        env = wrap_env(env)
        env.check_env_specs()

        r = env.reset()
        for _ in range(5):
            print("Executing model", end="\n\n")
            r = model(r)
            print("Executing step", end="\n\n")
            r = env.step(r)
            r = env.step_mdp(r)
        print(r["next", "history"][-1].to_dict())


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    sys.argv = [sys.argv[0]]  # Keep the script name, but remove all other arguments
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
