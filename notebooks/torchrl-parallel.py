import os

os.environ["AUTO_UNWRAP_TRANSFORMED_ENV"] = "0"
os.environ["CAPTURE_NONTENSOR_STACK"] = "0"

import argparse
import ast
import gymnasium as gym
import time
import json
import pytest
import re
import sys
import torch
import yaml
from pprint import pprint
from functools import partial
from torchrl.envs import GymWrapper, TransformedEnv
from transformers import GenerationConfig

from mlgym.main import (
    AgentArguments, CONFIG_DIR, EnvironmentArguments, ModelArguments, RichHelpFormatter, multiline_representer, parse,
)
from mlgym.torchrl.models import MetagenModel
from torchrl.envs import ParallelEnv, StepCounter, Timer
from mlgym.torchrl.transforms import wrap_env
from transformers import AutoModelForCausalLM, AutoTokenizer
DEBUG = False


def make_transformer_model():
    from unsloth import FastLanguageModel
    from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

    max_seq_length = 50000
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"

    return Seq({
        "clear_device": lambda td: td.clear_device_(), 
        "encode": Mod(
            tokenizer,
            in_keys=["chat_str"],
            out_keys=["tokens_in"],
            method_kwargs=dict(
                return_attention_mask=True,
                return_tensors="pt",
                padding=True,
                padding_side="left"
            ),
            strict=True,
        ), 
        "to_cuda": Mod(
            lambda tensor: tensor.cuda(), 
            in_keys=["tokens_in"], 
            out_keys=["tokens_in"], 
            strict=True
        ),
        "generate": Mod(
            model,
            method="generate",
            method_kwargs=dict(
                stop_strings=["<|im_end|>"], 
                tokenizer=tokenizer, 
                max_new_tokens=10000,
                top_p=0.95,
            ), 
            in_keys={
                "input_ids": ("tokens_in", "input_ids"),
                "attention_mask": ("tokens_in", "attention_mask"),
            },
            out_keys=["tokens_out"],
            out_to_in_map=True,
            strict=True,
        ),
        "decode": Mod(
            tokenizer.batch_decode,
            in_keys=["tokens_out"],
            out_keys=["action"],
            strict=True,
        ), 
        "to_cpu": lambda td: td.cpu(),
    }), tokenizer


def get_args(args=None) -> "ScriptArguments":
    """Parse command line arguments and return a ScriptArguments object.

    Args:
        args: Optional list of arguments to parse. If not provided, uses sys.argv.
    """
    # from sys import path as sys_path
    import pathlib
    # sys_path.append(pathlib.Path(__file__).parent.parent)
    from run import ScriptArguments
    defaults = ScriptArguments(
        environment=EnvironmentArguments(
            task_config_path="tasks/prisonersDilemma.yaml",
            max_steps=10,
            seed=42,
            container_type="podman",
            verbose=True, 
            aliases_file="../docker/aliases_podman.sh",
        ), 
        agent=AgentArguments(
            model=ModelArguments(
                model_name="metagen:gpt-4-turbo",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95, ), 
            config_file=CONFIG_DIR / "agents" / "better_thought_action_parser_with_insert.yaml", ), 
    )
    yaml.add_representer(str, multiline_representer)

    args = parse(
        ScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description="Run inference.", )

    return args

if __name__ == "__main__":
    
    torch.manual_seed(0)
    args = get_args(["--container_type", "podman", "--aliases_file", "./docker/aliases_podman.sh", "--max_steps", "1000"])
    traj_dir = args.get_trajectory_dir(0)
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = make_transformer_model()
    
    def make_env(gym=gym, wrap_env=wrap_env, tokenize=tokenizer):
        args = get_args(["--container_type", "podman", "--aliases_file", "./docker/aliases_podman.sh", "--max_steps", "1000"])
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        return wrap_env(base_env, tokenizer, device="cpu")
    env = ParallelEnv(
        4,
        make_env,
        use_buffers=False,
        mp_start_method="spawn",
        consolidate=False,
        device="cpu"
    )
    env = env.append_transform(StepCounter())
    env = env.append_transform(Timer())

    r = env.reset()

    for _ in range(5):
        print("Executing model", end="\n\n")
        r = model(r)
        print("Executing step", end="\n\n")
        r = env.step(r.cpu())
        r = env.step_mdp(r)
        print("time step", r["time_step"])
        print("time policy", r["time_policy"])
    pprint(list(zip(r[0]["history"].role, r[0]["history"].content)))
    
    print("rollout")

    # An optional callback to get a glimpse on what is going on with our data at each step
    t0 = time.time()
    def callback(self, td):
        t1 = time.time()
        print(f"Time until step {td["step_count"]}: {t1-t0:4.4f}")
        print("time step", td["time_step"])
        print("time policy", td["time_policy"])
        return

    rollout = env.rollout(10, model, callback=callback)

    print(rollout)
    env.close()
