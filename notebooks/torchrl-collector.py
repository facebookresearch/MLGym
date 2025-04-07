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
from functools import partial
from torchrl.envs import GymWrapper, TransformedEnv
from transformers import GenerationConfig

from mlgym.main import (
    AgentArguments, CONFIG_DIR, EnvironmentArguments, ModelArguments, RichHelpFormatter, multiline_representer, parse,
)
from mlgym.torchrl.models import MetagenModel
from torchrl.envs import StepCounter, Timer
from torchrl.collectors import SyncDataCollector
from mlgym.torchrl.transforms import wrap_env
from torch import multiprocessing as mp
from torchrl.data.utils import CloudpickleWrapper
from torchrl.data import ReplayBuffer, LazyStackStorage
DEBUG = False

def clear_device(td):
    td.clear_device_()
    return td
def cuda(tensor):
    return tensor.cuda()
    
def make_unsloth_model():
    from unsloth import FastLanguageModel
    from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

    max_seq_length = 50000
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length, 
        dtype=None, 
        load_in_4bit=True, 
    )
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)

    return Seq({
        "clear_device": clear_device,
        "encode": Mod(
            tokenizer,
            in_keys=["chat_str"],
            out_keys=["tokens_in"],
            method_kwargs=dict(return_attention_mask=True, return_tensors="pt", padding=True, padding_side="left"),
        ), 
        "to_cuda": Mod(cuda, in_keys=["tokens_in"], out_keys=["tokens_in"]), 
        "generate": Mod(
            model,
            method="generate",
            method_kwargs=dict(
                stop_strings=["<|im_end|>"], tokenizer=tokenizer, max_new_tokens=10000,
                top_p=0.95,
            ), 
            in_keys={"input_ids": ("tokens_in", "input_ids"), "attention_mask": ("tokens_in", "attention_mask")}, 
            out_keys=["tokens_out"],
            out_to_in_map=True,
        ),
        "decode": Mod(
            lambda tout: tokenizer.decode(tout.squeeze().tolist()),
            in_keys=["tokens_out"],
            out_keys=["action"]
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
    # mp.set_start_method("spawn")
    
    torch.manual_seed(0)
    args = get_args(["--container_type", "podman", "--aliases_file", "./docker/aliases_podman.sh", "--max_steps", "1000"])
    traj_dir = args.get_trajectory_dir(0)
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = make_unsloth_model()
    # A LazyStackStorage is a storage that does not forces the data to be contiguous
    # Since our data will be ragged (different number of tokens / event), this is the preferred format
    buffer = ReplayBuffer(storage=LazyStackStorage(1000), batch_size=32)
    
    def make_env(gym=gym, wrap_env=wrap_env, tokenize=tokenizer):
        args = get_args(["--container_type", "podman", "--aliases_file", "./docker/aliases_podman.sh", "--max_steps", "1000"])
        base_env = gym.make(f"mlgym/{args.environment.task.id}", devices=["cpu_0"], agent_args=args.agent).unwrapped
        env =  wrap_env(base_env, tokenizer, device="cpu")
        env = env.append_transform(StepCounter())
        env = env.append_transform(Timer())
        return env

    c = SyncDataCollector(
        make_env,
        policy=model,
        policy_device="cuda:0",
        env_device="cpu",
        storing_device="cpu",
        frames_per_batch=5, # 1 frame = 1 answer from the model
        total_frames=50,
        init_random_frames=0,
    )

    t0 = time.time()
    for i, r in enumerate(c):
        # We should be collecting 5 answers at a time
        # The collector will reset the env automatically

        # populate buffer
        buffer.extend(r)

        # sample from buffer
        s = buffer.sample()

    print(f"Time elapsed: {time.time() - t0: 4.4f}")
