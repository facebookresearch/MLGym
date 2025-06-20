"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Replay a trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import ArgumentParser

import yaml

import run as runscript


def process_single_traj(
    traj_path: str,
    config_file: str,
    task_id: str,
    suffix: str,
    *,
    forward_args: list[str],
) -> None:
    """

    Args:
        traj_path (str): _description_
        config_file (str): _description_
        suffix (str): _description_
        forward_args (List[str]): Passed to run.py

    Raises:
        ValueError: Incorrect paths or other config issue

    Returns:
        None
    """
    replay_action_trajs_path = "temp_replay.jsonl"

    # Open trajectory file, extract responses as actions
    if traj_path.endswith(".yaml"):
        traj_data = {}
        with open(traj_path) as f:
            traj_data["history"] = yaml.safe_load(f)
    else:
        with open(traj_path) as file:
            traj_data = json.load(file)
    actions = [x["content"] for x in traj_data["history"] if x["role"] == "assistant"]
    with open(replay_action_trajs_path, "w") as f:
        print(json.dumps({task_id: actions}), file=f, end="\n", flush=True)

    # Call run.py via subprocess
    run_args = [
        "--config_file",
        config_file,
        "--model_name",
        "replay",
        "--replay_path",
        replay_action_trajs_path,
        *forward_args,
    ]
    if suffix is not None:
        run_args.extend(["--suffix", suffix])
    script_args = runscript.get_args(run_args)
    runscript.main(script_args)

    os.remove(replay_action_trajs_path)


def main(
    traj_path: str,
    config_file: str,
    task_id: str,
    suffix: str,
    *,
    forward_args: list[str],
) -> None:
    process_single_traj(traj_path, config_file, task_id, suffix, forward_args=forward_args)


def get_args(args: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--traj_path", help="Path to trajectory to replay", required=True)
    parser.add_argument("--config_file", help="Path to template", required=True)
    parser.add_argument(
        "--task_id",
        help="Path to data file containing task instances ref'ed by replay trajectories",
        default=None,
    )
    parser.add_argument("--suffix", help="(Optional) Suffix argument appended to end of traj path", default=None)
    new_args, remaining_args = parser.parse_known_args(args=args)
    return new_args, remaining_args


if __name__ == "__main__":
    args, remaining_args = get_args()
    main(**vars(args), forward_args=remaining_args)
