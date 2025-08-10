"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Evaluation utilities for the MLGym framework.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties, fontManager

MODELS = [
    "gpt4o2",
    "gpt-o3-mini",
    "gpt-o1",
    "deepseek-r1",
    "llama3-405b-tools",
    "llama4-17b-16",
    "llama4-17b-128",
    "claude-35-sonnet-new",
    "claude-37-sonnet",
    "gemini-15-pro",
    "gemini-20-flash-thinking",
    "gemini-25-pro",
    # "gemini-20-pro",
]

ORIGINAL_MODELS = [
    "claude-35-sonnet-new",
    "gemini-15-pro",
    "gpt4o2",
    "gpt-o1",
    "llama3-405b-tools",
]

NEW_MODELS = [
    "deepseek-r1",
    "claude-37-sonnet",
    "gemini-20-flash-thinking",
    # "gemini-20-pro",
    "gemini-25-pro",
    "gpt-o3-mini",
    "llama4-17b-16",
    "llama4-17b-128",
]

MODEL_NAME_MAP = {
    "llama3-405b-tools": "Llama-3.1-405B",
    "deepseek-r1": "DeepSeek-R1",
    "gpt-o1": "O1-preview",
    "claude-35-sonnet-new": "Claude-3.5-Sonnet",
    "claude-37-sonnet": "Claude-3.7-Sonnet",
    "gemini-15-pro": "Gemini-1.5-Pro",
    "gemini-20-flash-thinking": "Gemini-2.0-Flash",
    "gemini-20-pro": "Gemini-2.0-Pro",
    "gemini-25-pro": "Gemini-2.5-Pro",
    "gpt4o2": "GPT-4o",
    "gpt-o3-mini": "O3-mini",
    "llama4-17b-16": "Llama-4-Scout",
    "llama4-17b-128": "Llama-4-Maverick",
}

MODEL_SHORT_NAME_MAP = {
    "llama3-405b-tools": "Llama-405B",
    "deepseek-r1": "R1",
    "gpt-o1": "O1-preview",
    "claude-35-sonnet-new": "Claude-3.5-Sonnet",
    "claude-37-sonnet": "Claude-3.7-Sonnet",
    "gemini-15-pro": "Gemini-1.5-Pro",
    "gemini-20-flash-thinking": "Gemini-2.0-Flash",
    "gemini-20-pro": "Gemini-2.0-Pro",
    "gemini-25-pro": "Gemini-2.5-Pro",
    "gpt4o2": "GPT-4o",
    "gpt-o3-mini": "O3-mini",
    "llama4-17b-16": "Llama-4-Scout",
    "llama4-17b-128": "Llama-4-Maverick",
}

MODEL_COST_MAP = {
    "llama3-405b-tools": {"input_price": 3.5e-06, "output_price": 3.5e-06},
    "deepseek-r1": {"input_price": 0.55e-06, "output_price": 2.19e-06},
    "gpt4o2": {"input_price": 2.5e-06, "output_price": 10e-06},
    "claude-35-sonnet-new": {"input_price": 3e-06, "output_price": 15e-06},
    "claude-37-sonnet": {"input_price": 3e-06, "output_price": 15e-06},
    "gemini-15-pro": {
        "input_price": 1.25e-6,  # $1.25 per 1M tokens for <= 128k
        "output_price": 5e-6,  # $5.00 per 1M tokens for <= 128k
    },
    "gemini-20-flash-thinking": {
        "input_price": 0.10e-6,  # $1.25 per 1M tokens for <= 128k
        "output_price": 0.40e-6,  # $5.00 per 1M tokens for <= 128k
    },
    "gemini-20-pro": {
        "input_price": 1.25e-6,  # $1.25 per 1M tokens for <= 128k
        "output_price": 5e-6,  # $5.00 per 1M tokens for <= 128k
    },
    "gemini-25-pro": {
        "input_price": 1.25e-6,  # $1.25 per 1M tokens for <= 128k
        "output_price": 10e-6,  # $5.00 per 1M tokens for <= 128k
    },
    "gpt-o1": {"input_price": 15e-06, "output_price": 60e-06},
    "gpt-o3-mini": {"input_price": 1.1e-06, "output_price": 4.40e-06},
    "llama4-17b-16": {"input_price": 0.18e-6, "output_price": 0.59e-6},
    "llama4-17b-128": {"input_price": 0.27e-6, "output_price": 0.85e-6},
}

MODEL_LOGOS = {
    "llama3-405b-tools": ("assets/logos/meta-logo.png", 0.15),
    "deepseek-r1": ("assets/logos/deepseek-logo.png", 0.15),
    "gpt-o1": ("assets/logos/openai-logo.png", 0.15),
    "claude-35-sonnet-new": ("assets/logos/anthropic-logo.png", 0.15),
    "claude-37-sonnet": ("assets/logos/anthropic-logo.png", 0.15),
    "gemini-15-pro": ("assets/logos/google-logo.png", 0.15),
    "gemini-20-flash-thinking": ("assets/logos/google-logo.png", 0.15),
    "gemini-20-pro": ("assets/logos/google-logo.png", 0.15),
    "gemini-25-pro": ("assets/logos/google-logo.png", 0.15),
    "gpt4o2": ("assets/logos/openai-green.png", 0.15),
    "gpt-o3-mini": ("assets/logos/openai-logo.png", 0.15),
    "llama4-17b-16": ("assets/logos/meta-logo.png", 0.15),
    "llama4-17b-128": ("assets/logos/meta-logo.png", 0.15),
}

MODEL_MARKER_MAP = {
    "llama3-405b-tools": "-",  # solid line
    "deepseek-r1": (0, (1, 1, 3, 1)),  # dash-dot-dash line
    "claude-35-sonnet-new": "--",  # dashed line
    "claude-37-sonnet": (0, (3, 1, 1, 1)),  # dash-dot-dot line
    "gemini-15-pro": ":",  # dash-dot line
    "gemini-20-flash-thinking": (0, (1, 1)),  # densely dotted line
    "gemini-20-pro": (0, (5, 1, 1, 1, 1, 1)),  # dash-dot-dot-dot line
    "gemini-25-pro": (0, (3, 1, 3, 1, 1, 1)),  # dash-dot-dash-dot line
    "gpt4o2": "-.",  # dotted line
    "gpt-o1": (5, (10, 3)),  # custom pattern
    "gpt-o3-mini": (0, (7, 3, 1, 3)),  # dash-dot with longer dashes
    "llama4-17b-16": (0, (5, 2, 1, 2, 1, 2)),  # complex dash-dot pattern
    "llama4-17b-128": (0, (3, 3, 1, 3, 1, 3, 1, 3)),  # alternating dash-dot pattern
}

EXIT_STATUS_MAP: dict[str, list[str]] = {
    "autosubmission (exit_context)": ["Context"],
    "autosubmission (max_steps)": ["Max Steps"],
    "submission_not_found (exit_format)": ["Evaluation", "Format"],
    "submission_not_found (exit_context)": ["Evaluation", "Context"],
    'unknown_error (open "data/train.csv" 0)': ["Permission"],
    "autosubmission (exit_cost)": ["Cost"],
    "autosubmission (exit_format)": ["Format"],
    "submission_not_found (max_steps)": ["Evaluation"],
    "evaluation_format_error (submit)": ["Evaluation", "Submit"],
    "evaluation_format_error (exit_format)": ["Evaluation", "Format"],
    "unknown_error (validate)": ["Evaluation"],
    "evaluation_format_error (max_steps)": ["Evaluation"],
    "submission_not_found (submit)": ["Evaluation"],
    "unknown_error (ls -R data/)": ["Runtime"],
    "unknown_error (ls data/train/)": ["Runtime"],
    "evaluation_format_error (exit_cost)": ["Evaluation", "Cost"],
    "evaluation_format_error (exit_context)": ["Evaluation", "Context"],
    "unknown_error (torchrun --nproc_per_node=2 --standalone baseline.py)": ["Runtime"],
    "unknown_error (torchrun --nproc_per_node=1 --standalone baseline.py)": ["Runtime"],
    "unknown_error (python train.py)": ["Runtime"],
    "unknown_error (python baseline.py)": ["Runtime"],
    "unknown_error (python evaluate.py)": ["Runtime"],
    "autosubmission (exit_api)": ["API"],
    "early_exit": ["Runtime"],
    "submitted": ["Success"],
}

# EXIT_STATUS_MAP = {
#     "autosubmission (exit_api)": "API",
#     "autosubmission (exit_context)": "Context",
#     "autosubmission (exit_cost)": "Cost",
#     "autosubmission (exit_format)": "Format",
#     "autosubmission (max_steps)": "Max Steps",
#     "early_exit": "Runtime",
#     "evaluation_format_error (exit_cost)": "Cost",
#     "evaluation_format_error (exit_format)": "Format",
#     "evaluation_format_error (max_steps)": "Evaluation",
#     "evaluation_format_error (submit)": "Evaluation",
#     "submission_not_found (max_steps)": "Evaluation",
#     "submitted": "Success",
#     'unknown_error (open "data/train.csv" 0)': "Permission",
#     "unknown_error (torchrun --nproc_per_node=1 --standalone baseline.py)": "Runtime",
#     "unknown_error (validate)": "Evaluation",
#     "submission_not_found (submit)": "Evaluation",
#     "unknown_error (ls -R data/)": "Runtime",
#     "unknown_error (ls data/train/)": "Runtime",
#     "submission_not_found (exit_format)": "Evaluation",
#     "unknown_error (python train.py)": "Runtime",
#     "unknown_error (python baseline.py)": "Runtime",
#     "unknown_error (python)": "Runtime",
#     "unknown_error (submit)": "Evaluation",
#     "unknown_error (insert)": "Format",
#     "unknown_error (search)": "Format",
#     "unknown_error (view)": "Format",
#     "unknown_error (edit)": "Format",
# }

ACTION_LIST = ["Edit", "View", "Validate", "Submit", "Search", "Python", "Bash"]

PAUL_TOL_REORDERED = [
    "#4477AA",  # blue
    "#228833",  # green
    "#AA3377",  # purple
    "#EE6677",  # red
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#BBBBBB",  # grey
]
PAUL_TOL_REORDERED_NAMES = ["blue", "green", "purple", "red", "yellow", "cyan", "grey"]

# Colors from the image - exact hex codes from the octagon shapes
PAUL_TOL_MUTED = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]

PAUL_TOL_MUTED_EXTENDED = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DD77CC",  # pinkish-mauve (new, complements rose and purple)
    "#7799DD",  # soft blue (new, complements indigo)
    "#66AA55",  # light muted green (new, between green and teal)
]

# Define PAUL_TOL that was missing
PAUL_TOL = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

# MODELS = ["llama3-405b-tools", "gpt4o2", "claude-35-sonnet-new", "gemini-15-pro", "gpt-o1"]

COLOR_CHOICE = PAUL_TOL_MUTED_EXTENDED
MODEL_COLOR_MAP = dict(zip(MODELS, COLOR_CHOICE, strict=False))
# MODEL_COLOR_MAP = {
#     "llama3-405b-tools": "#4477AA",
#     "gpt4o2": "#EE6677",
#     "claude-35-sonnet-new": "#AA3377",
#     "gemini-15-pro": "#228833",
#     "gpt-o1": "#CCBB44",
# }

ACTION_COLOR_MAP = dict(zip(ACTION_LIST, PAUL_TOL, strict=False))

# Action color mapping using seaborn deep palette for consistency across plots
ACTION_COLOR_MAP_DEEP = dict(zip(ACTION_LIST, sns.color_palette("deep", n_colors=len(ACTION_LIST)), strict=False))

# Exit status color mapping using darker colors from PAUL_TOL_MUTED_EXTENDED
EXIT_STATUS_COLOR_MAP = {
    "Permission": "#332288",  # indigo (darkest)
    "Evaluation": "#CC6677",  # rose
    "Cost": "#44AA99",  # teal
    "Format": "#882255",  # wine (dark)
    "Context": "#117733",  # green (dark)
    "Runtime": "#999933",  # olive
    "Submit": "#AA4499",  # purple
}

TASKS = {
    "regressionKaggleHousePrice": {
        "name": "House Price",
        "shortname": "Regression",
        "priority_metric": "r2",
        "metric_direction": "maximize",
    },
    "3SATTime": {
        "name": "3-SAT",
        "shortname": "3-SAT",
        "priority_metric": "Time",
        "metric_direction": "minimize",
    },
    "imageClassificationCifar10": {
        "name": "CIFAR-10",
        "shortname": "CIFAR-10",
        "priority_metric": "accuracy",
        "metric_direction": "maximize",
    },
    "imageClassificationFMnist": {
        "name": "Fashion MNIST",
        "shortname": "F-MNIST",
        "priority_metric": "accuracy",
        "metric_direction": "maximize",
    },
    "imageCaptioningCOCO": {
        "name": "MS-COCO",
        "shortname": "MS-COCO",
        "priority_metric": "BLEU Score",
        "metric_direction": "maximize",
    },
    "languageModelingFineWeb": {
        "name": "Language Modeling",
        "shortname": "FineWeb",
        "priority_metric": "val_loss",
        "metric_direction": "minimize",
    },
    "naturalLanguageInferenceMNLI": {
        "name": "MNLI",
        "shortname": "MNLI",
        "priority_metric": "validation_accuracy",
        "metric_direction": "maximize",
    },
    "battleOfSexes": {
        "name": "Battle of Sexes",
        "shortname": "BoS",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "prisonersDilemma": {
        "name": "Prisoners Dilemma",
        "shortname": "PD",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "blotto": {
        "name": "Blotto",
        "shortname": "Blotto",
        "priority_metric": "Score",
        "metric_direction": "maximize",
    },
    "rlBreakoutMinAtar": {
        "name": "Breakout",
        "shortname": "Breakout",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
    "rlMetaMazeMisc": {
        "name": "Meta Maze",
        "shortname": "Maze",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
    "rlMountainCarContinuous": {
        "name": "Mountain Car Continuous",
        "shortname": "MountainCar",
        "priority_metric": "Reward Mean",
        "metric_direction": "maximize",
    },
}


def set_custom_font() -> None:
    """Set the custom font for the plots."""
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    if not Path(font_path).exists():
        return
    fontManager.addfont(font_path)
    prop = FontProperties(fname=PosixPath(font_path))
    sns.set_theme(font=prop.get_name(), style="white", palette="pastel")
    plt.rcParams["font.family"] = prop.get_name()

    # Remove plot borders/spines
    spines = ["top", "right", "bottom", "left"]
    for spine in spines:
        plt.rcParams[f"axes.spines.{spine}"] = True

    # Set spine weight for better visibility
    plt.rcParams["axes.linewidth"] = 0.5  # Set spine width

    # # Improve bar appearance
    # plt.rcParams["patch.edgecolor"] = "black"  # Remove bar edges
    # plt.rcParams["patch.force_edgecolor"] = True
    # plt.rcParams["patch.linewidth"] = 0.5

    # Subtle grid for readability
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.3

    # Fix figure facecolor and edgecolor
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["figure.edgecolor"] = "none"


def get_fig_size() -> None:
    """Set the figure size for the plots."""
    fig_width_pt = 472.03123  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    print(f"Figure size: {fig_size}")


def get_best_attempt(results: dict, priority_metric: str, metric_direction: str) -> int:
    """Returns the index of the best agent according to the priority metric."""
    best_attempt_idx = -1
    best_metric_value = -float("inf") if metric_direction == "maximize" else float("inf")
    for i, result in enumerate(results):
        if priority_metric not in result:
            print(f"Priority metric {priority_metric} not found in result for agent {i}")
            continue
        metric_value = result[priority_metric]
        if (metric_direction == "maximize" and metric_value > best_metric_value) or (
            metric_direction == "minimize" and metric_value < best_metric_value
        ):
            best_metric_value = metric_value
            best_attempt_idx = i
    return best_attempt_idx


def get_best_scores(results: dict, priority_metric: str, metric_direction: str, models: list[str]) -> dict:
    """Computes the best attempts for all models on a task"""
    all_scores: dict[str, dict[str, list[float]]] = {model: {} for model in [*models, "baseline"]}

    for model in models:
        best_attempts = []
        best_submissions = []
        for score in results[model]["scores"]:
            best_attempt_idx = get_best_attempt(score["agent"], priority_metric, metric_direction)
            best_attempts.append(score["agent"][best_attempt_idx])
            best_submissions.append(score["agent"][-1][priority_metric])
        all_scores[model]["best_attempts"] = best_attempts
        all_scores[model]["best_submissions"] = best_submissions

        if metric_direction == "maximize":
            all_scores[model]["overall_best_submission"] = np.max(best_submissions)
            all_scores[model]["overall_best_attempt"] = np.max(best_attempts)
        else:
            all_scores[model]["overall_best_submission"] = np.min(best_submissions)
            all_scores[model]["overall_best_attempt"] = np.min(best_attempts)

    all_scores["baseline"] = {"overall_best_submission": results["scores"][0]["baseline"][priority_metric]}

    return all_scores


def process_trajectories(traj_parent_dir: str, traj_pattern: str, task_id: str, models: list[str]) -> dict:
    """
    Get all results.json and .traj files from the trajectory directory pattern for a given task
    """
    all_results = {}
    for model in models:
        model_results: dict[str, list[dict | list[str]]] = {"scores": [], "trajectories": [], "exit_statuses": []}
        traj_dir_pattern = f"{traj_parent_dir}/*{model}__{task_id}__{traj_pattern}*"
        traj_dirs = sorted(Path().glob(traj_dir_pattern))
        for traj_dir in traj_dirs:
            results_file = Path(traj_dir) / "results.json"
            traj_file = list(Path().glob(f"{traj_dir}/*.traj"))
            with open(results_file) as f:
                results = json.load(f)
            with open(traj_file[0]) as f:
                traj = json.load(f)
            exit_status = "unknown_error"

            # get last action
            last_action = "unknown"
            for history in reversed(traj["history"]):
                if history["role"] == "assistant":
                    last_action = history["action"].strip()
                    break

            exit_status = traj["info"]["exit_status"]
            # if exit_status == "":
            #     if "open" in last_action:
            #         exit_status = "unknown_error (open)"
            #     elif (
            #         "python" in last_action
            #         or "torchrun" in last_action
            #         or "deepspeed" in last_action
            #     ):
            #         exit_status = "unknown_error (python)"
            #     elif "edit" in last_action:
            #         exit_status = "unknown_error (edit)"
            #     elif "validate" in last_action:
            #         exit_status = "unknown_error (validate)"
            #     elif "submit" in last_action:
            #         exit_status = "unknown_error (submit)"
            #     elif "insert" in last_action:
            #         exit_status = "unknown_error (insert)"
            #     elif "search" in last_action:
            #         exit_status = "unknown_error (search)"
            #     elif "view" in last_action:
            #         exit_status = "unknown_error (view)"
            #     else:
            #         exit_status = f"unknown_error ({last_action})"
            if exit_status == "":
                exit_status = f"unknown_error ({last_action})"

            exit_statuses: list[str] = EXIT_STATUS_MAP.get(exit_status) or [exit_status]
            # exit_statuses = [exit_status]

            model_results["scores"].append(results)
            model_results["trajectories"].append(traj["trajectory"])
            model_results["exit_statuses"].append(exit_statuses)

        all_results[model] = model_results

    return all_results


def get_action_results(trajectories: dict) -> dict:  # noqa: C901
    """
    Get the number of times each action is taken across all tasks and models.

    Args:
        trajectories (dict): Dictionary containing trajectories for each task and model.
            Structure: {task_id: {model_id: {"trajectories": [trajectory_list], ...}, ...}, ...}
            Each trajectory_list contains a list of actions taken during that run.

    Returns:
        dict: Dictionary with action counts.
            Structure: {action_name: count, ...}
    """
    action_counts: dict[str, int] = defaultdict(int)
    actions_per_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    actions_per_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    actions_per_step: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    file_editing = ["create", "edit", "insert"]
    file_viewer = ["open", "goto", "scroll_up", "scroll_down"]
    validation = ["validate"]
    submit = ["submit"]
    search = ["search_dir", "search_file", "find_file"]
    python_scripts = ["torchrun", "python", "python3", "accelerate", "deepspeed"]

    for task_id, task_results in trajectories.items():
        for model_id, model_results in task_results.items():
            for trajectory in model_results["trajectories"]:
                for step_idx, step in enumerate(trajectory):
                    # Clean up the action string by removing whitespace
                    action = step["action"].strip()
                    # Map actions to categories
                    if any(action.startswith(cmd) for cmd in file_editing):
                        action = "Edit"
                    elif any(action.startswith(cmd) for cmd in file_viewer):
                        action = "View"
                    elif any(action.startswith(cmd) for cmd in validation):
                        action = "Validate"
                    elif any(action.startswith(cmd) for cmd in submit):
                        action = "Submit"
                    elif any(action.startswith(cmd) for cmd in search):
                        action = "Search"
                    elif any(action.startswith(cmd) for cmd in python_scripts):
                        action = "Python"
                    else:
                        action = "Bash"

                    action_counts[action] += 1
                    actions_per_model[model_id][action] += 1
                    actions_per_task[task_id][action] += 1
                    actions_per_step[step_idx + 1][action] += 1

    # Convert defaultdict to regular dict
    return {
        "action_counts": dict(action_counts),
        "actions_per_model": dict(actions_per_model),
        "actions_per_task": dict(actions_per_task),
        "actions_per_step": {step: dict(actions) for step, actions in actions_per_step.items()},
    }


def get_exit_status_results(trajectories: dict) -> dict[str, dict]:
    """
    Get the number of times each exit status occurs across all tasks and models.
    """
    total_es_counts: dict[str, int] = defaultdict(lambda: 0)
    es_counts_per_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    es_counts_per_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))

    # no agent scores
    failed_runs_per_model: dict[str, int] = defaultdict(lambda: 0)

    # agent scores but failed to submit at the end of the run
    incomplete_runs_per_model: dict[str, int] = defaultdict(lambda: 0)
    failed_runs_per_task: dict[str, int] = defaultdict(lambda: 0)
    incomplete_runs_per_task: dict[str, int] = defaultdict(lambda: 0)

    for task_id, task_results in trajectories.items():
        model_num = 0
        for model_id, model_results in task_results.items():
            assert len(model_results["exit_statuses"]) == len(model_results["scores"]), (
                f"Exit statuses and scores length mismatch for model {model_id} in task {task_id}"
            )
            for exit_status, score in zip(model_results["exit_statuses"], model_results["scores"], strict=False):
                success_status = False
                for es in exit_status:
                    total_es_counts[es] += 1
                    es_counts_per_model[model_id][es] += 1
                    es_counts_per_task[task_id][es] += 1
                    success_status = success_status or es in ["Success", "Max Steps"]

                failed = 0
                incomplete = 0
                if "agent" not in score or ("agent" in score and len(score["agent"]) == 0):
                    failed = 1
                elif "agent" in score and len(score["agent"]) > 0 and not success_status:
                    incomplete = 1

                failed_runs_per_model[model_id] += failed
                incomplete_runs_per_model[model_id] += incomplete
                failed_runs_per_task[task_id] += failed
                incomplete_runs_per_task[task_id] += incomplete

                model_num += 1

    print("All exit statuses:\n")
    print(f"{total_es_counts.keys()}")

    return {
        "total_es_counts": total_es_counts,
        "es_counts_per_model": es_counts_per_model,
        "es_counts_per_task": es_counts_per_task,
        "failed_runs_per_model": failed_runs_per_model,
        "incomplete_runs_per_model": incomplete_runs_per_model,
        "failed_runs_per_task": failed_runs_per_task,
        "incomplete_runs_per_task": incomplete_runs_per_task,
    }
