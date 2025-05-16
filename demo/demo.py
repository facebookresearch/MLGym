"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MLGym Trajectory Visualizer

This module provides a Streamlit-based web application for visualizing MLGym trajectories.
It allows users to inspect step-by-step progression of agents through various ML tasks,
including their thought processes, actions taken, execution results, and raw logs.

Usage:
    streamlit run trajectory_visualizer.py [--trajectory_dir PATH]
"""

import os
import json
import streamlit as st
import argparse
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict
import re

# --- Configuration & Styling ---
def configure_page_style() -> None:
    """Configure the Streamlit page layout and apply custom CSS styling."""
    st.set_page_config(
        page_title="MLGym Trajectory Visualizer",
        page_icon="👩‍🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* Base colors */
        :root {
            --slate-50: #f8fafc;
            --slate-100: #f1f5f9;
            --slate-200: #e2e8f0;
            --slate-300: #cbd5e1;
            --slate-400: #94a3b8;
            --slate-500: #64748b;
            --slate-600: #475569;
            --slate-700: #334155;
            --slate-800: #1e293b;
            --slate-900: #0f172a;
            --slate-950: #020617;
            --blue-500: #3b82f6;
            --green-500: #22c55e;
            --purple-500: #a855f7;
            --amber-500: #f59e0b; /* For raw logs */
        }

        .stApp {
            background: linear-gradient(135deg, var(--slate-900) 0%, var(--slate-800) 100%);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--slate-50) !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 1.5rem;
        }

        /* Step indicator */
        .step-indicator {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--slate-50);
            margin: 2rem 0 1rem 0;
        }

        .step-caption {
            font-size: 1.25rem;
            color: var(--slate-300);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            font-weight: normal;
        }

        /* Content Boxes */
        .content-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            width: 100%;
        }

        /* Add margin after progress bar */
        .stProgress {
            margin-bottom: 1rem;
        }

        .thought-box { border-top: 4px solid var(--blue-500); }
        .action-box { border-top: 4px solid var(--green-500); }
        .result-box { border-top: 4px solid var(--purple-500); }
        .rawlogs-box { border-top: 4px solid var(--amber-500); } /* New style for raw logs */

        /* Headers */
        .box-header {
            font-size: 1.5rem !important;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--slate-700);
        }

        .thought-header { color: var(--blue-500) !important; }
        .action-header { color: var(--green-500) !important; }
        .result-header { color: var(--purple-500) !important; }
        .rawlogs-header { color: var(--amber-500) !important; } /* New style for raw logs */


        /* Content styling */
        .box-content {
            padding: 0 1rem;
            font-size: 1.125rem;
            line-height: 1.75;
            color: var(--slate-200);
        }
        .box-content pre code {
            white-space: pre-wrap; /* Ensure long lines wrap */
            word-break: break-all; /* Break words if necessary */
        }

        /* Sidebar styling */
        .sidebar-section { /* This class was defined but not used, can be applied if needed */
            background: linear-gradient(to right, var(--slate-800) 0%, var(--slate-900) 100%);
            border-left: 6px solid var(--green-500);
            padding: 1.5rem;
            border-radius: 0 1rem 1rem 0;
            margin: 2rem 0;
        }
        .stButton button { /* Style sidebar buttons */
            text-align: left !important;
            white-space: normal; /* Allow text to wrap in button */
            word-break: break-word;
        }

        /* Task card */
        .task-card {
            background: var(--slate-800);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--slate-700);
        }

        /* Content Summary Styling */
        .summary-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 1rem;
            margin-bottom: 0.1em; /* Reduced margin */
            width: 100%;
            font-size: 1.3rem; /* Increased font size */
        }
        .summary-box details summary {
            cursor: pointer;
        }
        .summary-box pre code {
            font-size: 0.9rem; /* Smaller font for pre content */
        }
        
    </style>
    """,
        unsafe_allow_html=True,
    )

# --- Argument Parsing ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLGym Trajectory Visualizer")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=os.path.join(os.getcwd(), "trajectories"),
        help="Directory containing trajectory files",
    )
    return parser.parse_known_args()[0]

# --- Trajectory File Parsing & Loading ---

def parse_trajectory_path(filepath_str: str, root_dir_str: str) -> Dict[str, Any]:
    """Parses a trajectory filepath to extract metadata."""
    filepath = Path(filepath_str)
    root_dir = Path(root_dir_str)
    
    data = {
        "filepath": filepath_str,
        "filename": filepath.name,
        "model_name": "Unknown",
        "task_name": "Unknown",
        "run_name": "Unknown",
        "seed": None,
        "display_name": filepath.name, # Default display name
    }

    try:
        relative_path = filepath.relative_to(root_dir)
        # path_parts: e.g., ('bhavul', 'meta-claude-35-sonnet__...', 'contrastiveLearningCifar10Accuracy.traj')
        path_parts = relative_path.parts

        if len(path_parts) > 1:
            # The folder containing the .traj file is usually path_parts[-2]
            # This folder name often is: MODEL_NAME__TASK_NAME__PARAMS__RUN_NAME_WITH_SEED_AND_DATE
            run_folder_name = path_parts[-2]
            
            # The actual task name is often the stem of the .traj file
            data["task_name"] = filepath.stem

            folder_parts = run_folder_name.split('__')
            if len(folder_parts) >= 1:
                data["model_name"] = folder_parts[0]
            if len(folder_parts) >= 2:
                # task_name_from_folder = folder_parts[1] # This can also be used/checked
                pass # data["task_name"] already set from filename stem
            
            # Run name is usually the last part of the folder_parts, or the whole folder_parts if complex
            # For the example, it's the last major segment before date/seed specifics if using full folder name as run_name
            # Or more simply, the full unique folder name is the "run_name"
            data["run_name"] = run_folder_name

            seed_match = re.search(r'_s(\d+)', run_folder_name)
            if seed_match:
                data["seed"] = int(seed_match.group(1))
        
        # Create a nicer display name
        seed_str = f" (Seed {data['seed']})" if data['seed'] is not None else ""
        data["display_name"] = f"{data['model_name']} / {data['task_name']} / {data['run_name']}{seed_str}"
        # If the run_name is too long because it's the full folder, shorten it.
        # Heuristic: if run_name contains model and task, just show the unique part.
        shorter_run_name = data["run_name"].replace(data["model_name"], "").replace(data["task_name"], "").strip("_ ")
        if shorter_run_name and len(shorter_run_name) < len(data["run_name"]):
             data["display_name"] = f"{data['model_name']} / {data['task_name']} / {shorter_run_name}{seed_str}"
        else: # Fallback to relative path if parsing is too complex or parts are missing
            data["display_name"] = str(relative_path)


    except ValueError: # filepath might not be under root_dir, or other parsing issue
        data["display_name"] = str(filepath.parent.name) + "/" + filepath.name


    return data


def find_trajectory_files(root_dir: str) -> List[Dict[str, Any]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        st.error(f"Directory not found: {root_dir}")
        return []

    traj_files_data = []
    for file in root_path.rglob("*.traj"):
        parsed_data = parse_trajectory_path(str(file.resolve()), str(root_path.resolve()))
        traj_files_data.append(parsed_data)
    
    # Sort trajectories: by task, then model, then seed, then run_name
    traj_files_data.sort(key=lambda x: (x["task_name"], x["model_name"], x["seed"] if x["seed"] is not None else -1, x["run_name"]))
    return traj_files_data


def extract_step_from_model_input_block(block_lines: List[str]) -> Optional[int]:
    """Helper to parse (Current Step: K, ...) from a block of text."""
    for line in block_lines:
        match = re.search(r'\(Current Step: (\d+),', line)
        if match:
            return int(match.group(1))
    return None


def get_log_file_path(trajectory_file_path_str: str) -> Optional[Path]:
    """Finds the log file associated with a given trajectory file path."""
    traj_file_path_obj = Path(trajectory_file_path_str)
    log_dir = traj_file_path_obj.parent
    found_log_path: Optional[Path] = None

    # Attempt 1: Find log files with 'run-' in their name and ending with '.log'
    candidate_log_files_with_run = [
        item for item in log_dir.iterdir()
        if item.is_file() and "run-" in item.name.lower() and item.name.lower().endswith(".log")
    ]

    if len(candidate_log_files_with_run) == 1:
        found_log_path = candidate_log_files_with_run[0]
    elif len(candidate_log_files_with_run) > 1:
        exact_match_candidates = [f for f in candidate_log_files_with_run if f.stem == traj_file_path_obj.stem]
        if len(exact_match_candidates) == 1:
            found_log_path = exact_match_candidates[0]
        else:
            # Sort by name to have a deterministic choice if still ambiguous
            candidate_log_files_with_run.sort(key=lambda p: p.name)
            found_log_path = candidate_log_files_with_run[0]
            # User will be warned by the calling function if multiple were ambiguous
    elif len(candidate_log_files_with_run) == 0:
        # Attempt 2: No 'run-' files, so look for any '.log' file.
        all_log_files_in_dir = [
            item for item in log_dir.iterdir()
            if item.is_file() and item.name.lower().endswith(".log")
        ]
        if len(all_log_files_in_dir) == 1:
            found_log_path = all_log_files_in_dir[0]
        elif len(all_log_files_in_dir) > 1:
            exact_match_candidates = [f for f in all_log_files_in_dir if f.stem == traj_file_path_obj.stem]
            if len(exact_match_candidates) == 1:
                found_log_path = exact_match_candidates[0]
            else:
                default_log_path = traj_file_path_obj.with_suffix(".log")
                if default_log_path.exists() and default_log_path in all_log_files_in_dir:
                    found_log_path = default_log_path
                elif all_log_files_in_dir:
                    all_log_files_in_dir.sort(key=lambda p: p.name) # Sort for determinism
                    found_log_path = all_log_files_in_dir[0]
    return found_log_path

def load_and_segment_raw_logs(log_filepath: Path) -> Tuple[str, Dict[int, str]]:
    """Loads a .log file and segments it by trajectory step."""
    preamble_logs_str = ""
    step_log_segments: Dict[int, str] = {}

    if not log_filepath.exists():
        st.warning(f"Log file not found: {log_filepath}")
        return preamble_logs_str, step_log_segments

    with open(log_filepath, 'r', encoding='utf-8') as f:
        all_log_lines = f.readlines()

    model_input_markers = [] # Stores {"step": K, "index": line_index}
    for i, line in enumerate(all_log_lines):
        if "TRACE 🤖 MODEL INPUT (primary_0)" in line:
            # Look ahead a few lines for (Current Step: K, ...)
            # Max 20 lines lookahead for step info robustness
            current_step_in_block = extract_step_from_model_input_block(all_log_lines[i : i + 20])
            if current_step_in_block is not None:
                model_input_markers.append({"step": current_step_in_block, "index": i})
    
    model_input_markers.sort(key=lambda x: x["index"]) # Ensure they are in order of appearance

    if not model_input_markers:
        preamble_logs_str = "".join(all_log_lines) # Treat all as preamble if no markers
        return preamble_logs_str, step_log_segments

    # Preamble: logs before the first identified MODEL INPUT marker
    first_marker_index = model_input_markers[0]["index"]
    if first_marker_index > 0:
        preamble_logs_str = "".join(all_log_lines[0:first_marker_index])

    for i, marker in enumerate(model_input_markers):
        current_step_num = marker["step"]
        start_index = marker["index"]
        
        end_index = len(all_log_lines)
        if i + 1 < len(model_input_markers):
            end_index = model_input_markers[i+1]["index"]
        
        step_log_segments[current_step_num] = "".join(all_log_lines[start_index:end_index])
        
    return preamble_logs_str, step_log_segments


def load_full_trajectory_data(file_path: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """Loads .traj data and corresponding segmented .log data."""
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            trajectory_steps = json.load(file)["trajectory"]
    except FileNotFoundError:
        st.error(f"Trajectory file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in trajectory file: {file_path}")
        return None
    except KeyError:
        st.error(f"Missing 'trajectory' key in file: {file_path}")
        return None

    found_log_path = get_log_file_path(file_path) # Use the new helper

    if found_log_path:
        # It's good to inform the user which log file is being used for segmentation here
        st.info(f"Using log file for segmentation: {found_log_path.name}")
        preamble_logs, step_log_map = load_and_segment_raw_logs(found_log_path)
    else:
        st.warning(f"No suitable log file found for trajectory '{Path(file_path).name}'. Raw logs segments will be unavailable.")
        preamble_logs, step_log_map = "", {}

    augmented_steps = []
    for i, step_data in enumerate(trajectory_steps):
        # The 'i' here is the index in the trajectory list, which should correspond to the step number.
        step_data['raw_log_segment'] = step_log_map.get(i, "Raw logs not found for this step or step numbering mismatch.")
        augmented_steps.append(step_data)
    
    return preamble_logs, augmented_steps


# --- Evaluation Report Processing (largely unchanged) ---
def append_exit(content: Dict[str, Any]) -> Dict[str, Any]:
    last_entry = content["history"][-1]
    if last_entry["role"] == "system": return content
    exit_status = content.get("info", {}).get("exit_status")
    if not exit_status: return content
    if exit_status.startswith("submitted"):
        if "submission" in content["info"]:
            content["history"].append({"role": "model_patch", "content": content["info"]["submission"]})
        else:
            st.warning("Submission referenced in exit_status but not found in info.")
    return content

def format_metric_value(value: Optional[Union[int, float]]) -> str:
    if value is None: return "N/A"
    if isinstance(value, float): return f"{value:.2f}"
    return f"{value:,}"

def append_results(
    traj_path_obj: Path, # Changed to Path object
    instance_id: str,
    content: Dict[str, Any],
    results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stats: List[str] = []
    model_stats = {}
    exit_status = None

    if traj_path_obj.exists(): # Use the passed Path object
        data = json.loads(traj_path_obj.read_text(encoding='utf-8'))
        info = data.get("info", {})
        exit_status = info.get("exit_status")
        model_stats = info.get("model_stats", {})

    instance_cost = format_metric_value(model_stats.get("total_cost"))
    tokens_sent = format_metric_value(model_stats.get("tokens_sent"))
    tokens_received = format_metric_value(model_stats.get("tokens_received"))
    api_calls = format_metric_value(model_stats.get("api_calls"))

    stats.extend([
        "*" * 39, "Run Stats", "*" * 39,
        f"Instance Cost: ${instance_cost}",
        f"Tokens Sent: {tokens_sent}",
        f"Tokens Received: {tokens_received}",
        f"API Calls: {api_calls}",
        f"Exit Status: {exit_status}",
    ])

    status = process_results(results)
    eval_report = {"role": "Evaluation Report", "content": "\n".join([*stats, *status])}
    
    # Ensure history exists and is a list
    if "history" not in content or not isinstance(content["history"], list):
        content["history"] = []

    content["history"].insert(0, eval_report)
    # Avoid appending twice if it's already there or logic is different
    # content["history"].append(eval_report) # Original code had this, might be specific
    return content


def process_results(results: Optional[Dict[str, Any]]) -> List[str]:
    if not results: return ["No scores found"]
    agent_results = results.get("agent")
    baseline_results = results.get("baseline")
    if not agent_results and not baseline_results: return ["Baseline and Agent scores not found"]
    
    status = []
    if baseline_results and agent_results:
        status.extend(["*" * 39, "Agent vs Baseline Scores", "*" * 39])
        formatted_scores = defaultdict(dict)
        for score_type, score in baseline_results.items(): formatted_scores[score_type]["Baseline"] = score
        for i, agent_score in enumerate(agent_results): # Assuming agent_results is a list
            for score_type, score in agent_score.items(): formatted_scores[score_type][f"Attempt {i+1}"] = score
        for score_type, scores in formatted_scores.items():
            status.append(f"Metric: {score_type}")
            status.extend(f"  {model}: {score:.3f}" for model, score in scores.items())
    elif baseline_results:
        status.append("**** Baseline Scores ****")
        status.extend(f"  {score_type}: {score}" for score_type, score in baseline_results.items())
    elif agent_results:
        status.append("**** Agent Scores ****")
        if isinstance(agent_results, list):
            for i, agent_res in enumerate(agent_results):
                status.append(f"Attempt {i+1}:")
                status.extend(f"  {score_type}: {score}" for score_type, score in agent_res.items())
        elif isinstance(agent_results, dict): # If it's a single attempt
             status.extend(f"  {score_type}: {score}" for score_type, score in agent_results.items())
        else:
            status.append(str(agent_results))
    return status

def load_results(results_path: Path) -> Optional[Dict[str, Any]]:
    if not results_path.exists(): return None
    with open(results_path, encoding='utf-8') as infile: return json.load(infile)

def load_evaluation_content(file_name: str) -> Dict[str, Any]: # Renamed for clarity
    """Loads .traj content specifically for the evaluation report."""
    filepath = Path(file_name)
    try:
        with open(filepath, encoding='utf-8') as infile:
            content = json.load(infile)
    except FileNotFoundError:
        st.error(f"Trajectory file for eval content not found: {file_name}")
        return {"history": [{"role": "Evaluation Report", "content": "Error: Trajectory file not found."}]}
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in trajectory file for eval content: {file_name}")
        return {"history": [{"role": "Evaluation Report", "content": "Error: Invalid JSON in trajectory file."}]}


    results_file = filepath.parent / "results.json"
    results = load_results(results_file)

    content = append_exit(content) # Modifies content in place
    return append_results(filepath, filepath.stem, content, results)


# --- UI Display Functions ---

def display_content_summary(content: str) -> None:
    if not content: return
    # Using st.expander for better native feel
    with st.expander("📝 Evaluation Report", expanded=False):
        st.text(content) # Using st.text for pre-formatted like display


def display_step(step_data: Dict[str, Any], step_num: int, total_steps: int) -> None:
    st.markdown(f'<div class="step-indicator">Step {step_num + 1} / {total_steps}</div>', unsafe_allow_html=True)

    # Navigation and Progress (only if in step-by-step view)
    # This part was tied to session_state.view_mode == "step", assuming it's always step-by-step for now
    # If you re-introduce full vs step view, conditionalize this block
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Previous Step", key=f"prev_step_{step_num}", disabled=(step_num == 0), use_container_width=True):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        if st.button("Next Step ➡️", key=f"next_step_{step_num}", disabled=(step_num == total_steps - 1), use_container_width=True):
            st.session_state.current_step += 1
            st.rerun()
    st.progress((step_num + 1) / total_steps)
    
    # Caption might not be available in step_data directly, adjust if needed
    # st.markdown(f"""<div class="step-caption">{step_data.get('caption', '')}</div>""", unsafe_allow_html=True)

    display_step_components(step_data)

def display_step_components(step_data: Dict[str, Any]) -> None:
    # --- Thought Process ---
    thought = step_data.get("thought", "")  # Default to empty string
    thought_cleaned = thought.replace("DISCUSSION", "").strip()

    # We'll build the HTML string for the box and inject the content.
    # To render markdown within the thought_cleaned string, we can't just embed it.
    # So, we use the 3-part st.markdown approach which is usually reliable.

    st.markdown(
        '<div class="content-box thought-box">'
        '<div class="box-header thought-header">💭 Thought Process</div>'
        '<div class="box-content">',  # This div is for the actual thought content
        unsafe_allow_html=True
    )

    if thought_cleaned:
        # This is the crucial part: render the thought_cleaned as markdown here.
        # It should render into the currently open HTML context.
        st.markdown(thought_cleaned, unsafe_allow_html=True)
    else:
        # If there's no thought, display a placeholder.
        st.markdown("<em>No thought recorded.</em>", unsafe_allow_html=True)

    st.markdown(
        '</div></div>',  # Closes .box-content and then .content-box
        unsafe_allow_html=True
    )

    # --- Action Taken ---
    action = step_data.get("action", "No action recorded.")
    # For Action, Observation, and Raw Logs, we directly embed the content
    # as pre-formatted text because they are not typically rich markdown.
    st.markdown(
        f'<div class="content-box action-box">'
        f'<div class="box-header action-header">🤖 Action Taken</div>'
        f'<div class="box-content"><pre><code class="language-bash">{action.strip()}</code></pre></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Execution Result ---
    observation = step_data.get("observation", "No observation recorded.")
    st.markdown(
        f'<div class="content-box result-box">'
        f'<div class="box-header result-header">💻 Execution Result</div>'
        f'<div class="box-content"><pre><code>{observation.strip()}</code></pre></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Raw Logs Section ---
    raw_log_segment = step_data.get("raw_log_segment", "Raw log segment not available.")
    with st.expander("📄 Raw Logs for this Step", expanded=False):
        # Using st.code is better for log/code blocks as it handles formatting.
        st.code(raw_log_segment, language='text', line_numbers=False) # 'log' or 'text'


# --- Main Application Logic ---

def initialize_session_state(args: argparse.Namespace) -> None:
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'current_trajectory_path' not in st.session_state: # Renamed for clarity
        st.session_state.current_trajectory_path = None
    if 'all_trajectories' not in st.session_state:
        st.session_state.all_trajectories = find_trajectory_files(args.trajectory_dir)
    
    # Initialize from query params
    query_params = st.query_params
    if "trajectory_path" in query_params and not st.session_state.current_trajectory_path:
        # Validate if this path is in our list of known trajectories
        path_from_query = query_params["trajectory_path"]
        if any(t['filepath'] == path_from_query for t in st.session_state.all_trajectories):
            st.session_state.current_trajectory_path = path_from_query
            st.session_state.current_step = 0 # Reset step
        else:
            st.warning("Trajectory path from URL not found or invalid.")
            st.query_params.clear() # Clear invalid param

def display_welcome_message() -> None:
    st.markdown(
        """
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1>👋 Welcome to the MLGym Trajectory Visualizer</h1>
            <p style='font-size: 1.2rem; color: var(--slate-200); margin: 2rem 0;'>
                Select a task from the sidebar to view the MLGym Agent's trajectory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def setup_sidebar(args: argparse.Namespace) -> None:
    with st.sidebar:
        st.markdown("# 👩‍🔬 MLGym Visualizer")
        st.markdown(f"**Directory:** `{args.trajectory_dir}`")

        all_trajectories = st.session_state.all_trajectories
        if not all_trajectories:
            st.warning("No trajectory files found.")
            return

        # --- Search and Filters ---
        search_term = st.text_input("Search Trajectories", placeholder="Search by name, model, task...")

        model_names = sorted(list(set(t["model_name"] for t in all_trajectories)))
        task_names = sorted(list(set(t["task_name"] for t in all_trajectories)))

        selected_models = st.multiselect("Filter by Model", model_names)
        selected_tasks = st.multiselect("Filter by Task", task_names)

        # --- Apply Filters and Search ---
        filtered_trajectories = all_trajectories
        if search_term:
            search_lower = search_term.lower()
            filtered_trajectories = [
                t for t in filtered_trajectories 
                if search_lower in t["display_name"].lower() or \
                   search_lower in t["filepath"].lower()
            ]
        if selected_models:
            filtered_trajectories = [t for t in filtered_trajectories if t["model_name"] in selected_models]
        if selected_tasks:
            filtered_trajectories = [t for t in filtered_trajectories if t["task_name"] in selected_tasks]

        # --- Display Trajectories ---
        st.markdown("### Select Trajectory")
        if not filtered_trajectories:
            st.info("No trajectories match your criteria.")
        
        # Group by task for better organization
        trajectories_by_task = defaultdict(list)
        for t in filtered_trajectories:
            trajectories_by_task[t['task_name']].append(t)

        for task_name, trajectories_in_task in trajectories_by_task.items():
            with st.expander(f"Task: {task_name} ({len(trajectories_in_task)})"):
                for trajectory_data in trajectories_in_task:
                    # Use a more specific key for the button
                    button_key = f"btn_{trajectory_data['filepath']}"
                    if st.button(trajectory_data["display_name"], key=button_key, use_container_width=True):
                        st.session_state.current_trajectory_path = trajectory_data["filepath"]
                        st.session_state.current_step = 0
                        st.query_params["trajectory_path"] = trajectory_data["filepath"] # Update URL
                        st.rerun() # Rerun to reflect selection and URL change immediately


def display_trajectory_content() -> None:
    if not st.session_state.current_trajectory_path:
        display_welcome_message()
        return

    st.title("👩‍🔬 Trajectory Details")
    
    # --- Log and SLURM File Download Buttons ---
    current_traj_path_obj = Path(st.session_state.current_trajectory_path)
    log_dir = current_traj_path_obj.parent

    # Main Log File
    main_log_file_to_download = get_log_file_path(st.session_state.current_trajectory_path) # Uses existing helper

    # SLURM Files - We'll search for them directly here
    slurm_out_files = sorted(list(log_dir.glob("slurm_*.out")), reverse=True) # Get newest first if multiple
    slurm_err_files = sorted(list(log_dir.glob("slurm_*.err")), reverse=True) # Get newest first if multiple

    # Create columns for the buttons
    num_buttons = 0
    if main_log_file_to_download and main_log_file_to_download.exists():
        num_buttons += 1
    if slurm_out_files:
        num_buttons +=1
    if slurm_err_files:
        num_buttons +=1
    
    if num_buttons > 0:
        cols = st.columns(num_buttons)
        col_idx = 0

        if main_log_file_to_download and main_log_file_to_download.exists():
            with cols[col_idx]:
                try:
                    with open(main_log_file_to_download, "rb") as fp:
                        st.download_button(
                            label="📥 Download App Log", # Renamed for clarity
                            data=fp,
                            file_name=main_log_file_to_download.name,
                            mime="text/plain",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error (App Log): {e}")
            col_idx +=1
        elif main_log_file_to_download: # File was identified but doesn't exist
             with cols[col_idx]: # Still take up a column to show the warning
                st.warning(f"App log '{main_log_file_to_download.name}' not found.")
             col_idx +=1


        if slurm_out_files:
            slurm_out_file_to_download = slurm_out_files[0] # Pick the first one (newest due to sort)
            with cols[col_idx]:
                if slurm_out_file_to_download.exists():
                    try:
                        with open(slurm_out_file_to_download, "rb") as fp:
                            st.download_button(
                                label="📄 Download Slurm Out",
                                data=fp,
                                file_name=slurm_out_file_to_download.name,
                                mime="text/plain",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error (Slurm Out): {e}")
                else: # Should not happen if glob found it, but good check
                    st.warning(f"Slurm out file '{slurm_out_file_to_download.name}' not found.")
            col_idx+=1
        
        if slurm_err_files:
            slurm_err_file_to_download = slurm_err_files[0] # Pick the first one (newest due to sort)
            with cols[col_idx]:
                if slurm_err_file_to_download.exists():
                    try:
                        with open(slurm_err_file_to_download, "rb") as fp:
                            st.download_button(
                                label="⚠️ Download Slurm Err",
                                data=fp,
                                file_name=slurm_err_file_to_download.name,
                                mime="text/plain",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error (Slurm Err): {e}")
                else: # Should not happen if glob found it
                    st.warning(f"Slurm err file '{slurm_err_file_to_download.name}' not found.")
            col_idx+=1
        
        st.markdown("---") # Add a separator after the buttons
    else: # No log files of any kind found to offer for download
        st.info("No application log or SLURM log files found for download in the trajectory directory.")
        st.markdown("---")
    
    # Display Evaluation Report Summary
    eval_content_data = load_evaluation_content(st.session_state.current_trajectory_path)
    if eval_content_data and "history" in eval_content_data and eval_content_data["history"]:
         # Assuming the first item is the main report string after processing
        report_str = eval_content_data["history"][0].get("content", "Evaluation report not available.")
        display_content_summary(report_str)
    else:
        st.info("Evaluation report data could not be loaded or is empty.")

    # Load and display trajectory steps with raw logs
    trajectory_data = load_full_trajectory_data(st.session_state.current_trajectory_path)
    if not trajectory_data:
        return # Error already shown by load_full_trajectory_data

    preamble_logs, augmented_steps = trajectory_data
    
    if preamble_logs:
        with st.expander("📜 Log Preamble (System Prompts, Initial Setup)", expanded=False):
            st.text(preamble_logs)
    
    if not augmented_steps:
        st.info("No steps found in this trajectory.")
        return

    # View mode selection (optional, if you want full vs. step-by-step)
    # For now, always step-by-step
    current_step_idx = st.session_state.current_step
    if 0 <= current_step_idx < len(augmented_steps):
        display_step(augmented_steps[current_step_idx], current_step_idx, len(augmented_steps))
    else:
        st.warning(f"Current step index {current_step_idx} is out of bounds for {len(augmented_steps)} steps. Resetting.")
        st.session_state.current_step = 0
        st.rerun()


def main() -> None:
    args = parse_args()
    configure_page_style()
    initialize_session_state(args) # Pass args for initial loading
    setup_sidebar(args)
    display_trajectory_content()

if __name__ == "__main__":
    main()
