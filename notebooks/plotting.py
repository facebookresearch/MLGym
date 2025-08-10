# %%
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlgym.evaluation.utils import (  # EXIT_STATUS_MAP,; MODEL_LOGOS,
    ACTION_COLOR_MAP,
    ACTION_COLOR_MAP_DEEP,
    MODEL_NAME_MAP,
    MODEL_SHORT_NAME_MAP,
    EXIT_STATUS_COLOR_MAP,
    MODELS,
    TASKS,
    get_action_results,
    get_exit_status_results,
    process_trajectories,
)

print(ACTION_COLOR_MAP_DEEP)
# set_custom_font()

# %%
# Variables
traj_parent_dir = "../trajectories/mlgym_bench_v0/"
traj_pattern = "default__t-0.00__p-0.95__c-4.00__install-0__parallel_agents"
models = MODELS
output_dir = Path("../assets/figs/")


# %%
# Get all trajectory information
all_trajectories = defaultdict(dict)
acceptable_exit_statuses = ["autosubmission (max_steps)", "submitted"]

for task_id, _ in TASKS.items():
    task_results = process_trajectories(traj_parent_dir, traj_pattern, task_id, models)
    all_trajectories[task_id] = task_results

exit_status_results = get_exit_status_results(all_trajectories)
action_results = get_action_results(all_trajectories)


# %%
def plot_es_counts_per_model(exit_status_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart of exit status counts with flipped axes using seaborn.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # sort the columns by the name of the model
    df = df[df.sum().sort_values(ascending=False).index]

    # Remove the "Success" exit status column if it exists.
    if "Success" in df.columns:
        df.drop("Success", axis=1, inplace=True)
    if "Max Steps" in df.columns:
        df.drop("Max Steps", axis=1, inplace=True)
    if "API" in df.columns:
        df.drop("API", axis=1, inplace=True)

    # Transpose so that exit statuses are on x-axis and models are columns.
    df_flip: pd.DataFrame = df.T

    # Convert to long format for seaborn
    df_melted = df_flip.reset_index().melt(
        id_vars="index", var_name="Model", value_name="Count"
    )
    df_melted.rename(columns={"index": "Exit Status"}, inplace=True)

    # Map model names to short names
    df_melted["Model_Short"] = df_melted["Model"].map(
        lambda x: MODEL_SHORT_NAME_MAP.get(x, x)
    )

    # Order exit statuses by total counts (descending)
    exit_status_totals = (
        df_melted.groupby("Exit Status")["Count"].sum().sort_values(ascending=False)
    )
    df_melted["Exit Status"] = pd.Categorical(
        df_melted["Exit Status"], categories=exit_status_totals.index, ordered=True
    )

    # Create pivot table for manual stacking
    pivot_df = df_melted.pivot(
        index="Exit Status", columns="Model_Short", values="Count"
    ).fillna(0)
    pivot_df = pivot_df.reindex(exit_status_totals.index)  # Maintain order

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Get seaborn colors
    colors = sns.color_palette("deep", n_colors=len(pivot_df.columns))

    # Manual stacking with seaborn styling
    bottom_values = np.zeros(len(pivot_df.index))
    legend_handles = []

    for i, model in enumerate(pivot_df.columns):
        bars = ax.bar(
            range(len(pivot_df.index)),
            pivot_df[model],
            bottom=bottom_values,
            color=colors[i],
            alpha=1.0,
            label=model,
            width=0.6,
            edgecolor="black",
            linewidth=0.5,
        )
        legend_handles.append(bars[0])
        bottom_values += pivot_df[model]

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Set labels and ticks
    ax.set_xlabel("Exit Status", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, rotation=0, ha="center")

    # Style the legend with thinner frame
    legend = ax.legend(
        legend_handles,
        pivot_df.columns,
        loc="upper right",
        fontsize=8,
        frameon=True,
        title=None,
    )

    # Explicitly override global patch settings to make frame visible
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)  # Thinner frame
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Add grid
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_es_counts_per_model(exit_status_results, "es_counts_per_model.pdf")


# %%


def plot_es_counts_per_model_horizontal_bar(exit_status_results: dict, output_path: str) -> None:
    """
    Plot a modern horizontal stacked bar chart showing exit status distribution by model.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # Remove unwanted exit status columns
    columns_to_remove = ["Success", "Max Steps", "API"]
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Ensure we have all models from MODELS list and sort by total counts
    df = df.reindex(MODELS, fill_value=0)
    df = df.reindex(df.sum(axis=1).sort_values(ascending=True).index)  # Sort by total, ascending for better visual flow

    # Use seaborn deep color palette (same as vertical bar chart)
    exit_statuses = df.columns.tolist()
    colors = sns.color_palette("deep", n_colors=len(exit_statuses))

    # Create compact figure matching es_counts_per_model aesthetics
    fig, ax = plt.subplots(figsize=(6.5, 4))  # Match exact size of vertical chart

    # Create horizontal stacked bar chart matching vertical chart aesthetic
    y_pos = np.arange(len(df.index))
    left = np.zeros(len(df.index))

    legend_handles = []
    for i, status in enumerate(exit_statuses):
        values = df[status].values
        bars_segment = ax.barh(y_pos, values, left=left,
                              color=colors[i],
                              alpha=1.0,
                              label=status,
                              height=0.6,  # Match width=0.6 from vertical chart
                              edgecolor="black",  # Match vertical chart
                              linewidth=0.5)  # Match vertical chart
        legend_handles.append(bars_segment[0])
        left += values

    # Apply styling to match es_counts_per_model aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_NAME_MAP.get(model, model) for model in df.index],
                       fontsize=10)
    ax.set_xlabel('Count', fontsize=10)

    # Style the plot - keep all spines for box appearance (matching es_counts_per_model)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks (matching es_counts_per_model style)
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Add grid (matching es_counts_per_model)
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)  # Horizontal grid for horizontal bars
    ax.set_axisbelow(True)

    # Set x-axis limits and ticks to create gap at the end with 10-unit intervals
    max_total = df.sum(axis=1).max()
    ax.set_xlim(0, max_total + 10)  # Add gap of 10 units after the longest bar

    # Set x-axis ticks at intervals of 10
    tick_max = int(max_total + 10)
    xticks = list(range(0, tick_max + 1, 10))
    ax.set_xticks(xticks)

    # Create legend with frame (matching es_counts_per_model style)
    legend = ax.legend(legend_handles, exit_statuses, loc='lower right',
                      frameon=True, fontsize=8, title=None)

    # Style the legend frame (exactly like es_counts_per_model)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)  # Thinner frame
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_es_counts_per_model_horizontal_bar(exit_status_results, "es_counts_per_model_horizontal_bar.pdf")


# %%


def plot_es_counts_per_model_heatmap(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a heatmap of exit status counts per model with counts displayed in each cell.

    Args:
        exit_status_results (dict): Dictionary containing exit status counts per model.
            Expected key 'es_counts_per_model' with structure:
            {model: {exit_status: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    raw_counts: dict = exit_status_results.get("es_counts_per_model", {})
    data: dict = {model: dict(counts) for model, counts in raw_counts.items()}
    df: pd.DataFrame = pd.DataFrame.from_dict(data, orient="index").fillna(0)

    # sort the columns by the name of the model
    df = df[df.sum().sort_values(ascending=False).index]

    # Remove the "Success" exit status column if it exists.
    if "Success" in df.columns:
        df.drop("Success", axis=1, inplace=True)
    if "Max Steps" in df.columns:
        df.drop("Max Steps", axis=1, inplace=True)
    if "API" in df.columns:
        df.drop("API", axis=1, inplace=True)

    df = df.reindex(MODELS)

    # Map model names to short names for y-axis
    df.index = [MODEL_NAME_MAP.get(col, col) for col in df.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create heatmap with darker colors
    sns.heatmap(
        df,  # Transpose so models are on y-axis, exit statuses on x-axis
        annot=True,
        fmt="g",  # Format numbers as integers
        cmap=sns.light_palette(
            "navy", as_cmap=True
        ),  # Use continuous sequential colormap for count data
        robust=True,
        # square=True,
        linewidths=0.1,
        linecolor="lightgray",
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    # ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    # ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Set labels
    # ax.set_xlabel("Exit Status", fontsize=10)
    # ax.set_ylabel("Model", fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=0)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Count", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_es_counts_per_model_heatmap(exit_status_results, "es_counts_per_model_heatmap.pdf")

# %%


def plot_failed_incomplete_runs_per_model(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a bar chart of the failed and incomplete runs for each model.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.
    Uses seaborn barplot with consistent color palette.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_model', 'incomplete_runs_per_model'
        output_path (str): Path to save the plotted figure in PDF format.
    """

    failed_runs = exit_status_results["failed_runs_per_model"]
    incomplete_runs = exit_status_results["incomplete_runs_per_model"]

    # Create DataFrame with model order
    df = pd.DataFrame(
        {
            "Failed Runs": [failed_runs[m] for m in MODELS],
            "Incomplete Runs": [incomplete_runs[m] for m in MODELS],
        },
        index=MODELS,
    )

    # Sort by total failed runs while preserving model order
    totals = df["Failed Runs"]
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)

    # Map model names to short names
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Convert to long format for seaborn
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="Status", value_name="Count"
    )
    df_melted.rename(columns={"index": "Model"}, inplace=True)

    # Print melted data for debugging
    print("Melted data sample:")
    print(df_melted.head(10))
    print(f"Max count: {df_melted['Count'].max()}")
    print(
        f"Incomplete run counts: {df_melted[df_melted['Status'] == 'Incomplete Runs']['Count'].tolist()}"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn barplot with consistent color palette
    colors = sns.color_palette("deep", 2)
    sns.barplot(
        data=df_melted, x="Model", y="Count", hue="Status", palette=colors, ax=ax
    )

    # Style the plot - fix x-axis labels (fontsize=8, not bold)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=8, fontweight="normal"
    )
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlabel("")

    # Fix y-axis scaling - use max value with minimal padding
    max_val = df_melted["Count"].max()
    y_max = max_val + max(2, int(max_val * 0.05))  # Add minimal padding
    ax.set_ylim(0, y_max)

    # Set y-axis ticks with appropriate steps
    if y_max <= 20:
        step = 5
    elif y_max <= 50:
        step = 10
    else:
        step = 15
    yticks = list(range(0, y_max + 1, step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(str, yticks)), fontsize=8)

    # Style the legend
    legend = ax.legend(loc="upper right", fontsize=8, title=None)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)

    # Add grid
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Style spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_failed_incomplete_runs_per_model(
    exit_status_results, "failed_incomplete_runs_per_model.pdf"
)


# %%


def plot_failed_incomplete_runs_per_model_horizontal(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a horizontal stacked bar chart showing failed and incomplete runs by model.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.
    Matches aesthetic of es_counts_per_model_horizontal_bar.pdf with seaborn deep palette.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_model', 'incomplete_runs_per_model'
        output_path (str): Path to save the plotted figure in PDF format.
    """

    failed_runs = exit_status_results["failed_runs_per_model"]
    incomplete_runs = exit_status_results["incomplete_runs_per_model"]

    # Create DataFrame with model order
    df = pd.DataFrame(
        {
            "Failed Runs": [failed_runs[m] for m in MODELS],
            "Incomplete Runs": [incomplete_runs[m] for m in MODELS],
        },
        index=MODELS,
    )

    # Ensure we have all models from MODELS list and sort by total counts
    df = df.reindex(MODELS, fill_value=0)
    df = df.reindex(df.sum(axis=1).sort_values(ascending=True).index)  # Sort by total, ascending for better visual flow

    # Use seaborn deep color palette (same as horizontal bar chart)
    statuses = df.columns.tolist()
    colors = sns.color_palette("deep", n_colors=len(statuses))

    # Create compact figure matching es_counts_per_model aesthetics
    fig, ax = plt.subplots(figsize=(6.5, 4))  # Match exact size of horizontal chart

    # Create horizontal stacked bar chart matching horizontal chart aesthetic
    y_pos = np.arange(len(df.index))
    left = np.zeros(len(df.index))

    legend_handles = []
    for i, status in enumerate(statuses):
        values = df[status].values
        bars_segment = ax.barh(y_pos, values, left=left,
                              color=colors[i],
                              alpha=1.0,
                              label=status,
                              height=0.6,  # Match width=0.6 from horizontal chart
                              edgecolor="black",  # Match horizontal chart
                              linewidth=0.5)  # Match horizontal chart
        legend_handles.append(bars_segment[0])
        left += values

    # Apply styling to match es_counts_per_model aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_NAME_MAP.get(model, model) for model in df.index],
                       fontsize=10)
    ax.set_xlabel('Count', fontsize=10)

    # Style the plot - keep all spines for box appearance (matching es_counts_per_model)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks (matching es_counts_per_model style)
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Add grid (matching es_counts_per_model)
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)  # Horizontal grid for horizontal bars
    ax.set_axisbelow(True)

    # Set x-axis limits and ticks to create gap at the end with appropriate intervals
    max_total = df.sum(axis=1).max()
    ax.set_xlim(0, max_total + 5)  # Add gap of 5 units after the longest bar

    # Set x-axis ticks at appropriate intervals
    if max_total <= 20:
        step = 5
    elif max_total <= 50:
        step = 10
    else:
        step = 15

    tick_max = int(max_total + 5)
    xticks = list(range(0, tick_max + 1, step))
    ax.set_xticks(xticks)

    # Create legend with frame (matching es_counts_per_model style)
    legend = ax.legend(legend_handles, statuses, loc='lower right',
                      frameon=True, fontsize=8, title=None)

    # Style the legend frame (exactly like es_counts_per_model)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)  # Thinner frame
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_failed_incomplete_runs_per_model_horizontal(
    exit_status_results, "failed_incomplete_runs_per_model_horizontal.pdf"
)

# %%


def plot_failed_incomplete_runs_per_task(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a bar chart of the failed and incomplete runs for each task.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_task', 'incomplete_runs_per_task'
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    failed_runs = exit_status_results["failed_runs_per_task"]
    incomplete_runs = exit_status_results["incomplete_runs_per_task"]

    # Create DataFrame with task names from TASKS dictionary
    df = pd.DataFrame(
        {
            "Failed": [failed_runs[t] for t in TASKS],
            "Incomplete": [incomplete_runs[t] for t in TASKS],
        },
        index=[TASKS[t]["shortname"] for t in TASKS],
    )

    df.drop(index="PD", inplace=True)
    df.drop(index="F-MNIST", inplace=True)

    # Sort by total while preserving task names
    totals = df["Failed"]
    df = df.reindex(totals.sort_values(ascending=False).index)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df.index))
    width = 0.35

    # Using first two colors from MODEL_COLOR_MAP
    failed_color = "#FD5901"
    incomplete_color = "#F78104"

    # Solid bars for failed runs
    ax.bar(x - width / 2, df["Failed"], width, label="Failed Runs", color=failed_color)

    # Hollow hatched bars for incomplete runs
    ax.bar(
        x + width / 2,
        df["Incomplete"],
        width,
        label="Incomplete Runs",
        edgecolor=incomplete_color,
        facecolor="none",
        hatch="////",
        linewidth=2,
    )

    ax.set_xticks(x)
    # yticks = list(range(0, 25, 5))
    # ax.set_yticks(yticks, yticks, fontsize=12, fontweight='bold')
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    plt.ylabel("Count", fontsize=10)

    # Create legend handles with correct colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=failed_color, label="Failed Runs"),
        Patch(
            facecolor="none",
            edgecolor=incomplete_color,
            hatch="///",
            label="Incomplete Runs",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # Add horizontal grid lines
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

plot_failed_incomplete_runs_per_task(
    exit_status_results, "failed_incomplete_runs_per_task.pdf"
)


# %%


def plot_failed_incomplete_runs_per_task_horizontal(
    exit_status_results: dict, output_path: str
) -> None:
    """
    Plot a horizontal stacked bar chart showing failed and incomplete runs by task.
    Failed runs have no agent scores, incomplete runs have scores but failed to submit.
    Matches exact aesthetic of plot_failed_incomplete_runs_per_model_horizontal with seaborn deep palette.

    Args:
        exit_status_results (dict): Dictionary containing failed and incomplete run counts.
            Expected keys: 'failed_runs_per_task', 'incomplete_runs_per_task'
        output_path (str): Path to save the plotted figure in PDF format.
    """

    failed_runs = exit_status_results["failed_runs_per_task"]
    incomplete_runs = exit_status_results["incomplete_runs_per_task"]

    # Create DataFrame with task names from TASKS dictionary
    df = pd.DataFrame(
        {
            "Failed Runs": [failed_runs[t] for t in TASKS],
            "Incomplete Runs": [incomplete_runs[t] for t in TASKS],
        },
        index=[TASKS[t]["name"] for t in TASKS],
    )

    # Drop specific tasks as in original function
    # df.drop(index="PD", inplace=True)
    # df.drop(index="F-MNIST", inplace=True)

    # Sort by total count (ascending for better visual flow)
    df = df.reindex(df.sum(axis=1).sort_values(ascending=True).index)

    # Use seaborn deep color palette (same as horizontal model chart)
    statuses = df.columns.tolist()
    colors = sns.color_palette("deep", n_colors=len(statuses))

    # Create compact figure matching es_counts_per_model aesthetics
    fig, ax = plt.subplots(figsize=(6.5, 4))  # Match exact size of horizontal chart

    # Create horizontal stacked bar chart matching horizontal chart aesthetic
    y_pos = np.arange(len(df.index))
    left = np.zeros(len(df.index))

    legend_handles = []
    for i, status in enumerate(statuses):
        values = df[status].values
        bars_segment = ax.barh(y_pos, values, left=left,
                              color=colors[i],
                              alpha=1.0,
                              label=status,
                              height=0.6,  # Match width=0.6 from horizontal chart
                              edgecolor="black",  # Match horizontal chart
                              linewidth=0.5)  # Match horizontal chart
        legend_handles.append(bars_segment[0])
        left += values

    # Apply styling to match es_counts_per_model aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=10)  # Task shortnames already correct
    ax.set_xlabel('Count', fontsize=10)

    # Style the plot - keep all spines for box appearance (matching es_counts_per_model)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks (matching es_counts_per_model style)
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Add grid (matching es_counts_per_model)
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)  # Horizontal grid for horizontal bars
    ax.set_axisbelow(True)

    # Set x-axis limits and ticks to create gap at the end with appropriate intervals
    max_total = df.sum(axis=1).max()
    ax.set_xlim(0, max_total + 5)  # Add gap of 5 units after the longest bar

    # Set x-axis ticks at appropriate intervals
    if max_total <= 20:
        step = 5
    elif max_total <= 50:
        step = 10
    else:
        step = 15

    tick_max = int(max_total + 5)
    xticks = list(range(0, tick_max + 1, step))
    ax.set_xticks(xticks)

    # Create legend with frame (matching es_counts_per_model style)
    legend = ax.legend(legend_handles, statuses, loc='lower right',
                      frameon=True, fontsize=8, title=None)

    # Style the legend frame (exactly like es_counts_per_model)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)  # Thinner frame
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_failed_incomplete_runs_per_task_horizontal(
    exit_status_results, "failed_incomplete_runs_per_task_horizontal.pdf"
)

# %%


def plot_total_actions(action_results: dict, output_path: str) -> None:
    """
    Plot a bar chart of the number of times each action type is taken across all tasks and models
    Args:
        action_results (dict): Dictionary containing action counts.
            Expected key 'action_counts' with structure:
            {action_type: count, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    # Get the total counts for each action type
    action_counts = action_results["action_counts"]

    # Create DataFrame
    df = pd.DataFrame({"Count": action_counts.values()}, index=action_counts.keys())

    sort_order = df["Count"].sort_values(ascending=False).index
    df = df.reindex(sort_order)
    colors = [ACTION_COLOR_MAP_DEEP[action] for action in sort_order]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(df.index))

    bars = ax.bar(x, df["Count"], color=colors, width=0.6,
                  edgecolor="black", linewidth=0.5, alpha=1.0)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=8)

    ax.set_ylim(0, max(df["Count"]) * 1.1)
    yticks = list(range(0, max(df["Count"]) + 900, 1000))
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(str, yticks)), fontsize=8)
    ax.set_ylabel("Count", fontsize=10)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

plot_total_actions(action_results, "total_actions_analysis.pdf")


# %%


def plot_total_actions_donut(action_results: dict, output_path: str) -> None:
    """
    Plot a donut chart of the number of times each action type is taken across all tasks and models.
    Matches the aesthetic of donut_example.png with external labels and connecting lines.

    Args:
        action_results (dict): Dictionary containing action counts.
            Expected key 'action_counts' with structure:
            {action_type: count, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    action_counts = action_results["action_counts"]

    # Prepare data
    labels = list(action_counts.keys())
    sizes = list(action_counts.values())

    # Sort by size (descending)
    sorted_pairs = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
    labels, sizes = zip(*sorted_pairs)

    # Calculate percentages
    total = sum(sizes)
    percentages = [size/total * 100 for size in sizes]

    # Use consistent action color map
    colors = [ACTION_COLOR_MAP_DEEP[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5), subplot_kw=dict(aspect="equal"))

    # Create the pie chart wedges with a hole in the center (donut chart)
    wedges, texts = ax.pie(
        sizes, wedgeprops=dict(width=0.5, edgecolor="w", linewidth=3),
        startangle=-40, colors=colors, radius=1.0
    )

    # Common properties for the labels and arrows
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0, alpha=0)
    arrow_props = dict(arrowstyle="-", color="gray", lw=1.2)
    kw = dict(bbox=bbox_props, zorder=0, va="center", fontsize=9)

    # Calculate all label positions first to detect overlaps
    label_positions = []
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y_start = np.sin(np.deg2rad(ang))
        x_start = np.cos(np.deg2rad(ang))

        # Initial position
        x_end = 1.4 * np.sign(x_start)
        y_end = 1.4 * y_start

        label_positions.append((x_start, y_start, x_end, y_end, ang, i))

    # Sort by side (left/right) and then by y-position to handle overlaps
    left_labels = [pos for pos in label_positions if pos[0] <= 0]
    right_labels = [pos for pos in label_positions if pos[0] > 0]

    # Sort by y-position and spread out overlapping labels
    left_labels.sort(key=lambda x: x[3])  # Sort by y_end
    right_labels.sort(key=lambda x: x[3])  # Sort by y_end

    # Adjust positions to prevent overlap
    min_spacing = 0.15

    # Adjust left side labels
    for j in range(1, len(left_labels)):
        if left_labels[j][3] - left_labels[j-1][3] < min_spacing:
            left_labels[j] = (left_labels[j][0], left_labels[j][1], left_labels[j][2],
                             left_labels[j-1][3] + min_spacing, left_labels[j][4], left_labels[j][5])

    # Adjust right side labels
    for j in range(1, len(right_labels)):
        if right_labels[j][3] - right_labels[j-1][3] < min_spacing:
            right_labels[j] = (right_labels[j][0], right_labels[j][1], right_labels[j][2],
                              right_labels[j-1][3] + min_spacing, right_labels[j][4], right_labels[j][5])

    # Combine and draw all labels
    all_positions = left_labels + right_labels

    for x_start, y_start, x_end, y_end, ang, i in all_positions:
        # Determine if the label should be on the left or right side
        horizontalalignment = "left" if x_start > 0 else "right"

        # Create the label text
        label_text = f"{labels[i]}\n{percentages[i]:.1f}%"

        # Define the connection style for the arrow line
        connectionstyle = f"angle,angleA=0,angleB={ang}"

        # Draw the annotation using the calculated positions and styles
        ax.annotate(
            label_text,
            xy=(x_start, y_start),
            xytext=(x_end, y_end),
            horizontalalignment=horizontalalignment,
            arrowprops={**arrow_props, "connectionstyle": connectionstyle},
            **kw,
        )

    # Ensure the plot is drawn as a circle
    ax.axis("equal")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_total_actions_donut(action_results, "total_actions_donut.pdf")

# %%


def plot_actions_per_step(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions at each step.

    Args:
        action_results (dict): Dictionary containing action counts per step.
            Expected key 'actions_per_step' with structure:
            {step_number: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    # Get the actions per step
    actions_per_step = action_results["actions_per_step"]

    # Create DataFrame with all steps from 0 to 50
    df = pd.DataFrame(index=range(51))  # 0 to 50 inclusive
    action_types = list(ACTION_COLOR_MAP.keys())

    # Fill in the counts for each action type at each step
    for action in action_types:
        df[action] = [
            actions_per_step.get(step, {}).get(action, 0) for step in range(51)
        ]

    # Fill NaN values with 0
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Create stacked bars
    bottom = np.zeros(51)
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP_DEEP[action],
            label=action,
            width=1.0,
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
        bottom += df[action]

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Set x-axis ticks and labels
    xticks = [1] + list(range(5, 51, 5))
    yticks = list(range(0, 700, 100))
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(str, xticks)), fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(str, yticks)), fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xlim(-0.5, 51)
    ax.set_xlabel("Step Number", fontsize=10)

    # Add grid
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Style the legend
    legend = ax.legend(loc="upper right", fontsize=8, ncols=len(action_types) // 2,
                      frameon=True, title=None)
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_actions_per_step(action_results, "actions_per_step.pdf")

# %%


def plot_actions_per_model(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions for each model.

    Args:
        action_results (dict): Dictionary containing action counts per model.
            Expected key 'actions_per_model' with structure:
            {model_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    actions_per_model = action_results["actions_per_model"]
    print(actions_per_model)
    action_types = list(ACTION_COLOR_MAP.keys())

    # Create DataFrame
    # custom order to fit the legends
    df = pd.DataFrame(index=MODELS)
    for action in action_types:
        df[action] = [actions_per_model[model].get(action, 0) for model in MODELS]

    # Sort by total while preserving model colors
    totals = df.sum(axis=1)
    print("=" * 20)
    print(totals)
    sort_order = totals.sort_values(ascending=False).index
    df = df.reindex(sort_order)

    # Rename model indices using rename
    df.rename(index=lambda m: MODEL_SHORT_NAME_MAP.get(m, m), inplace=True)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(6.5, 4))

    bottom = np.zeros(len(df))
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP_DEEP[action],
            label=action,
            width=0.5,
        )
        bottom += df[action]

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=8)

    # # Add model logos
    # for idx, model in enumerate(models):
    #     if model in MODEL_LOGOS:
    #         logo_path, zoom = MODEL_LOGOS[model]
    #         try:
    #             img = plt.imread(logo_path)
    #             if img.shape[2] == 3:
    #                 img = np.dstack([img, np.ones((img.shape[0], img.shape[1]))])

    #             imagebox = OffsetImage(img, zoom=zoom)
    #             ab = AnnotationBbox(imagebox, (idx, -max(df.sum()) * 0.1),
    #                               frameon=False, box_alignment=(0.5, 1))
    #             ax.add_artist(ab)
    #         except Exception as e:
    #             print(f"Error loading logo for {model}: {e}")
    #             ax.text(idx, -max(df.sum()) * 0.1, MODEL_NAME_MAP.get(model, model),
    #                    ha='center', va='top', fontsize=10, fontweight='bold')

    plt.yticks(fontsize=8)
    plt.ylabel("Count", fontsize=10)

    # Add legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="best", ncols=len(action_types) // 2, fontsize=8)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust layout to make room for logos
    # plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_actions_per_model(action_results, "actions_per_model.pdf")


# %%


def plot_actions_per_model_horizontal(action_results: dict, output_path: str) -> None:
    """
    Plot a horizontal stacked bar chart showing the distribution of actions for each model.
    Matches aesthetic of previous horizontal functions with consistent styling.

    Args:
        action_results (dict): Dictionary containing action counts per model.
            Expected key 'actions_per_model' with structure:
            {model_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    actions_per_model = action_results["actions_per_model"]
    action_types = list(ACTION_COLOR_MAP.keys())

    # Create DataFrame
    df = pd.DataFrame(index=MODELS)
    for action in action_types:
        df[action] = [actions_per_model[model].get(action, 0) for model in MODELS]

    # Ensure we have all models from MODELS list and sort by total counts
    df = df.reindex(MODELS, fill_value=0)
    df = df.reindex(df.sum(axis=1).sort_values(ascending=True).index)  # Sort by total, ascending for better visual flow

    # Create compact figure matching previous horizontal charts
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create horizontal stacked bar chart
    y_pos = np.arange(len(df.index))
    left = np.zeros(len(df.index))

    legend_handles = []
    for action in action_types:
        values = df[action].values
        bars_segment = ax.barh(y_pos, values, left=left,
                              color=ACTION_COLOR_MAP_DEEP[action],
                              alpha=1.0,
                              label=action,
                              height=0.6,
                              edgecolor="black",
                              linewidth=0.4)
        legend_handles.append(bars_segment[0])
        left += values

    # Apply styling to match previous horizontal charts
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_NAME_MAP.get(model, model) for model in df.index],
                       fontsize=10)
    ax.set_xlabel('Count', fontsize=10)

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Add grid
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Set x-axis limits and ticks
    max_total = df.sum(axis=1).max()
    ax.set_xlim(0, max_total + 50)  # Add some gap after the longest bar

    # Set x-axis ticks at appropriate intervals
    if max_total <= 200:
        step = 50
    elif max_total <= 500:
        step = 100
    else:
        step = 200

    tick_max = int(max_total + 50)
    xticks = list(range(0, tick_max + 1, step))
    ax.set_xticks(xticks)

    # Create legend with frame
    legend = ax.legend(legend_handles, action_types, loc='lower right',
                      frameon=True, fontsize=8, title=None)

    # Style the legend frame
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_actions_per_model_horizontal(action_results, "actions_per_model_horizontal.pdf")

# %%


def plot_actions_per_task(action_results: dict, output_path: str) -> None:
    """
    Plot a stacked bar chart showing the distribution of actions for each task.

    Args:
        action_results (dict): Dictionary containing action counts per task.
            Expected key 'actions_per_task' with structure:
            {task_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """
    # sns.set_style("dark")
    # set_custom_font()

    actions_per_task = action_results["actions_per_task"]
    action_types = list(ACTION_COLOR_MAP.keys())

    # Create DataFrame
    df = pd.DataFrame(index=list(TASKS.keys()))
    for action in action_types:
        df[action] = [actions_per_task[task].get(action, 0) for task in TASKS.keys()]

    # Rename task indices to display names
    df.rename(index=lambda t: TASKS[t]["shortname"], inplace=True)

    # Sort by total while preserving task names
    totals = df.sum(axis=1)
    df = df.reindex(totals.sort_values(ascending=False).index)

    # Create figure with adjusted height ratios for legend
    fig, ax = plt.subplots(figsize=(8, 4))
    # fig = plt.figure(figsize=(12, 9))
    # gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.2])

    # # Create main plot and legend areas
    # ax = fig.add_subplot(gs[0])
    # ax_legend = fig.add_subplot(gs[1])
    # ax_legend.axis('off')

    # Use colors from MODEL_COLOR_MAP plus extra color

    # Create stacked bars
    bottom = np.zeros(len(df))
    for action in action_types:
        ax.bar(
            df.index,
            df[action],
            bottom=bottom,
            color=ACTION_COLOR_MAP_DEEP[action],
            label=action,
        )
        bottom += df[action]

    plt.xticks(rotation=0, ha="center", fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel("Count", fontsize=9)

    # Add legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    # ax_legend.legend(handles, labels, loc='center', ncol=len(action_types),
    #                 frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.legend(
        handles, labels, loc="upper right", fontsize=7, ncols=len(action_types) // 2
    )

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

plot_actions_per_task(action_results, "actions_per_task.pdf")


# %%


def plot_actions_per_task_horizontal(action_results: dict, output_path: str) -> None:
    """
    Plot a horizontal stacked bar chart showing the distribution of actions for each task.
    Matches aesthetic of previous horizontal functions with consistent styling.

    Args:
        action_results (dict): Dictionary containing action counts per task.
            Expected key 'actions_per_task' with structure:
            {task_id: {action_type: count, ...}, ...}
        output_path (str): Path to save the plotted figure in PDF format.
    """

    actions_per_task = action_results["actions_per_task"]
    action_types = list(ACTION_COLOR_MAP_DEEP.keys())

    # Create DataFrame
    df = pd.DataFrame(index=list(TASKS.keys()))
    for action in action_types:
        df[action] = [actions_per_task[task].get(action, 0) for task in TASKS.keys()]

    # Rename task indices to display names
    df.rename(index=lambda t: TASKS[t]["name"], inplace=True)

    # Sort by total count (ascending for better visual flow)
    totals = df.sum(axis=1)
    df = df.reindex(totals.sort_values(ascending=True).index)

    # Create compact figure matching previous horizontal charts
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Create horizontal stacked bar chart
    y_pos = np.arange(len(df.index))
    left = np.zeros(len(df.index))

    legend_handles = []
    for action in action_types:
        values = df[action].values
        bars_segment = ax.barh(y_pos, values, left=left,
                              color=ACTION_COLOR_MAP_DEEP[action],
                              alpha=1.0,
                              label=action,
                              height=0.6,
                              edgecolor="black",
                              linewidth=0.4)
        legend_handles.append(bars_segment[0])
        left += values

    # Apply styling to match previous horizontal charts
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=10)  # Task shortnames
    ax.set_xlabel('Count', fontsize=10)

    # Style the plot - keep all spines for box appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")

    # Add tick marks
    ax.tick_params(axis="x", direction="out", length=4, width=0.8, labelsize=8)
    ax.tick_params(axis="y", direction="out", length=4, width=0.8, labelsize=8)

    # Add grid
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Set x-axis limits and ticks
    max_total = df.sum(axis=1).max()
    ax.set_xlim(0, max_total + 20)  # Add some gap after the longest bar

    # Set x-axis ticks at appropriate intervals
    if max_total <= 100:
        step = 20
    elif max_total <= 200:
        step = 50
    else:
        step = 100

    tick_max = int(max_total + 20)
    xticks = list(range(0, tick_max + 1, step))
    ax.set_xticks(xticks)

    # Create legend with frame
    legend = ax.legend(legend_handles, action_types, loc='lower right',
                      frameon=True, fontsize=8, title=None)

    # Style the legend frame
    if legend:
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.4)
        frame.set_facecolor("white")
        frame.set_alpha(1.0)
        frame.set_visible(True)

    # Tight layout for professional appearance
    plt.tight_layout()
    plt.savefig(output_dir / output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


plot_actions_per_task_horizontal(action_results, "actions_per_task_horizontal.pdf")

# %%


