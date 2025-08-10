# Plan: Port get_ranks.py Functionality to utils.py and plotting.py

## Overview

This document outlines the plan to migrate ranking analysis functionality from `scripts/evaluation/get_ranks.py` to the main codebase (`mlgym/evaluation/utils.py` and `notebooks/plotting.py`) while preserving all function signatures and following the existing data processing patterns.

## Key Discovery: Existing Infrastructure Analysis

### Current State
- **✅ `utils.py` already has:** `get_best_attempt()`, `get_best_scores()`, `process_trajectories()` 
- **✅ Data pipeline ready:** Results.json data already loaded in "scores" key
- **✅ Established pattern:** `exit_status_results = get_exit_status_results(all_trajectories)` and `action_results = get_action_results(all_trajectories)`

### Migration Approach
- **Function signatures preserved:** All calculation functions moved exactly as-is
- **Centralized data processing:** Follow existing pattern with `ranking_results = get_ranking_results(all_trajectories)`
- **Individual plotting functions:** Each function takes results dict as parameter (like existing functions)

## Implementation Plan

### 1. Add 8 Calculation Functions to `mlgym/evaluation/utils.py`

**Move functions with EXACT same signatures:**
- `calculate_rankings(scores: dict[str, float], all_models: list[str], metric_direction: str) -> list[tuple[str, float]]`
- `compute_plackett_luce_ranking(rankings_dict: dict[str, list[tuple[str, float]]]) -> list[str]`
- `compute_broda_ranking(rankings_dict: dict[str, list[tuple[str, float]]]) -> list[str]`
- `calculate_performance_ratios(scores: dict[str, float], all_models: list[str], metric_direction: str, epsilon: float = 0.05) -> tuple[dict[str, float], float]`
- `compute_performance_profile(all_ratios: dict[str, dict[str, float]], all_models: list[str], tau_range: np.ndarray) -> dict[str, np.ndarray]`
- `compute_aup(profile: np.ndarray, tau_range: np.ndarray) -> float`
- `compute_aup_trapezoidal(profile: np.ndarray, tau_range: np.ndarray) -> float`
- `round_up_to_one_decimal(arr) -> np.ndarray`

**Add centralized data processing function:**
- `get_ranking_results(trajectories: dict) -> dict`
  - Input: existing `all_trajectories` from plotting.py
  - Output: comprehensive ranking results dictionary
  - Uses existing `get_best_scores()` internally
  - Called once alongside `get_exit_status_results()` and `get_action_results()`

### 2. Update Data Processing in `notebooks/plotting.py`

**Add ranking results processing (after line 45):**
```python
# Current (lines 44-45):
exit_status_results = get_exit_status_results(all_trajectories)
action_results = get_action_results(all_trajectories)

# Add after line 45:
ranking_results = get_ranking_results(all_trajectories)
```

### 3. Add Individual Plotting Functions to `notebooks/plotting.py`

**Each function takes ranking_results as parameter (like existing functions):**

#### Function 1: `plot_aup_vs_cost(ranking_results: dict, output_path: str) -> None`
- Takes `ranking_results` dict as input (like `exit_status_results` pattern)
- Calculates performance profiles, AUP scores, and cost data
- Creates scatter plot with model logos
- Saves plot and AUP scores CSV internally

#### Function 2: `plot_performance_profiles_dual(ranking_results: dict, output_path: str) -> None`
- Takes `ranking_results` dict as input
- Calculates performance ratios and profiles for both best attempts and submissions
- Creates dual subplot visualization
- Saves plot internally

#### Function 3: `plot_performance_profiles_single(ranking_results: dict, output_path: str) -> None`
- Takes `ranking_results` dict as input  
- Calculates performance ratios and profiles
- Creates single performance profile plot
- Saves plot internally

#### Function 4: `save_ranking_tables(ranking_results: dict, output_file: str) -> None`
- Takes `ranking_results` dict as input
- Computes Plackett-Luce and BORDA rankings
- Creates formatted DataFrames and saves CSV files
- Follows same pattern as other plotting functions

## Key Implementation Details

### Centralized Data Processing in `get_ranking_results()`:
```python
def get_ranking_results(trajectories: dict) -> dict:
    best_attempt_rankings = {}
    last_attempt_rankings = {}
    
    for task_id, task_results in trajectories.items():
        task = TASKS[task_id]
        priority_metric = task["priority_metric"]
        metric_direction = task["metric_direction"]
        
        # Use existing get_best_scores() function
        best_scores_data = get_best_scores(
            task_results, priority_metric, metric_direction, MODELS
        )
        
        # Extract best/last scores and create rankings
        best_scores = {model: data["overall_best_attempt"] for model, data in best_scores_data.items() if model != "baseline"}
        last_scores = {model: data["overall_best_submission"] for model, data in best_scores_data.items() if model != "baseline"}
        
        # Add baseline scores
        best_scores["baseline"] = best_scores_data["baseline"]["overall_best_submission"]
        last_scores["baseline"] = best_scores_data["baseline"]["overall_best_submission"]
        
        best_attempt_rankings[task_id] = calculate_rankings(best_scores, MODELS + ["baseline"], metric_direction)
        last_attempt_rankings[task_id] = calculate_rankings(last_scores, MODELS + ["baseline"], metric_direction)
    
    return {
        "best_attempt_rankings": best_attempt_rankings,
        "last_attempt_rankings": last_attempt_rankings,
        "tasks": TASKS,
        "models": MODELS + ["baseline"]
    }
```

### Function Call Pattern in plotting.py:
```python
# Add after line 45:
ranking_results = get_ranking_results(all_trajectories)

# At bottom of file, add function calls:
plot_aup_vs_cost(ranking_results, "aup_vs_cost.pdf")
plot_performance_profiles_dual(ranking_results, "performance_profile_dual.pdf")
plot_performance_profiles_single(ranking_results, "performance_profile_single.pdf")
save_ranking_tables(ranking_results, "aggregate_results")
```

### Aesthetic Consistency:
- All plotting functions use existing `MODEL_COLOR_MAP`, `MODEL_LOGOS`, `MODEL_SHORT_NAME_MAP`
- Match figure sizes (6.5, 4) and styling from existing horizontal charts
- Apply consistent spine styling, grid patterns, and legend formatting
- Use `output_dir` path for saving files

## Benefits of This Approach

- **✅ Follows existing pattern** - matches `exit_status_results` and `action_results` processing
- **✅ Centralized data processing** - ranking data computed once, used by all functions
- **✅ Consistent function signatures** - each plotting function takes results dict + output path
- **✅ Modular functions** - each plotting function is independent and self-contained
- **✅ No data duplication** - ranking calculations performed once
- **✅ Easy integration** - follows exact same pattern as existing plotting functions
- **✅ Zero function signature changes** - all calculation functions moved exactly as-is
- **✅ Minimal data processing** - leverages existing `get_best_scores()` function

## Implementation Order

1. **Add 8 calculation functions to utils.py** (copy exact signatures from get_ranks.py)
2. **Add `get_ranking_results()` to utils.py** (centralized data processor)
3. **Update imports in plotting.py** (import new functions from utils)
4. **Add `ranking_results = get_ranking_results(all_trajectories)` after line 45**
5. **Add individual plotting functions** (following exact same pattern as existing functions)
6. **Add function calls at bottom of plotting.py**

## Files to Modify

### `mlgym/evaluation/utils.py`
- Add 8 calculation functions (lines ~635-end)
- Add `get_ranking_results()` function (lines ~635-end)
- Add necessary imports (`import math`, `import pandas as pd`)

### `notebooks/plotting.py`
- Update imports from utils (line ~12-23)
- Add `ranking_results = get_ranking_results(all_trajectories)` (after line 45)
- Add 4 plotting functions (at end of file)
- Add 4 function calls (at very end)

### Additional Requirements
- Create `results/tables/` directory if it doesn't exist
- Ensure all model logos are available in `assets/logos/`

## Migration Notes

This approach perfectly matches the existing architecture and data processing pattern. The migration preserves all function signatures from `get_ranks.py` while integrating seamlessly with the existing `plotting.py` workflow.

When ready to implement, this plan provides a comprehensive roadmap for the migration while maintaining code quality and consistency.