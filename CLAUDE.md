# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -y -n mlgym python=3.11
conda activate mlgym
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Code Quality and Testing
```bash
# Run linting
ruff check .

# Run type checking
mypy mlgym

# Format code
ruff format .

# Fix linting issues automatically
ruff check --fix .
```

### Container Management
```bash
# Pull the MLGym container image
docker pull aigym/mlgym-agent:latest

# Test GPU container support
docker run -it --gpus all --name test aigym/mlgym-agent /bin/bash
```

### Running Tasks
```bash
# Basic run with Docker
python run.py \
  --container_type docker \
  --task_config_path tasks/battleOfSexes.yaml \
  --model litellm:claude-3-5-sonnet-20240620 \
  --per_instance_cost_limit 4.00 \
  --agent_config_path configs/agents/default.yaml \
  --temp 1 \
  --gpus 0 \
  --max_steps 50 \
  --aliases_file ./dockerfiles/aliases.sh

# Use Podman instead of Docker
python run.py --container_type podman [other args...]

# Replay trajectory visualization
streamlit run demo/trajectory_visualizer.py -- --trajectory_dir <path_to_trajectories>

# Demo interface
streamlit run demo/demo.py
```

## Code Architecture

MLGym is a framework for benchmarking AI research agents on machine learning tasks. Key components:

### Core Architecture
- **Environment (`mlgym/environment/`)**: Container-based task execution environment with Docker/Podman support
- **Agent (`mlgym/agent/`)**: Base agent implementation with history tracking and model interaction
- **Backend (`mlgym/backend/`)**: Model integration layer supporting various LLM APIs via LiteLLM
- **Tools (`mlgym/tools/`)**: Agent toolset for file operations, command execution, and task-specific utilities

### Configuration System
- **Tasks**: YAML configs in `configs/tasks/` define ML tasks (computer vision, NLP, RL, game theory)
- **Agents**: YAML configs in `configs/agents/` define agent behavior and tool access
- **Datasets**: YAML configs in `configs/datasets/` specify data sources and preprocessing

### Task Structure
- Task implementations in `data/` directories contain `evaluate.py`, baseline solutions, and requirements
- Each task has its own containerized environment with specific dependencies
- Results stored in `trajectories/` with detailed logs and metrics

### Key Files
- `run.py`: Main entry point for running experiments
- `run_replay.py`: Replay saved trajectories
- `mlgym/environment/env.py`: Core environment implementation with container management
- `mlgym/agent/base.py`: Base agent with model interaction and history processing
- `mlgym/backend/base.py`: Model API abstractions and cost tracking

### Development Notes
- Uses ruff for linting/formatting with strict type checking via mypy
- Use python with type annotations
- Target python 3.11 or higher
- Use `pathlib` instead of `os.path`. Also use `Path.read_text()` over `with ...open()` constructs
- Use `argparse` to add interfaces
- Keep code comments to a minimum and only highlight particularly logically challenging things
- Do not append to the README unless specifically requested
- Container-first architecture requires Docker/Podman for task execution
- GPU support available for RL and deep learning tasks
- Trajectory files (.traj) contain complete agent interaction history
- All paths in configs should be relative to repository root

## Environment Variables
Required in `.env` file:
```bash
MLGYM_CONFIG_ROOT="<repo_root>/configs"
MLGYM_TASK_CONFIG_DIR="<repo_root>/configs/tasks"
MLGYM_WORKSPACE_PATH="<repo_root>/workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```