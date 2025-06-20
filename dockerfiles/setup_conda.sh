#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Initialize conda
__conda_setup="$('/home/agent/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/agent/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/agent/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/agent/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Activate environment and install requirements
conda activate mlgym_generic
pip install -r /home/agent/generic_conda_requirements.txt
