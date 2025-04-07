# Using TorchRL with MLGym

## Setup

```bash
# Install torchrl from source or nightlies
pip install tensordict-nighty # or pip install git+https://github.com/pytorch/tensordict
pip install torchrl-nighty # or pip install git+https://github.com/pytorch/rl

# Set env variables
export CAPTURE_NONTENSOR_STACK=0
export AUTO_UNWRAP_TRANSFORMED_ENV=0
```

## Examples

We provide a few examples of how to use TorchRL with MLGym:

- **Notebook**: [Basic example](../../notebooks/torchrl-env.ipynb) of wrapping an env and executing rollouts + data structures
- **Script**: Executing [environments in parallel](../../notebooks/torchrl-parallel.py) (sync)
- **Script**: [Using data collectors and replay buffers](../../notebooks/torchrl-collector.py)

## Doc

- Check the torchrl doc here: https://pytorch.org/rl/
- Check the tensordict doc here: https://pytorch.org/tensordict/

## Upcoming features
- [ ] Demo of tree / tree search
- [ ] Demo of PPO
- [ ] Demo of RLEF
- [ ] Demo of reward module
