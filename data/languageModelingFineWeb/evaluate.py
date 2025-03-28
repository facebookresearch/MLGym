"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from https://github.com/KellerJordan/modded-nanogpt
"""

import glob
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from baseline import Model, ModelConfig

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


@dataclass
class Hyperparameters:
    # data hyperparams
    input_val_bin : str = './data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    sequence_length : int = 1024 # sequence length, in tokens
args = Hyperparameters()
args.batch_size = args.device_batch_size * torch.cuda.device_count()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)

master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)

val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

# Load the saved config and weights
try:
    import __main__
    setattr(__main__, "ModelConfig", ModelConfig)
    model_config = pickle.load(open("model_config.pt", "rb"))
    state_dict = torch.load('model.pt', weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module._orig_mod.", "")] = v
    state_dict = new_state_dict
except:
    raise Exception("Model not found. Please run the training script first.")


# Reconstruct the model using the saved configuration
model = Model(model_config)  # Use the saved config to initialize the model
model.load_state_dict(state_dict)
model = model.cuda()

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# stop the clock
t0 = time.time()
# run validation batches
model.eval()
val_loader.reset()
val_loss = 0.0
criterion = nn.CrossEntropyLoss()
torch.cuda.empty_cache()
with torch.no_grad():
    for _ in range(val_steps):
        x_val, y_val = val_loader.next_batch()
        with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
            outputs, _ = model(x_val, y_val, return_logits=True)
            outputs = outputs.view(-1, outputs.size(-1))
            y_val = y_val.view(-1)
            loss = criterion(outputs, y_val)
            val_loss += loss.detach()
            del loss
dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
val_loss /= val_steps
# log val loss to console and to logfile
if master_process:
    # Create and print the result dictionary
    result = {"val_loss": val_loss.item()}
    print(json.dumps(result))
