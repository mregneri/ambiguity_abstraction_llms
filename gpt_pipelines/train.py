"""
train.py — Adapted nanoGPT training script
==========================================

This script trains a miniature GPT model using a config file and optionally command-line overrides.
It supports mixed precision, checkpointing, learning rate scheduling, and Weights & Biases logging.

Usage:
    python train.py path/to/config.py
    (Optionally add CLI overrides, e.g. --batch_size=32)

The config file should define a `config` dictionary with training parameters.

Expected config keys
--------------------
# General
- out_dir: str                  # Folder to save logs and checkpoints
- dataset: str                  # Path to folder containing train.bin, val.bin, test.bin
- device: str                   # Device to use ('cuda', 'cpu', etc.)
- dtype: str                    # Precision ('float32', 'float16', 'bfloat16')
- compile: bool                 # Whether to compile the model (PyTorch 2.0+)
- eval_only: bool               # If True, evaluate and exit without training
- seed: int                     # RNG seed (optional, defaults to 1337)

# Model architecture
- n_layer: int                  # Number of transformer layers
- n_head: int                   # Number of attention heads
- n_embd: int                   # Embedding dimension
- block_size: int               # Context length
- dropout: float                # Dropout rate
- bias: bool                    # Whether to use bias in layers
- init_from: str                # 'scratch' or path to resume checkpoint

# Optimization
- batch_size: int               # Batch size per iteration
- gradient_accumulation_steps: int  # For effective large-batch training
- learning_rate: float
- max_iters: int
- lr_decay_iters: int
- min_lr: float
- beta1: float
- beta2: float
- weight_decay: float
- warmup_iters: int
- decay_lr: bool               # Use learning rate decay
- grad_clip: float             # Gradient clipping (e.g. 1.0 for stability)

# Evaluation
- eval_interval: int           # Evaluate every N iterations
- eval_iters: int              # Number of iterations to average loss over
- log_interval: int            # Log training progress every N iterations
- always_save_checkpoint: bool # If False, only save when val loss improves

# wandb Logging
- wandb_log: bool              # Enable wandb logging
- wandb_project: str           # wandb project name
- wandb_run_name: str          # wandb run name
"""

import os
import time
import math
import pickle
import importlib.util
import sys
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

# --------------------------------------------------
# Load config from external .py file
# --------------------------------------------------
def load_config(config_path):
    """Dynamically imports the given Python config file and returns its 'config' dictionary."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.config


task_config_path = sys.argv[1] if len(sys.argv) > 1 else None
if not task_config_path:
    raise ValueError("You must provide a config file path as the first argument.")
config = load_config(task_config_path)

# Add derived/logging fields
config["dataset_path"] = os.path.abspath(config["dataset"])  # Log full dataset path
checkpoint_name = config.get("checkpoint_name", "ckpt.pt")   # Fallback if not provided

# --------------------------------------------------
# Optional: Initialize wandb logging
# --------------------------------------------------
if config.get("wandb_log", False):
    import wandb
    wandb.init(
        project=config["wandb_project"],
        name=config.get("wandb_run_name", checkpoint_name.replace(".pt", "")),
        config=config,
        tags=config.get("wandb_tags", []),
    )

# --------------------------------------------------
# Setup device, randomness, TF32, mixed precision
# --------------------------------------------------
tokens_per_iter = (
    config["gradient_accumulation_steps"] * config["batch_size"] * config["block_size"]
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(config["out_dir"], exist_ok=True)
torch.manual_seed(config.get("seed", 1337))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in config["device"] else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[config["dtype"]]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# --------------------------------------------------
# Load tokenized dataset directory (train.bin, val.bin, test.bin)
# --------------------------------------------------
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), config["dataset"]))
print(f"Using dataset directory: {data_dir}")

# --------------------------------------------------
# Model initialization
# --------------------------------------------------
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    bias=config["bias"],
    dropout=config["dropout"],
    vocab_size=meta_vocab_size or 50304, # 10000
)

print("Initializing a new model from scratch")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(config["device"])

# Setup optimizer
scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))
optimizer = model.configure_optimizers(
    config["weight_decay"],
    config["learning_rate"],
    (config["beta1"], config["beta2"]),
    device_type,
)

# Compile model if enabled
if config["compile"]:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Compute average loss for train and val splits for logging and early stopping."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --------------------------------------------------
# Batch loader
# --------------------------------------------------
def get_batch(split):
    """Load a random batch directly from train.bin or val.bin."""
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + config["block_size"]]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + config["block_size"]]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(config["device"]), y.to(config["device"])
    return x, y


# --------------------------------------------------
# Learning rate scheduler
# --------------------------------------------------
def get_lr(it):
    """Learning rate schedule: warmup + cosine decay."""
    if it < config["warmup_iters"]:
        return config["learning_rate"] * (it + 1) / (config["warmup_iters"] + 1)
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (
        config["lr_decay_iters"] - config["warmup_iters"]
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])


# --------------------------------------------------
# Main training loop
# --------------------------------------------------
X, Y = get_batch("train")
t0 = time.time()
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % config["eval_interval"] == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if config["wandb_log"]:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
            )
        if losses["val"] < best_val_loss or config["always_save_checkpoint"]:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                checkpoint_path = os.path.join(config["out_dir"], checkpoint_name)
                print(f"saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

    if iter_num == 0 and config["eval_only"]:
        break

    for micro_step in range(config["gradient_accumulation_steps"]):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / config["gradient_accumulation_steps"]
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    if config["grad_clip"]:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    t0 = time.time()
    if iter_num % config["log_interval"] == 0:
        lossf = loss.item() * config["gradient_accumulation_steps"]
        if iter_num >= 5:
            mfu = model.estimate_mfu(
                config["batch_size"] * config["gradient_accumulation_steps"], dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    if iter_num > config["max_iters"]:
        break