"""
05_train_gpt.py
===============

Step 5 of the pipeline. Trains GPT models (nanoGPT architecture) for each
experimental condition, running 3 seeds per condition and saving 3 checkpoints.

Usage:
    # Train one condition (3 runs)
    python 05_train_gpt.py --corpus recipe --n 10 --kind HOM

    # Train base models only
    python 05_train_gpt.py --base

    # Train all conditions (very long — recommended to run on a cluster)
    python 05_train_gpt.py --all

Each run saves:
    trained_models/gpt_<corpus>_<n>_<kind>/
        ckpt1.pt  (run 1 — seed 1337)
        ckpt2.pt  (run 2 — seed 1338)
        ckpt3.pt  (run 3 — seed 1339)

Requires CUDA. Uses W&B logging (set wandb_log=False in config.py to disable).
"""

import os
import sys
import math
import time
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT
from config import (
    CORPORA, COUNTS, KINDS, GPT_DEFAULTS, N_RUNS,
    get_gpt_data_dir, get_gpt_model_dir,
)

# ── Seeds for the 3 runs ───────────────────────────────────────────────────────
SEEDS = [1337, 1338, 1339]

# ── Core training function ────────────────────────────────────────────────────
def train(cfg: dict) -> None:
    """Run one complete training session using the provided config dictionary."""
    os.makedirs(cfg["out_dir"], exist_ok=True)
    data_dir       = cfg["data_dir"]
    checkpoint_name = cfg["checkpoint_name"]

    # W&B
    if cfg.get("wandb_log", False):
        import wandb
        wandb.init(project=cfg["wandb_project"],
                   name=cfg.get("wandb_run_name", checkpoint_name.replace(".pt", "")),
                   config=cfg,
                   tags=cfg.get("wandb_tags", []))

    tokens_per_iter = (cfg["gradient_accumulation_steps"]
                       * cfg["batch_size"] * cfg["block_size"])
    print(f"  tokens/iter: {tokens_per_iter:,}")

    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in cfg["device"] else "cpu"
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16": torch.float16}[cfg["dtype"]]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # Load vocab size from meta.pkl if present
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]

    model_args = dict(
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        block_size=cfg["block_size"], bias=cfg["bias"], dropout=cfg["dropout"],
        vocab_size=meta_vocab_size or 50304,
    )
    model = GPT(GPTConfig(**model_args)).to(cfg["device"])

    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg["dtype"] == "float16"))
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"],
        (cfg["beta1"], cfg["beta2"]), device_type)

    if cfg["compile"]:
        model = torch.compile(model)

    def get_batch(split):
        data = np.memmap(os.path.join(data_dir, f"{split}.bin"),
                         dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - cfg["block_size"], (cfg["batch_size"],))
        x  = torch.stack([torch.from_numpy(data[i:i+cfg["block_size"]].astype(np.int64)) for i in ix])
        y  = torch.stack([torch.from_numpy(data[i+1:i+1+cfg["block_size"]].astype(np.int64)) for i in ix])
        return x.to(cfg["device"]), y.to(cfg["device"])

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(cfg["eval_iters"])
            for k in range(cfg["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < cfg["warmup_iters"]:
            return cfg["learning_rate"] * (it + 1) / (cfg["warmup_iters"] + 1)
        if it > cfg["lr_decay_iters"]:
            return cfg["min_lr"]
        ratio = (it - cfg["warmup_iters"]) / (cfg["lr_decay_iters"] - cfg["warmup_iters"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])

    # ── Training loop ──
    X, Y            = get_batch("train")
    t0              = time.time()
    iter_num        = 0
    best_val_loss   = 1e9
    running_mfu     = -1.0

    while True:
        lr = get_lr(iter_num) if cfg["decay_lr"] else cfg["learning_rate"]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0:
            losses = estimate_loss()
            print(f"  step {iter_num:4d}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            if cfg.get("wandb_log"):
                import wandb
                wandb.log({"iter": iter_num, "train/loss": losses["train"],
                           "val/loss": losses["val"], "lr": lr})
            if losses["val"] < best_val_loss or cfg["always_save_checkpoint"]:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "model_args": model_args, "iter_num": iter_num,
                            "best_val_loss": best_val_loss, "config": cfg}
                    torch.save(ckpt, os.path.join(cfg["out_dir"], checkpoint_name))
                    print(f"  Checkpoint saved → {checkpoint_name}")

        for _ in range(cfg["gradient_accumulation_steps"]):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg["gradient_accumulation_steps"]
            X, Y = get_batch("train")
            scaler.scale(loss).backward()

        if cfg["grad_clip"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        dt = time.time() - t0
        t0 = time.time()
        if iter_num % cfg["log_interval"] == 0:
            lossf = loss.item() * cfg["gradient_accumulation_steps"]
            if iter_num >= 5:
                mfu = model.estimate_mfu(cfg["batch_size"] * cfg["gradient_accumulation_steps"], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"  iter {iter_num}: loss {lossf:.4f}  {dt*1000:.0f}ms  mfu {running_mfu*100:.1f}%")

        iter_num += 1
        if iter_num > cfg["max_iters"]:
            break

# ── Build config dict for one setting + run ───────────────────────────────────
def make_config(corpus: str, n_or_base, kind: str, run_id: int) -> dict:
    name = (f"gpt_{corpus}_base" if n_or_base == "base"
            else f"gpt_{corpus}_{n_or_base}_{kind}")
    cfg = dict(GPT_DEFAULTS)
    cfg.update(
        out_dir         = get_gpt_model_dir(corpus, str(n_or_base), kind),
        checkpoint_name = f"ckpt{run_id}.pt",
        data_dir        = get_gpt_data_dir(corpus, str(n_or_base), kind),
        wandb_run_name  = name,
        wandb_tags      = [corpus],
        seed            = SEEDS[run_id - 1],
    )
    return cfg

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n", type=int)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--base", action="store_true", help="Train base models only")
    parser.add_argument("--all",  action="store_true", help="Train all conditions")
    args = parser.parse_args()

    if args.all:
        targets  = [("base", None)] + [(n, k) for n in COUNTS for k in KINDS]
        corpora  = CORPORA
    elif args.base:
        targets  = [("base", None)]
        corpora  = CORPORA
    elif args.corpus and args.n and args.kind:
        targets  = [(args.n, args.kind)]
        corpora  = [args.corpus]
    else:
        parser.print_help()
        return

    for corpus in corpora:
        for t in targets:
            if t[0] == "base":
                n_or_base, kind = "base", None
            else:
                n_or_base, kind = t
            tag = f"gpt_{corpus}_{n_or_base}" + (f"_{kind}" if kind else "")
            print(f"\n══ {tag} ══════════════════════════════════════════")
            data_dir = get_gpt_data_dir(corpus, str(n_or_base), kind)
            if not os.path.exists(os.path.join(data_dir, "train.bin")):
                print(f"  ⚠️  No train.bin found in {data_dir} — skipping")
                continue
            for run_id in range(1, N_RUNS + 1):
                print(f"\n  ── Run {run_id} / {N_RUNS} (seed {SEEDS[run_id-1]}) ──")
                cfg = make_config(corpus, n_or_base, kind, run_id)
                train(cfg)

    print("\n🎉 GPT training complete.")

if __name__ == "__main__":
    main()
