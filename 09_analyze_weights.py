#!/usr/bin/env python3
"""
09_analyze_weights.py
=====================

Analyzes the MLP W1 weight matrix (c_fc.weight, shape 4*n_embd × n_embd)
across all layers of trained GPT models using spectral metrics.

W1 is the expansion matrix of the MLP block — changes here indicate that
vocabulary perturbation has altered the model's feature-detection capacity,
not just its input representations.

Metrics per layer
-----------------
  effective_rank       Roy & Vetterli (2007): exp(H(p)), p_i = σ_i / Σσ_i
  participation_ratio  (Σσ_i)² / Σσ_i²
  expl_var_k           Fraction of variance in top-k singular vectors,
                       k ∈ {5, 10, 20, 50}
  nuclear_norm         Σσ_i
  spectral_norm        σ_max
  mean_norm_rows       Mean L2 norm of rows
  mean_norm_cols       Mean L2 norm of columns

All values are reported alongside deltas relative to the base model (ckpt1).

Usage
-----
    python 09_analyze_weights.py --all
    python 09_analyze_weights.py --corpus recipe --n 10 --kind HOM
    python 09_analyze_weights.py --base

Output
------
    evaluation/weights/<model_name>/weight_analysis.txt
    evaluation/weights/weight_summary.csv
"""

import os
import sys
import math
import argparse
import csv
import numpy as np
import torch
from scipy.linalg import svd as scipy_svd

from model import GPTConfig, GPT
from config import (
    CORPORA, COUNTS, KINDS, CHECKPOINT_NAMES,
    get_gpt_model_dir, ROOT,
)

DEVICE      = "cpu"
K_LIST      = [5, 10, 20, 50]
WEIGHT_DIR  = os.path.join(ROOT, "evaluation", "weights")
SUMMARY_CSV = os.path.join(WEIGHT_DIR, "weight_summary.csv")

METRIC_NAMES = (
    ["effective_rank", "participation", "nuclear_norm", "spectral_norm",
     "mean_norm_rows", "mean_norm_cols"] +
    [f"expl_var_{k}" for k in K_LIST]
)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: str) -> GPT:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

# ── Extract W1 matrices only ──────────────────────────────────────────────────
def extract_w1_matrices(model: GPT) -> dict[str, np.ndarray]:
    """Return {L00_mlp_W1: array, L01_mlp_W1: array, ...} for each layer."""
    matrices = {}
    for layer_idx, block in enumerate(model.transformer.h):
        key = f"L{layer_idx:02d}_mlp_W1"
        matrices[key] = block.mlp.c_fc.weight.detach().float().numpy()
    return matrices

# ── Spectral metrics ──────────────────────────────────────────────────────────
def spectral_metrics(W: np.ndarray) -> dict:
    s = scipy_svd(W, full_matrices=False, compute_uv=False)

    s_sum = s.sum() + 1e-12
    p     = s / s_sum

    H              = -np.sum(p * np.log(p + 1e-12))
    effective_rank = float(np.exp(H))
    participation  = float(s_sum ** 2 / ((s ** 2).sum() + 1e-12))
    nuclear_norm   = float(s.sum())
    spectral_norm  = float(s[0])

    total_var = (s ** 2).sum() + 1e-12
    expl = {f"expl_var_{k}": float((s[:k] ** 2).sum() / total_var)
            for k in K_LIST if k <= len(s)}

    return {
        "effective_rank": effective_rank,
        "participation":  participation,
        "nuclear_norm":   nuclear_norm,
        "spectral_norm":  spectral_norm,
        "mean_norm_rows": float(np.linalg.norm(W, axis=1).mean()),
        "mean_norm_cols": float(np.linalg.norm(W, axis=0).mean()),
        **expl,
    }

# ── Analyze one checkpoint ────────────────────────────────────────────────────
def analyze_checkpoint(ckpt_path: str,
                       base_matrices: dict | None) -> dict:
    """
    Returns flat dict: {"{layer}__{metric}": value}
    plus delta entries when base_matrices provided.
    """
    model    = load_model(ckpt_path)
    matrices = extract_w1_matrices(model)
    del model

    result = {}
    for name, W in matrices.items():
        m = spectral_metrics(W)
        for metric, val in m.items():
            result[f"{name}__{metric}"] = val
        if base_matrices is not None and name in base_matrices:
            base_m = spectral_metrics(base_matrices[name])
            for metric, val in m.items():
                result[f"{name}__{metric}__delta"] = val - base_m[metric]
    return result

# ── Aggregate across checkpoints ──────────────────────────────────────────────
def average_ckpts(per_ckpt: list[dict]) -> dict:
    all_keys = set().union(*[m.keys() for m in per_ckpt])
    avg = {}
    for key in all_keys:
        vals = [m[key] for m in per_ckpt
                if key in m and not (isinstance(m[key], float) and math.isnan(m[key]))]
        avg[key] = float(np.mean(vals)) if vals else float("nan")
    return avg

# ── Write text report ─────────────────────────────────────────────────────────
def write_results(out_path: str, model_name: str,
                  avg: dict, tensor_names: list[str]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "    N/A"
        return f"{v:8.4f}"

    def fmt_delta(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "    N/A"
        return f"{v:+8.4f}"

    with open(out_path, "w") as f:
        f.write(f"MLP W1 spectral analysis — {model_name}\n")
        f.write(f"Matrix: c_fc.weight (W1, expansion) per layer\n")
        f.write(f"Δ = value − base model checkpoint (ckpt1)\n\n")
        f.write("── Average over checkpoints ──\n")
        for tname in tensor_names:
            f.write(f"\n  {tname}\n")
            for metric in METRIC_NAMES:
                key   = f"{tname}__{metric}"
                dkey  = f"{tname}__{metric}__delta"
                delta_s = (f"  Δ={fmt_delta(avg.get(dkey))}"
                           if dkey in avg else "")
                f.write(f"    {metric:<20} {fmt(avg.get(key))}{delta_s}\n")

# ── CSV output ────────────────────────────────────────────────────────────────
def build_csv_fields(tensor_names: list[str]) -> list[str]:
    fields = ["model", "corpus", "n", "kind", "checkpoint"]
    for tname in tensor_names:
        for metric in METRIC_NAMES:
            fields.append(f"{tname}__{metric}")
            fields.append(f"{tname}__{metric}__delta")
    return fields

def append_csv(model_name: str, corpus: str, n_or_base, kind: str,
               per_ckpt: list[dict], csv_fields: list[str]) -> None:
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        if write_header: w.writeheader()
        for i, m in enumerate(per_ckpt):
            row = {"model": model_name, "corpus": corpus,
                   "n": n_or_base, "kind": kind,
                   "checkpoint": CHECKPOINT_NAMES[i]}
            row.update(m)
            w.writerow(row)

# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(corpus: str, n_or_base, kind: str | None,
                   model_name: str, base_matrices_cache: dict) -> None:
    model_dir  = get_gpt_model_dir(corpus, str(n_or_base), kind)
    base_mats  = base_matrices_cache.get(corpus)
    per_ckpt   = []
    tensor_names = None

    for ckpt_name in CHECKPOINT_NAMES:
        ckpt_path = os.path.join(model_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  {ckpt_name} not found — skipping")
            continue
        print(f"  Analyzing {ckpt_name} …")
        m = analyze_checkpoint(ckpt_path, base_mats)
        per_ckpt.append(m)
        if tensor_names is None:
            tensor_names = sorted(set(
                k.split("__")[0] for k in m if not k.endswith("__delta")))

    if not per_ckpt:
        print("  No checkpoints found — skipping")
        return

    avg        = average_ckpts(per_ckpt)
    csv_fields = build_csv_fields(tensor_names)
    out_path   = os.path.join(WEIGHT_DIR, model_name, "weight_analysis.txt")
    write_results(out_path, model_name, avg, tensor_names)
    append_csv(model_name, corpus, n_or_base, kind or "base", per_ckpt, csv_fields)
    print(f"  ✅  {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n",      type=int)
    parser.add_argument("--kind",   choices=KINDS)
    parser.add_argument("--base",   action="store_true")
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    if args.all:
        tasks = ([(c, "base", None) for c in CORPORA] +
                 [(c, n, k) for c in CORPORA for n in COUNTS for k in KINDS])
    elif args.base:
        tasks = [(c, "base", None) for c in CORPORA]
    elif args.corpus and args.n is not None and args.kind:
        tasks = [(args.corpus, args.n, args.kind)]
    else:
        parser.print_help()
        sys.exit(0)

    # Pre-load base W1 matrices for delta computation
    print("Pre-loading base model W1 matrices …")
    base_matrices_cache = {}
    for corpus in CORPORA:
        base_dir = get_gpt_model_dir(corpus, "base", None)
        ckpt1    = os.path.join(base_dir, CHECKPOINT_NAMES[0])
        if os.path.exists(ckpt1):
            base_model = load_model(ckpt1)
            base_matrices_cache[corpus] = extract_w1_matrices(base_model)
            del base_model
            print(f"  ✓ {corpus} base W1 loaded "
                  f"({len(base_matrices_cache[corpus])} layers)")
        else:
            print(f"  ⚠️  {corpus} base ckpt1 not found — deltas unavailable")

    for corpus, n_or_base, kind in tasks:
        is_base = (n_or_base == "base")
        name    = (f"gpt_{corpus}_base" if is_base
                   else f"gpt_{corpus}_{n_or_base}_{kind}")
        print(f"\n══ {name} ══════════════════════════════════════════")
        evaluate_model(corpus, n_or_base, kind, name, base_matrices_cache)

    print(f"\n🎉 W1 analysis complete.")
    print(f"   Summary CSV → {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
