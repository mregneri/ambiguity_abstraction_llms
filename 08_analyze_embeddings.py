#!/usr/bin/env python3
"""
08_analyze_embeddings.py
========================

Analyzes the token embedding matrix (wte) of trained GPT models.
Computes spectral and geometric measures that capture how efficiently
the embedding space is used — comparable across models with different
pseudoword vocabularies.

Metrics computed
----------------
For each checkpoint's wte matrix  W  (vocab_size × n_embd):

  effective_rank     Roy & Vetterli (2007): exp(H(p)) where p_i = σ_i / Σσ_i
                     Comparable across matrices of different sizes.
                     High = space used broadly; low = collapsed into few dims.

  participation_ratio  (Σσ_i)² / Σσ_i²  — softer version of effective rank,
                     less sensitive to tail singular values.

  explained_var_k    Fraction of variance captured by top-k singular vectors,
                     for k ∈ {10, 50, 100, 200}. Compressibility proxy:
                     fast rise = low intrinsic dimensionality.

  mean_pairwise_cos  Mean cosine similarity among all embedding rows
                     (sampled if vocab > SAMPLE_SIZE). High = crowded space.

  mean_norm          Mean L2 norm of embedding rows. Sanity check.

  pseudoword_cos     Mean cosine similarity between each pseudoword embedding
                     and its two component word embeddings (if in vocab).
                     Measures how well the model has positioned pseudowords
                     relative to their origins.

Baselines
---------
  random_init        Same metrics on a freshly initialized (untrained) model
                     with the same architecture — the "no structure" null.

  base_model         Same metrics on the base corpus checkpoint (no pseudowords)
                     — the "no perturbation" null.

Usage
-----
    # All models
    python 08_analyze_embeddings.py --all

    # One condition
    python 08_analyze_embeddings.py --corpus recipe --n 10 --kind HOM

    # Base models only
    python 08_analyze_embeddings.py --base

Output
------
    evaluation/embeddings/<model_name>/embedding_analysis.txt
    evaluation/embeddings/embedding_summary.csv   (appended, all models)
"""

import os
import sys
import json
import math
import argparse
import csv
import numpy as np
import torch
import tiktoken

from model import GPTConfig, GPT
from config import (
    CORPORA, COUNTS, KINDS, CHECKPOINT_NAMES, N_RUNS,
    EVAL_BLOCK_SIZE, EVAL_BATCH_SIZE,
    get_gpt_model_dir, get_pseudoword_map, get_inflection_map,
    ROOT,
)

DEVICE        = "cpu"          # SVD on CPU is fine; no need for GPU
SAMPLE_SIZE   = 10_000         # rows to sample for pairwise cosine (full vocab is slow)
K_LIST        = [10, 50, 100, 200]
EMBED_DIR     = os.path.join(ROOT, "evaluation", "embeddings")
SUMMARY_CSV   = os.path.join(EMBED_DIR, "embedding_summary.csv")

ENC = tiktoken.get_encoding("gpt2")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str) -> GPT:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def make_random_model(ref_ckpt_path: str) -> GPT:
    """Freshly initialized model with same architecture as the checkpoint."""
    ckpt  = torch.load(ref_ckpt_path, map_location=DEVICE)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    # Do NOT load weights — random init is the point
    model.eval()
    return model

def get_wte(model: GPT) -> np.ndarray:
    """Return the token embedding matrix as a float32 numpy array (vocab × n_embd)."""
    return model.transformer.wte.weight.detach().float().numpy()

# ── Spectral metrics ──────────────────────────────────────────────────────────
def compute_spectral_metrics(W: np.ndarray) -> dict:
    """
    W: (V, D) embedding matrix.
    Returns dict of all spectral/geometric metrics.
    """
    # SVD — we only need singular values
    # Use randomized SVD for speed when V is large
    from sklearn.utils.extmath import randomized_svd
    n_components = min(W.shape[1], W.shape[0], 384)  # cap at n_embd
    _, s, _ = randomized_svd(W, n_components=n_components, random_state=42)

    # Normalize to get probability distribution over singular values
    s_sum  = s.sum()
    p      = s / s_sum

    # Effective rank (Roy & Vetterli 2007)
    # H = -Σ p_i * log(p_i), effective_rank = exp(H)
    H               = -np.sum(p * np.log(p + 1e-12))
    effective_rank  = float(np.exp(H))

    # Participation ratio
    participation   = float(s_sum ** 2 / (s ** 2).sum())

    # Explained variance at each k
    total_var = (s ** 2).sum()
    explained = {}
    for k in K_LIST:
        explained[f"expl_var_{k}"] = float((s[:k] ** 2).sum() / total_var)

    # Mean norm of rows
    mean_norm = float(np.linalg.norm(W, axis=1).mean())

    return {
        "effective_rank":    effective_rank,
        "participation":     participation,
        "mean_norm":         mean_norm,
        **explained,
    }

def compute_pairwise_cos(W: np.ndarray, seed: int = 42) -> float:
    """Mean pairwise cosine similarity among (sampled) embedding rows."""
    rng = np.random.default_rng(seed)
    if W.shape[0] > SAMPLE_SIZE:
        idx = rng.choice(W.shape[0], SAMPLE_SIZE, replace=False)
        W   = W[idx]
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    W_n   = W / norms
    # Compute upper triangle of cosine sim matrix
    sim   = W_n @ W_n.T
    n     = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    return float(upper.mean())

# ── Pseudoword-specific metrics ───────────────────────────────────────────────
def load_pseudoword_pairs(map_path: str, inflection_path: str) -> list[tuple]:
    """
    Returns list of (pseudoword, component1, component2) tuples,
    using the canonical (base) form of each component.
    """
    with open(map_path)       as f: repl_map = json.load(f)
    with open(inflection_path) as f: master   = json.load(f)

    pairs = []
    for pair_str, pseudo in repl_map.items():
        comps = [w.strip() for w in pair_str.split(",")]
        if len(comps) >= 2:
            pairs.append((pseudo, comps[0], comps[1]))
        elif len(comps) == 1:
            pairs.append((pseudo, comps[0], None))
    return pairs

def token_id(word: str) -> int | None:
    """Return the single token ID for a word, or None if it tokenizes to multiple pieces."""
    ids = ENC.encode_ordinary(" " + word)   # leading space = word-initial context
    return ids[0] if len(ids) == 1 else None

def compute_pseudoword_cos(W: np.ndarray, pairs: list[tuple]) -> dict:
    """
    For each (pseudoword, comp1, comp2):
      - cos(pseudo, comp1), cos(pseudo, comp2)
      - cos(comp1, comp2)
    Averages over all pairs where all words are single tokens in vocab.
    """
    cos_pseudo_c1, cos_pseudo_c2, cos_c1_c2 = [], [], []
    n_found = 0

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for pseudo, c1, c2 in pairs:
        p_id  = token_id(pseudo)
        c1_id = token_id(c1)
        c2_id = token_id(c2) if c2 else None

        if p_id is None or c1_id is None: continue
        if p_id  >= W.shape[0]: continue
        if c1_id >= W.shape[0]: continue

        p_vec  = W[p_id]
        c1_vec = W[c1_id]
        cos_pseudo_c1.append(cos(p_vec, c1_vec))

        if c2_id is not None and c2_id < W.shape[0]:
            c2_vec = W[c2_id]
            cos_pseudo_c2.append(cos(p_vec, c2_vec))
            cos_c1_c2.append(cos(c1_vec, c2_vec))
        n_found += 1

    return {
        "n_pseudoword_pairs_found": n_found,
        "mean_cos_pseudo_c1":  float(np.mean(cos_pseudo_c1))  if cos_pseudo_c1  else float("nan"),
        "mean_cos_pseudo_c2":  float(np.mean(cos_pseudo_c2))  if cos_pseudo_c2  else float("nan"),
        "mean_cos_c1_c2":      float(np.mean(cos_c1_c2))      if cos_c1_c2      else float("nan"),
    }

# ── Analyze one checkpoint ────────────────────────────────────────────────────
def analyze_checkpoint(ckpt_path: str, pairs: list[tuple] | None,
                       rand_model: GPT | None, base_wte: np.ndarray | None) -> dict:
    model = load_model(ckpt_path)
    W     = get_wte(model)

    metrics = compute_spectral_metrics(W)
    metrics["mean_pairwise_cos"] = compute_pairwise_cos(W)

    if pairs:
        metrics.update(compute_pseudoword_cos(W, pairs))

    # Random baseline (same arch, no training)
    if rand_model is not None:
        W_rand   = get_wte(rand_model)
        r_spec   = compute_spectral_metrics(W_rand)
        r_cos    = compute_pairwise_cos(W_rand)
        metrics["rand_effective_rank"]  = r_spec["effective_rank"]
        metrics["rand_participation"]   = r_spec["participation"]
        metrics["rand_pairwise_cos"]    = r_cos
        for k in K_LIST:
            metrics[f"rand_expl_var_{k}"] = r_spec[f"expl_var_{k}"]

    # Base model delta
    if base_wte is not None:
        b_spec  = compute_spectral_metrics(base_wte)
        b_cos   = compute_pairwise_cos(base_wte)
        metrics["base_effective_rank"]  = b_spec["effective_rank"]
        metrics["base_participation"]   = b_spec["participation"]
        metrics["base_pairwise_cos"]    = b_cos
        for k in K_LIST:
            metrics[f"base_expl_var_{k}"] = b_spec[f"expl_var_{k}"]

    return metrics

# ── Write results ─────────────────────────────────────────────────────────────
def write_results(out_path: str, model_name: str,
                  per_ckpt: list[dict], avg: dict) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return "     N/A"
        return f"{v:8.4f}"

    with open(out_path, "w") as f:
        f.write(f"Embedding analysis — {model_name}\n")
        f.write(f"Matrix: wte (token embeddings)  shape: (vocab_size × n_embd)\n")
        f.write(f"Baselines: random init + base model checkpoint\n\n")

        for i, m in enumerate(per_ckpt):
            f.write(f"── {CHECKPOINT_NAMES[i]} ──\n")
            f.write(f"  effective_rank:       {fmt(m.get('effective_rank'))}"
                    f"  (rand: {fmt(m.get('rand_effective_rank'))},"
                    f"  base: {fmt(m.get('base_effective_rank'))})\n")
            f.write(f"  participation_ratio:  {fmt(m.get('participation'))}"
                    f"  (rand: {fmt(m.get('rand_participation'))},"
                    f"  base: {fmt(m.get('base_participation'))})\n")
            f.write(f"  mean_pairwise_cos:    {fmt(m.get('mean_pairwise_cos'))}"
                    f"  (rand: {fmt(m.get('rand_pairwise_cos'))},"
                    f"  base: {fmt(m.get('base_pairwise_cos'))})\n")
            f.write(f"  mean_norm:            {fmt(m.get('mean_norm'))}\n")
            f.write(f"  explained variance:\n")
            for k in K_LIST:
                f.write(f"    top-{k:>3}:  {fmt(m.get(f'expl_var_{k}'))}"
                        f"  (rand: {fmt(m.get(f'rand_expl_var_{k}'))},"
                        f"  base: {fmt(m.get(f'base_expl_var_{k}'))})\n")
            if "n_pseudoword_pairs_found" in m:
                f.write(f"  pseudoword pairs found: {m['n_pseudoword_pairs_found']}\n")
                f.write(f"  mean cos(pseudo, comp1): {fmt(m.get('mean_cos_pseudo_c1'))}\n")
                f.write(f"  mean cos(pseudo, comp2): {fmt(m.get('mean_cos_pseudo_c2'))}\n")
                f.write(f"  mean cos(comp1,  comp2): {fmt(m.get('mean_cos_c1_c2'))}\n")
            f.write("\n")

        f.write("── Average over all checkpoints ──\n")
        for key in ["effective_rank", "participation", "mean_pairwise_cos",
                    "mean_norm"] + [f"expl_var_{k}" for k in K_LIST]:
            base_key = f"base_{key}" if f"base_{key}" in avg else None
            rand_key = f"rand_{key}" if f"rand_{key}" in avg else None
            base_s = fmt(avg.get(base_key)) if base_key else "     N/A"
            rand_s = fmt(avg.get(rand_key)) if rand_key else "     N/A"
            f.write(f"  {key:<25} {fmt(avg.get(key))}  (rand: {rand_s},  base: {base_s})\n")
        if "mean_cos_pseudo_c1" in avg:
            f.write(f"\n  mean cos(pseudo, comp1): {fmt(avg.get('mean_cos_pseudo_c1'))}\n")
            f.write(f"  mean cos(pseudo, comp2): {fmt(avg.get('mean_cos_pseudo_c2'))}\n")
            f.write(f"  mean cos(comp1,  comp2): {fmt(avg.get('mean_cos_c1_c2'))}\n")

# ── Append to summary CSV ─────────────────────────────────────────────────────
CSV_FIELDS = (
    ["model", "corpus", "n", "kind", "checkpoint"] +
    ["effective_rank", "participation", "mean_pairwise_cos", "mean_norm"] +
    [f"expl_var_{k}" for k in K_LIST] +
    ["rand_effective_rank", "rand_participation", "rand_pairwise_cos"] +
    ["base_effective_rank", "base_participation", "base_pairwise_cos"] +
    ["n_pseudoword_pairs_found",
     "mean_cos_pseudo_c1", "mean_cos_pseudo_c2", "mean_cos_c1_c2"]
)

def append_csv(model_name: str, corpus: str, n_or_base, kind: str,
               per_ckpt: list[dict]) -> None:
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header: w.writeheader()
        for i, m in enumerate(per_ckpt):
            row = {"model": model_name, "corpus": corpus,
                   "n": n_or_base, "kind": kind,
                   "checkpoint": CHECKPOINT_NAMES[i]}
            row.update(m)
            w.writerow(row)

# ── Main evaluation function ──────────────────────────────────────────────────
def evaluate_model(corpus: str, n_or_base, kind: str, model_name: str,
                   base_wte_cache: dict) -> None:
    is_base   = n_or_base == "base"
    model_dir = get_gpt_model_dir(corpus, str(n_or_base), kind)

    # Load pseudoword pairs if applicable
    pairs = None
    if not is_base:
        map_path  = get_pseudoword_map(corpus, n_or_base, kind)
        base_kind = kind.replace("_HALF", "")
        inf_path  = get_inflection_map(corpus, base_kind)
        if os.path.exists(map_path) and os.path.exists(inf_path):
            pairs = load_pseudoword_pairs(map_path, inf_path)

    # Get base model wte (cached)
    base_wte = base_wte_cache.get(corpus)

    per_ckpt   = []
    rand_model = None  # lazily created from first checkpoint

    for i, ckpt_name in enumerate(CHECKPOINT_NAMES):
        ckpt_path = os.path.join(model_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  {ckpt_name} not found — skipping")
            continue
        if rand_model is None:
            print(f"  Building random baseline …")
            rand_model = make_random_model(ckpt_path)
        print(f"  Analyzing {ckpt_name} …")
        m = analyze_checkpoint(ckpt_path, pairs, rand_model, base_wte)
        per_ckpt.append(m)

    if not per_ckpt:
        print("  No checkpoints found — skipping")
        return

    # Average over checkpoints
    avg = {}
    all_keys = set().union(*[m.keys() for m in per_ckpt])
    for key in all_keys:
        vals = [m[key] for m in per_ckpt
                if key in m and m[key] is not None
                and not (isinstance(m[key], float) and math.isnan(m[key]))]
        avg[key] = float(np.mean(vals)) if vals else float("nan")

    out_dir  = os.path.join(EMBED_DIR, model_name)
    out_path = os.path.join(out_dir, "embedding_analysis.txt")
    write_results(out_path, model_name, per_ckpt, avg)
    append_csv(model_name, corpus, n_or_base, kind, per_ckpt)
    print(f"  ✅ Results → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n",     type=int)
    parser.add_argument("--kind",  choices=KINDS)
    parser.add_argument("--base",  action="store_true")
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    if args.all:
        tasks = [(c, "base", None) for c in CORPORA] + [
            (c, n, k) for c in CORPORA for n in COUNTS for k in KINDS
        ]
    elif args.base:
        tasks = [(c, "base", None) for c in CORPORA]
    elif args.corpus and args.n and args.kind:
        tasks = [(args.corpus, args.n, args.kind)]
    else:
        parser.print_help()
        sys.exit(0)

    # Pre-load base model wte for both corpora (used as baseline in all conditions)
    print("Pre-loading base model embeddings …")
    base_wte_cache = {}
    for corpus in CORPORA:
        base_dir = get_gpt_model_dir(corpus, "base", None)
        # Use ckpt1 as the representative base embedding
        ckpt1 = os.path.join(base_dir, CHECKPOINT_NAMES[0])
        if os.path.exists(ckpt1):
            base_wte_cache[corpus] = get_wte(load_model(ckpt1))
            print(f"  ✓ {corpus} base wte loaded  shape={base_wte_cache[corpus].shape}")
        else:
            print(f"  ⚠️  {corpus} base ckpt1 not found — base baseline unavailable")

    for corpus, n_or_base, kind in tasks:
        is_base  = n_or_base == "base"
        name = (f"gpt_{corpus}_base" if is_base
                else f"gpt_{corpus}_{n_or_base}_{kind}")
        print(f"\n══ {name} ══════════════════════════════════════════")
        evaluate_model(corpus, n_or_base, kind, name, base_wte_cache)

    print(f"\n🎉 Embedding analysis complete.")
    print(f"   Summary CSV → {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
