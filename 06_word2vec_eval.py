"""
06_word2vec_eval.py
===================

Step 6 of the pipeline. Evaluates Word2Vec models by computing neighborhood
distance metrics for each target word in the Beekhuizen word list.

For each model, computes for K ∈ {20, 100, 200}:
  • nbhd_dist_K      — mean cosine distance from target to its K nearest neighbours
  • nbhd_pairdist_K  — mean pairwise cosine distance among those K neighbours

Target words include the Beekhuizen list (homonyms, polysemes, monosemes) plus,
for non-base models, the artificial pseudowords for that condition.

Usage:
    # Evaluate all models
    python 06_word2vec_eval.py

    # Evaluate one model
    python 06_word2vec_eval.py --corpus recipe --n 10 --kind HOM

    # Evaluate base models only
    python 06_word2vec_eval.py --base

Output per model:
    evaluation/nbhd/<model_name>_nbhd_results/
        <model_name>_nbhd_results.csv
        <model_name>_ambiguity_type_counts.csv
        grouped_averages_<model_name>.csv
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import pairwise_distances
import gensim

from config import (
    CORPORA, COUNTS, KINDS, BEEKHUIZEN_WORDS,
    NBHD_DIR, get_word2vec_model_dir, get_pseudoword_map,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
K_LIST       = [20, 100, 200]
INCLUDE_SELF = True

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_w2v(model_dir: str):
    path = os.path.join(model_dir, "word2vec.bin")
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

def load_target_words(kind_group: str, corpus: str, n) -> pd.DataFrame:
    """
    Load Beekhuizen word list and augment with artificial pseudowords for
    non-base conditions.
    """
    df = pd.read_csv(BEEKHUIZEN_WORDS)
    if kind_group == "BASE":
        return df

    base_kind = kind_group.replace("_HALF", "")
    map_path  = get_pseudoword_map(corpus, n, kind_group)
    if not os.path.exists(map_path):
        print(f"  ⚠️  Map not found: {map_path}")
        return df

    with open(map_path) as f:
        repl_map = json.load(f)

    pseudowords = list(repl_map.values())
    label       = "HOM*" if "HOM" in kind_group else "HYP*"
    extra = pd.DataFrame({"word": pseudowords, "ambiguity_type": label})
    return pd.concat([df, extra], ignore_index=True)

def get_neighbor_indices(word, model, k) -> np.ndarray:
    idx        = model.key_to_index[word]
    target_vec = model.vectors[idx][None, :]
    dists      = pairwise_distances(target_vec, model.vectors, metric="cosine")[0]
    order      = np.argsort(dists)
    if INCLUDE_SELF:
        return order[:k]
    order = order[order != idx]
    return order[:k]

def compute_nbhd_dist(word, neigh_idx, model) -> float:
    idx        = model.key_to_index[word]
    target_vec = model.vectors[idx][None, :]
    neigh_vecs = model.vectors[neigh_idx]
    return pairwise_distances(target_vec, neigh_vecs, metric="cosine")[0].mean()

def compute_nbhd_pairdist(neigh_idx, model) -> float:
    neigh_vecs = model.vectors[neigh_idx]
    return pairwise_distances(neigh_vecs, metric="cosine").mean()

def process_word(word, model):
    if word not in model.key_to_index:
        return word, {}
    metrics = {}
    for k in K_LIST:
        idx = get_neighbor_indices(word, model, k)
        metrics[f"nbhd_dist_{k}"]     = compute_nbhd_dist(word, idx, model)
        metrics[f"nbhd_pairdist_{k}"] = compute_nbhd_pairdist(idx, model)
    return word, metrics

def evaluate_model(corpus: str, n_or_base, kind: str, model_name: str) -> None:
    kind_group = "BASE" if n_or_base == "base" else kind.upper()
    model_dir  = get_word2vec_model_dir(corpus, str(n_or_base), kind)
    out_dir    = os.path.join(NBHD_DIR, f"{model_name}_nbhd_results")
    os.makedirs(out_dir, exist_ok=True)

    bin_path = os.path.join(model_dir, "word2vec.bin")
    if not os.path.exists(bin_path):
        print(f"  ⚠️  Model not found: {bin_path} — skipping")
        return

    print(f"  Loading model …")
    model  = load_w2v(model_dir)
    target_df = load_target_words(kind_group, corpus,
                                  n_or_base if n_or_base != "base" else None)
    words  = target_df["word"].tolist()
    print(f"  {len(words)} target words  |  vocab: {len(model.key_to_index):,}")

    results = []
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(process_word, w, model): w for w in words}
        for fut in tqdm(futures, desc="  Computing", leave=False):
            w, m = fut.result()
            if m:
                results.append((w, m))

    # Build DataFrame
    rows = []
    for word, metrics in results:
        row_df = target_df[target_df["word"] == word]
        amb    = row_df["ambiguity_type"].values[0] if not row_df.empty else "UNKNOWN"
        row    = {"word": word, "ambiguity_type": amb, "model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)

    results_path = os.path.join(out_dir, f"{model_name}_nbhd_results.csv")
    df.to_csv(results_path, index=False)

    # Ambiguity counts
    counts = df["ambiguity_type"].value_counts(dropna=False).reset_index()
    counts.columns = ["ambiguity_type", "count"]
    counts.to_csv(os.path.join(out_dir, f"{model_name}_ambiguity_type_counts.csv"), index=False)

    # Grouped averages
    metric_cols = [c for c in df.columns if c.startswith("nbhd_")]
    grouped = df.groupby("ambiguity_type")[metric_cols].mean().reset_index()
    grouped.to_csv(os.path.join(out_dir, f"grouped_averages_{model_name}.csv"), index=False)

    print(f"  ✅ Results saved to {out_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n", type=int)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--base", action="store_true", help="Evaluate base models only")
    parser.add_argument("--all",  action="store_true", help="Evaluate all models")
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
        tasks = [(c, "base", None) for c in CORPORA]

    for corpus, n_or_base, kind in tasks:
        if n_or_base == "base":
            name = f"word2vec_{corpus}_base"
        else:
            name = f"word2vec_{corpus}_{n_or_base}_{kind}"
        print(f"\n── {name} ─────────────────────────────────────────────")
        evaluate_model(corpus, n_or_base, kind, name)

    print("\n🎉 Neighbourhood evaluation complete.")

if __name__ == "__main__":
    main()
