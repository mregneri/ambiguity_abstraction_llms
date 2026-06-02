"""
analyze_nbhd_distances.py
==========================

This script computes two neighborhood metrics for each target word in a
specified Word2Vec model, for K ∈ {20, 100, 200}:

1) nbhd_dist_K
   Mean cosine distance from the target word to its K nearest neighbors.
   Neighbors are selected by computing cosine distances from the target to the
   entire vocabulary and taking the K smallest distances (including the
   target itself at distance 0, by default).

2) nbhd_pairdist_K
   Mean pairwise cosine distance **among** those K neighbor vectors.

The target word list can be augmented with artificial homonyms/hypernyms from
mapping files when evaluating HOM/HOM_HALF and HYP/HYP_HALF models.

Usage
-----
    python analyze_nbhd_distances.py <model_name>

Examples
--------
    python analyze_nbhd_distances.py word2vec_recipe_10_HOM
    python analyze_nbhd_distances.py word2vec_recipe_30_HYP_HALF
    python analyze_nbhd_distances.py word2vec_recipe_base

Expected Directory Structure
----------------------------
- Word2Vec model:
    /home/nina/ambiguity/trained_models/allkinds/<model_name>/word2vec.bin

- Wordlist CSV:
    ../data/beekhuizen_all_words.csv

- Homonym maps (required for HOM / HOM_HALF):
    /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/<map_name>_map.json

- Hypernym maps (required for HYP / HYP_HALF):
    /home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/hypernym_maps/<map_name>_map.json

Where
-----
<model_name> looks like:
    word2vec_<base>_<count>_<KIND>      # e.g., word2vec_recipe_10_HOM
    word2vec_<base>_base                # e.g., word2vec_recipe_base

And
---
<map_name> is <model_name> without the 'word2vec_' prefix.

Outputs
-------
- /home/nina/ambiguity/evaluation/neighborhoods/allkinds/<model_name>_nbhd_results/
    Per-word results including:
      word, ambiguity_type, model,
      nbhd_dist_20, nbhd_pairdist_20,
      nbhd_dist_100, nbhd_pairdist_100,
      nbhd_dist_200, nbhd_pairdist_200

- <model_name>_nbhd_results/<model_name>_ambiguity_type_counts.csv
    Counts of rows per ambiguity_type.

- <model_name>_nbhd_results/grouped_averages_<model_name>.csv
    Mean of all neighborhood metrics grouped by ambiguity_type.

Notes
-----
- Neighbor selection mirrors the legacy implementation: cosine distances are
  computed against the **entire vocabulary**, and the target token itself is
  included among the top-K (distance 0). Set INCLUDE_SELF = False to
  exclude it.
"""

import sys
import os
import json
import gensim
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re

# ---- Config ----
K_LIST = [20, 100, 200]     # match old script sweep
INCLUDE_SELF = True         # match old script behavior (self at rank 0 is included)
TARGET_WORDS_FILE = "../data/beekhuizen_all_words.csv"
BASE_MODEL_DIR = "/home/nina/ambiguity/trained_models/allkinds"
HOMONYM_MAP_DIR = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/"
HYPERNYM_MAP_DIR = "/home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/hypernym_maps/"
NBHD_OUTPUT_BASE = "/home/nina/ambiguity/experiments/nbhd_analysis/allkinds"

# ---- Parse model name ----
if len(sys.argv) < 2:
    print("Usage: python analyze_nbhd_distances.py <model_name>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]  # e.g., word2vec_recipe_10_HOM
MAP_NAME = MODEL_NAME.removeprefix("word2vec_")
MODEL_PATH = os.path.join(BASE_MODEL_DIR, MODEL_NAME, "word2vec.bin")
OUTPUT_DIR = os.path.join(NBHD_OUTPUT_BASE, f"{MODEL_NAME}_nbhd_results")
RESULTS_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_nbhd_results.csv")

def get_neighbor_indices(word: str, model, include_self: bool = INCLUDE_SELF, k: int = 100) -> np.ndarray:
    """
       Return the indices of the K nearest neighbors by smallest cosine distance,
       computed against the entire vocabulary.

       This reproduces the legacy behavior:
       - Distances are computed from the target to every vocab vector.
       - Results are sorted ascending (0 for the target itself).
       - If include_self is True, the target's own index is included among the K.

       Parameters
       ----------
       word : str
           Target word (must be in model.key_to_index).
       model : gensim.models.KeyedVectors
           Loaded Word2Vec keyed vectors.
       include_self : bool
           Whether to include the target's own index (distance 0) in the top-K.
       k : int
           Number of items to return.

       Returns
       -------
       np.ndarray
           Array of integer indices into model.vectors of length K.
       """
    idx = model.key_to_index[word]
    target_vec = model.vectors[idx][None, :]  # shape (1, d)
    # cosine distances to *all* vocab vectors
    dists = pairwise_distances(target_vec, model.vectors, metric="cosine")[0]  # shape (V,)
    order = np.argsort(dists)  # ascending: self at position 0
    if include_self:
        return order[:k]
    else:
        # skip self (which should be first), then take next k
        order_wo_self = order[order != idx]
        return order_wo_self[:k]


def compute_nbhd_dist(word: str, neigh_idx: np.ndarray, model) -> float:
    """
    Compute nbhd_dist_K: mean cosine distance from the target word to its K neighbors.

    Parameters
    ----------
    word : str
        Target word present in the model vocabulary.
    neigh_idx : np.ndarray
        Indices of the K neighbor rows in model.vectors (possibly including the target).
    model : gensim.models.KeyedVectors
        Loaded Word2Vec keyed vectors.

    Returns
    -------
    float
        Mean cosine distance from target to each neighbor.
    """
    idx = model.key_to_index[word]
    target_vec = model.vectors[idx][None, :]
    neigh_vecs = model.vectors[neigh_idx]
    # pairwise_distances returns shape (1, K); take [0] then mean
    return pairwise_distances(target_vec, neigh_vecs, metric="cosine")[0].mean()


def compute_nbhd_pairdist(neigh_idx: np.ndarray, model) -> float:
    """
    Compute nbhd_pairdist_K: mean pairwise cosine distance among the K neighbor vectors.

    Parameters
    ----------
    neigh_idx : np.ndarray
        Indices of the K neighbor rows in model.vectors.
    model : gensim.models.KeyedVectors
        Loaded Word2Vec keyed vectors.

    Returns
    -------
    float
        Mean of the full K×K cosine-distance matrix among neighbors.
    """
    neigh_vecs = model.vectors[neigh_idx]
    pdm = pairwise_distances(neigh_vecs, metric="cosine")  # shape (K, K)
    return pdm.mean()

def infer_kind_group(model_name: str) -> str:
    m = model_name.lower()
    if m.endswith("_base"):
        return "BASE"
    if m.endswith("_hom_half"):
        return "HOM_HALF"
    if m.endswith("_hyp_half"):
        return "HYP_HALF"
    if m.endswith("_hom"):
        return "HOM"
    if m.endswith("_hyp"):
        return "HYP"
    return "BASE"

KIND_GROUP = infer_kind_group(MODEL_NAME)

def map_basename_for_lookup(map_name: str) -> str:
    """
    Normalize HALF variants to their full counterparts for map lookup.
    e.g., recipe_40_HYP_HALF -> recipe_40_HYP
          tiny_20_HOM_HALF  -> tiny_20_HOM
    """
    base = re.sub(r'_HOM_HALF$', '_HOM', map_name, flags=re.IGNORECASE)
    base = re.sub(r'_HYP_HALF$', '_HYP', base, flags=re.IGNORECASE)
    return base

MAP_NAME_FOR_LOOKUP = map_basename_for_lookup(MAP_NAME)

HOMONYM_MAP_PATH = os.path.join(HOMONYM_MAP_DIR, f"{MAP_NAME_FOR_LOOKUP}_map.json")
HYPERNYM_MAP_PATH = os.path.join(HYPERNYM_MAP_DIR, f"{MAP_NAME_FOR_LOOKUP}_map.json")

# Log resolved paths for transparency
print(f"Resolved map name for lookup: {MAP_NAME_FOR_LOOKUP}")
print(f"→ Homonym map path:  {HOMONYM_MAP_PATH}")
print(f"→ Hypernym map path: {HYPERNYM_MAP_PATH}")

# ---- Load Word2Vec model ----
def load_word2vec_model(path):
    if not os.path.isfile(path):
        print(f"❌ Model file not found: {path}")
        sys.exit(1)
    print(f"Loading Word2Vec model from: {path}")
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    print("Model loaded successfully.")
    return model

# ---- Load and (optionally) extend target words ----
def load_target_words(base_file, kind_group: str) -> pd.DataFrame:
    if not os.path.isfile(base_file):
        print(f"❌ Target words file not found: {base_file}")
        sys.exit(1)

    print(f"Loading base target words from: {base_file}")
    df = pd.read_csv(base_file, names=["word", "ambiguity_type"])

    # BASE models: never augment
    if kind_group == "BASE":
        print("Model kind: BASE → no augmentation (using base word list only).")
        return df

    # HOM / HOM_HALF: require homonym map
    if kind_group in {"HOM", "HOM_HALF"}:
        if not os.path.exists(HOMONYM_MAP_PATH):
            print(f"❌ Required homonym map missing for {MODEL_NAME}: {HOMONYM_MAP_PATH}")
            print("This model is HOM/HOM_HALF; skipping as requested.")
            sys.exit(1)
        print(f"Adding artificial homonyms from: {HOMONYM_MAP_PATH}")
        with open(HOMONYM_MAP_PATH, "r") as f:
            homonym_map = json.load(f)
        artificial_words = list(homonym_map.values())
        artificial_df = pd.DataFrame({"word": artificial_words, "ambiguity_type": "HOM*"})
        df = pd.concat([df, artificial_df], ignore_index=True)
        return df

    # HYP / HYP_HALF: require hypernym map (changed to required)
    if kind_group in {"HYP", "HYP_HALF"}:
        if not os.path.exists(HYPERNYM_MAP_PATH):
            print(f"❌ Required hypernym map missing for {MODEL_NAME}: {HYPERNYM_MAP_PATH}")
            print("This model is HYP/HYP_HALF; skipping as requested.")
            sys.exit(1)
        print(f"Adding artificial hypernyms from: {HYPERNYM_MAP_PATH}")
        with open(HYPERNYM_MAP_PATH, "r") as f:
            hypernym_map = json.load(f)
        artificial_words = list(hypernym_map.values())
        artificial_df = pd.DataFrame({"word": artificial_words, "ambiguity_type": "HYP*"})
        df = pd.concat([df, artificial_df], ignore_index=True)
        return df

    # Fallback (shouldn't occur)
    print("Unrecognized model kind; proceeding with base word list.")
    return df

# ---- Compute neighborhood mean distance and neighborhood mean pairwise distance ----
def process_word_all_k(word: str, model, k_list: list[int]) -> tuple[str, dict]:
    """
    For a single target word, compute both neighborhood metrics for each K in k_list:

      - nbhd_dist_K: mean cosine distance target → K neighbors
      - nbhd_pairdist_K: mean pairwise cosine distance among those neighbors

    Parameters
    ----------
    word : str
        Target word to evaluate (skipped if OOV).
    model : gensim.models.KeyedVectors
        Loaded Word2Vec keyed vectors.
    k_list : list[int]
        List of K values to evaluate (e.g., [20, 100, 200]).

    Returns
    -------
    (str, dict)
        word, and a dict mapping metric names (e.g., "nbhd_dist_100") to floats.
        Returns an empty dict if the word is OOV.
    """
    if word not in model.key_to_index:
        return word, {}

    metrics = {}
    for k in k_list:
        neigh_idx = get_neighbor_indices(word, model, include_self=INCLUDE_SELF, k=k)
        # target→neighbors mean distance
        metrics[f"nbhd_dist_{k}"] = compute_nbhd_dist(word, neigh_idx, model)
        # neighbors↔neighbors mean pairwise distance
        metrics[f"nbhd_pairdist_{k}"] = compute_nbhd_pairdist(neigh_idx, model)
    return word, metrics


def process_all_words_parallel(words: list[str], model, k_list: list[int]) -> list[tuple[str, dict]]:
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_word_all_k, word, model, k_list): word for word in words}
        for future in tqdm(futures, desc="Processing words"):
            word, metrics = future.result()
            if metrics:
                results.append((word, metrics))
    return results

# ---- Save output ----
def save_results(results, target_df):
    """
    Save per-word metrics, counts by ambiguity type, and grouped averages.

    Parameters
    ----------
    results : list[tuple[str, dict]]
        Output from process_all_words_parallel: (word, metrics_dict) pairs.
    target_df : pandas.DataFrame
        DataFrame of target words with their ambiguity_type.

    Writes
    ------
    - RESULTS_PATH: CSV with per-word metrics and metadata.
    - <model>_ambiguity_type_counts.csv: Counts per ambiguity_type.
    - grouped_averages_<model>.csv: Mean of each nbhd_* column grouped by ambiguity_type.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build per-word records with all requested columns
    rows = []
    for word, metrics in results:
        row = target_df.loc[target_df["word"] == word]
        ambiguity_type = row["ambiguity_type"].values[0] if not row.empty else "UNKNOWN"
        base = {
            "word": word,
            "ambiguity_type": ambiguity_type,
            "model": MODEL_NAME,
        }
        base.update(metrics)  # nbhd_dist_20/100/200, nbhd_pairdist_20/100/200
        rows.append(base)

    df = pd.DataFrame(rows)

    # Ensure metric columns exist even if some words are missing
    for k in K_LIST:
        for col in (f"nbhd_dist_{k}", f"nbhd_pairdist_{k}"):
            if col not in df.columns:
                df[col] = np.nan

    df.to_csv(RESULTS_PATH, index=False)
    print(f"Results saved to: {RESULTS_PATH}")

    # Ambiguity counts
    count_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_ambiguity_type_counts.csv")
    ambiguity_counts = df["ambiguity_type"].value_counts(dropna=False).reset_index()
    ambiguity_counts.columns = ["ambiguity_type", "count"]
    ambiguity_counts.to_csv(count_path, index=False)
    print(f"Ambiguity type counts saved to: {count_path}")

    # Grouped averages (wide): mean of each metric per ambiguity_type
    avg_path = os.path.join(OUTPUT_DIR, f"grouped_averages_{MODEL_NAME}.csv")
    metric_cols = [c for c in df.columns if c.startswith("nbhd_dist_") or c.startswith("nbhd_pairdist_")]
    grouped = df.groupby("ambiguity_type")[metric_cols].mean().reset_index()
    grouped.to_csv(avg_path, index=False)
    print(f"Grouped averages saved to: {avg_path}")

# ---- Run all steps ----
def main():
    print(f"Model: {MODEL_NAME}")
    print(f"Kind group detected: {KIND_GROUP}")

    target_df = load_target_words(TARGET_WORDS_FILE, KIND_GROUP)
    target_words = target_df["word"].tolist()

    added_hom = (target_df["ambiguity_type"] == "HOM*").sum()
    added_hyp = (target_df["ambiguity_type"] == "HYP*").sum()
    if added_hom > 0:
        print(f"Added {added_hom} artificial homonyms (labelled as 'HOM*').")
    if added_hyp > 0:
        print(f"Added {added_hyp} artificial hypernyms (labelled as 'HYP*').")

    model = load_word2vec_model(MODEL_PATH)

    print(f"Computing neighborhood metrics for {MODEL_NAME} (K_LIST={K_LIST}, include_self={INCLUDE_SELF})")
    results = process_all_words_parallel(target_words, model, K_LIST)

    save_results(results, target_df)

if __name__ == "__main__":
    main()