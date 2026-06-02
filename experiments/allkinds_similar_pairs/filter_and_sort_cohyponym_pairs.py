"""
filter_and_sort_cohyponym_pairs.py
==================================

This script processes word pairs from the relation-annotated hypernym files and
extracts only those pairs classified as **Co-Hyponyms**. It performs all necessary
cleaning, reordering, and sorting steps to prepare these pairs for downstream work
on creating artificial hypernym names.

For each corpus (RecipeNLG, TinyStories), the script:

1. Loads the annotated hypernym pair file produced by earlier analysis.
2. Filters the data to retain only pairs whose `relation` is exactly `"Co-Hyponym"`.
3. Reorders each retained pair so that the **more frequent word** (based on token
   frequency in the Word2Vec model) always appears as `word1`.
4. Sorts all reordered co-hyponym pairs in **descending order of cosine similarity**,
   ensuring that the most semantically similar pairs appear first.
5. Writes a full, sorted output file that includes all remaining fields
   (including `relation`), preserving the original information.
6. Creates a second output file containing only the **top N** most similar
   co-hyponym pairs:
   - Drops the `relation` column.
   - Adds an empty `name` column to support later manual construction of
     artificial hypernym names.
   - Writes these top pairs in the same similarity-sorted order.

Inputs:
-------
- hypernym_pairs_recipe_with_relations.csv
- hypernym_pairs_tiny_with_relations.csv

Outputs:
--------
TinyStories:
  - tiny_cohyponym_pairs_sorted.csv
  - tiny_cohyponym_pairs_top50_for_naming.csv

RecipeNLG:
  - recipe_cohyponym_pairs_sorted.csv
  - recipe_cohyponym_pairs_top50_for_naming.csv

Usage:
------
$ python filter_and_sort_cohyponym_pairs.py
"""

import csv

# ==================== CONFIGURATION ====================

TOP_N = 50  # Number of most similar co-hyponym pairs to keep for naming

RECIPE_INPUT = "hypernym_pairs_recipe_with_relations.csv"
TINY_INPUT = "hypernym_pairs_tiny_with_relations.csv"

RECIPE_FULL_OUTPUT = "recipe_cohyponym_pairs_sorted.csv"
TINY_FULL_OUTPUT = "tiny_cohyponym_pairs_sorted.csv"

RECIPE_TOP_OUTPUT = "recipe_cohyponym_pairs_top50_for_naming.csv"
TINY_TOP_OUTPUT = "tiny_cohyponym_pairs_top50_for_naming.csv"

DATASETS = [
    {
        "name": "RecipeNLG",
        "input": RECIPE_INPUT,
        "full_output": RECIPE_FULL_OUTPUT,
        "top_output": RECIPE_TOP_OUTPUT,
    },
    {
        "name": "TinyStories",
        "input": TINY_INPUT,
        "full_output": TINY_FULL_OUTPUT,
        "top_output": TINY_TOP_OUTPUT,
    },
]

# ==================== HELPERS ====================

def load_pairs(csv_path):
    """
    Load rows from a CSV file into a list of dictionaries.

    Expects columns:
        word1, word2, cosine_similarity,
        word1_freq, word1_docfreq, word2_freq, word2_docfreq, relation
    """
    pairs = []
    print(f"Loading pairs from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields to appropriate types
            try:
                row["cosine_similarity"] = float(row["cosine_similarity"])
            except (KeyError, ValueError):
                continue  # skip malformed rows

            for key in ("word1_freq", "word1_docfreq", "word2_freq", "word2_docfreq"):
                try:
                    row[key] = int(row[key])
                except (KeyError, ValueError):
                    row[key] = 0

            pairs.append(row)
    print(f"  Loaded {len(pairs)} rows.")
    return pairs


def filter_co_hyponyms(pairs):
    """Return only those pairs whose relation is exactly 'Co-Hyponym'."""
    filtered = [row for row in pairs if row.get("relation") == "Co-Hyponym"]
    print(f"  Filtered to {len(filtered)} Co-Hyponym pairs.")
    return filtered


def reorder_pair(row):
    """
    Ensure that the more frequent word (by token frequency) is in word1.

    If word1_freq < word2_freq, swaps:
      - word1 <-> word2
      - word1_freq <-> word2_freq
      - word1_docfreq <-> word2_docfreq

    cosine_similarity and relation remain unchanged.
    """
    if row["word1_freq"] < row["word2_freq"]:
        # Swap the words
        row["word1"], row["word2"] = row["word2"], row["word1"]
        # Swap their frequencies
        row["word1_freq"], row["word2_freq"] = row["word2_freq"], row["word1_freq"]
        # Swap their document frequencies
        row["word1_docfreq"], row["word2_docfreq"] = row["word2_docfreq"], row["word1_docfreq"]
    return row


def reorder_all_pairs(pairs):
    """Apply reorder_pair to all pairs."""
    return [reorder_pair(row) for row in pairs]


def sort_by_similarity(pairs):
    """Return a new list of pairs sorted by cosine_similarity (descending)."""
    return sorted(pairs, key=lambda row: row["cosine_similarity"], reverse=True)


def write_full_output(pairs, output_path):
    """
    Write all co-hyponym pairs (including 'relation') to a CSV file.

    Fields:
      word1, word2, cosine_similarity,
      word1_freq, word1_docfreq, word2_freq, word2_docfreq, relation
    """
    print(f"  Writing full co-hyponym list to {output_path}...")
    fieldnames = [
        "word1", "word2", "cosine_similarity",
        "word1_freq", "word1_docfreq",
        "word2_freq", "word2_docfreq",
        "relation",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow({
                "word1": row["word1"],
                "word2": row["word2"],
                "cosine_similarity": row["cosine_similarity"],
                "word1_freq": row["word1_freq"],
                "word1_docfreq": row["word1_docfreq"],
                "word2_freq": row["word2_freq"],
                "word2_docfreq": row["word2_docfreq"],
                "relation": row["relation"],
            })


def write_top_for_naming(pairs, output_path, top_n):
    """
    Write the top `top_n` pairs (by cosine_similarity) for manual naming.

    Drops the 'relation' column and adds a 'name' column at the end (empty for now).

    Fields:
      word1, word2, cosine_similarity,
      word1_freq, word1_docfreq, word2_freq, word2_docfreq, name
    """
    top_pairs = pairs[:top_n]
    print(f"  Writing top {len(top_pairs)} pairs for naming to {output_path}...")

    fieldnames = [
        "word1", "word2", "cosine_similarity",
        "word1_freq", "word1_docfreq",
        "word2_freq", "word2_docfreq",
        "name",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_pairs:
            writer.writerow({
                "name": "",
                "word1": row["word1"],
                "word2": row["word2"],
                "cosine_similarity": row["cosine_similarity"],
                "word1_freq": row["word1_freq"],
                "word1_docfreq": row["word1_docfreq"],
                "word2_freq": row["word2_freq"],
                "word2_docfreq": row["word2_docfreq"],
                "name": "",
            })


def process_dataset(config):
    """
    Full processing pipeline for one dataset (RecipeNLG or TinyStories):

    1. Load pairs from input CSV.
    2. Filter for Co-Hyponym relations.
    3. Reorder pairs so the more frequent word is in word1.
    4. Sort by cosine_similarity (descending).
    5. Write full sorted list (with relation).
    6. Write top-N list for naming (without relation, with empty 'name').
    """
    print(f"\nProcessing dataset: {config['name']}")
    pairs = load_pairs(config["input"])

    cohypo_pairs = filter_co_hyponyms(pairs)
    if not cohypo_pairs:
        print("  No Co-Hyponym pairs found; skipping outputs.")
        return

    reordered = reorder_all_pairs(cohypo_pairs)
    sorted_pairs = sort_by_similarity(reordered)

    write_full_output(sorted_pairs, config["full_output"])
    write_top_for_naming(sorted_pairs, config["top_output"], TOP_N)

    print(f"  Done processing {config['name']}. "
          f"{len(sorted_pairs)} co-hyponym pairs in full file, "
          f"{min(TOP_N, len(sorted_pairs))} in top-N file.")


def main():
    for cfg in DATASETS:
        process_dataset(cfg)


if __name__ == "__main__":
    main()