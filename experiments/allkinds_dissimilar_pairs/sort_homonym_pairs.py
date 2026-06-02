"""
sort_homonym_pairs.py
======================

This script processes homonym candidate word pairs from the all-kinds homonym files
and prepares them for downstream work on creating artificial homonyms.

For each corpus (RecipeNLG, TinyStories), the script:

1. Loads the homonym pair file produced by earlier analysis
   (`allkinds_select_homonym_pairs.py`).
2. Reorders each pair so that the **more frequent word** (based on token
   frequency in the Word2Vec model) always appears as `word1`.
3. Sorts all reordered homonym pairs in **ascending order of cosine similarity**,
   so pairs with lower cosine similarity (more dissimilar) appear first.
4. Writes a full, sorted output file that includes all fields from the input.
5. Creates a second output file containing only the **top N** most dissimilar
   homonym pairs:
   - Adds an empty `name` column to support later manual construction of
     artificial homonym names.
   - Writes these top pairs in the same dissimilarity-sorted order.

Inputs:
-------
- homonym_pairs_recipe_allkinds.csv
- homonym_pairs_tiny_allkinds.csv

Outputs:
--------
TinyStories:
  - tiny_homonym_pairs_sorted.csv
  - tiny_homonym_pairs_top50_for_naming.csv

RecipeNLG:
  - recipe_homonym_pairs_sorted.csv
  - recipe_homonym_pairs_top50_for_naming.csv

Usage:
------
$ python sort_homonym_pairs.py
"""

import csv

# ==================== CONFIGURATION ====================

TOP_N = 50  # Number of most dissimilar homonym pairs to keep for naming

RECIPE_INPUT = "homonym_pairs_recipe_allkinds.csv"
TINY_INPUT = "homonym_pairs_tiny_allkinds.csv"

RECIPE_FULL_OUTPUT = "recipe_homonym_pairs_sorted.csv"
TINY_FULL_OUTPUT = "tiny_homonym_pairs_sorted.csv"

RECIPE_TOP_OUTPUT = "recipe_homonym_pairs_top50_for_naming.csv"
TINY_TOP_OUTPUT = "tiny_homonym_pairs_top50_for_naming.csv"

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
        word1_freq, word1_docfreq, word2_freq, word2_docfreq
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


def reorder_pair(row):
    """
    Ensure that the more frequent word (by token frequency) is in word1.

    If word1_freq < word2_freq, swaps:
      - word1 <-> word2
      - word1_freq <-> word2_freq
      - word1_docfreq <-> word2_docfreq

    cosine_similarity remains unchanged.
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
    """
    Return a new list of pairs sorted by cosine_similarity (ascending).

    Lowest cosine_similarity = most dissimilar.
    """
    return sorted(pairs, key=lambda row: row["cosine_similarity"])


def write_full_output(pairs, output_path):
    """
    Write all homonym pairs to a CSV file.

    Fields:
      word1, word2, cosine_similarity,
      word1_freq, word1_docfreq, word2_freq, word2_docfreq
    """
    print(f"  Writing full homonym pair list to {output_path}...")
    fieldnames = [
        "word1", "word2", "cosine_similarity",
        "word1_freq", "word1_docfreq",
        "word2_freq", "word2_docfreq",
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
            })


def write_top_for_naming(pairs, output_path, top_n):
    """
    Write the top `top_n` most dissimilar pairs (by cosine_similarity, lowest values)
    for manual naming.

    Adds a 'name' column at the end (empty for now).

    Fields:
      word1, word2, cosine_similarity,
      word1_freq, word1_docfreq, word2_freq, word2_docfreq, name
    """
    top_pairs = pairs[:top_n]
    print(f"  Writing top {len(top_pairs)} most dissimilar pairs for naming to {output_path}...")

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
    2. Reorder pairs so the more frequent word is in word1.
    3. Sort by cosine_similarity (ascending, most dissimilar first).
    4. Write full sorted list.
    5. Write top-N most dissimilar list for naming (with empty 'name').
    """
    print(f"\nProcessing dataset: {config['name']}")
    pairs = load_pairs(config["input"])

    if not pairs:
        print("  No pairs found; skipping outputs.")
        return

    reordered = reorder_all_pairs(pairs)
    sorted_pairs = sort_by_similarity(reordered)

    write_full_output(sorted_pairs, config["full_output"])
    write_top_for_naming(sorted_pairs, config["top_output"], TOP_N)

    print(f"  Done processing {config['name']}. "
          f"{len(sorted_pairs)} homonym pairs in full file, "
          f"{min(TOP_N, len(sorted_pairs))} in top-N file.")


def main():
    for cfg in DATASETS:
        process_dataset(cfg)


if __name__ == "__main__":
    main()