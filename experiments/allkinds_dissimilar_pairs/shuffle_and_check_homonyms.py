"""
shuffle_and_check_homonyms.py
==============================

This script performs the final preparation step for experiments with artificial
homonyms. It takes the manually named top-50 homonym pairs for each corpus and:

1. Checks, for each homonym name, whether it already occurs in the respective
   base corpus (as a token in the preprocessed corpus JSON), using a
   punctuation-aware, case-insensitive matching strategy similar to the
   artificial word insertion scripts:
   - Tokens are split into word and punctuation parts (r'\\w+|[^\\w\\s]').
   - Only the word-like parts (\\w+) are kept.
   - Word parts are lowercased before being added to the token set.
   - Homonym names from the CSV are stripped and lowercased before comparison.

2. Produces a shuffled version of the pairs where:
   - The first row remains in place.
   - All remaining rows are randomly shuffled in a reproducible way.

For each corpus (TinyStories, RecipeNLG), the script:

- Reads the corresponding `*_top50_homonym_pairs_with_names.csv` file.
- Loads the preprocessed corpus and builds a normalized set of word tokens.
- Writes a summary text file listing, for every homonym name, whether it was
  FOUND or NOT FOUND in the corpus.
- Writes a final shuffled CSV file, with the same columns as the input:
  word1, word2, cosine_similarity, word1_freq, word1_docfreq,
  word2_freq, word2_docfreq, name

Inputs:
-------
- tiny_top50_homonym_pairs_with_names.csv
- recipe_top50_homonym_pairs_with_names.csv
- Preprocessed corpus JSON files for TinyStories and RecipeNLG.

Outputs:
--------
TinyStories:
  - tiny_homonym_pairs_final_shuffled.csv
  - tiny_homonym_pairs_final_shuffled_summary.txt

RecipeNLG:
  - recipe_homonym_pairs_final_shuffled.csv
  - recipe_homonym_pairs_final_shuffled_summary.txt

Usage:
------
$ python shuffle_and_check_homonyms.py
"""

import csv
import json
import re
import pandas as pd

# ==================== CONFIGURATION ====================

TINY_INPUT = "tiny_top50_homonym_pairs_with_names.csv"
RECIPE_INPUT = "recipe_top50_homonym_pairs_with_names.csv"

TINY_OUTPUT = "tiny_homonym_pairs_final_shuffled.csv"
RECIPE_OUTPUT = "recipe_homonym_pairs_final_shuffled.csv"

# Preprocessed base corpora (tokenized JSON lists of lists)
TINY_CORPUS_PATH = "/home/nina/ambiguity/trained_models/word2vec_tiny_base/preprocessed_corpus.json"
RECIPE_CORPUS_PATH = "/home/nina/ambiguity/trained_models/word2vec_recipe_base/preprocessed_corpus.json"

DATASETS = [
    {
        "name": "TinyStories",
        "input_csv": TINY_INPUT,
        "output_csv": TINY_OUTPUT,
        "corpus_path": TINY_CORPUS_PATH,
    },
    {
        "name": "RecipeNLG",
        "input_csv": RECIPE_INPUT,
        "output_csv": RECIPE_OUTPUT,
        "corpus_path": RECIPE_CORPUS_PATH,
    },
]

# ==================== HELPERS ====================

def load_pairs(csv_path):
    """
    Load rows from a CSV file into a list of dictionaries.

    Expects at least the columns:
        word1, word2, cosine_similarity,
        word1_freq, word1_docfreq, word2_freq, word2_docfreq, name
    """
    pairs = []
    print(f"Loading pairs from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    print(f"  Loaded {len(pairs)} rows.")
    return pairs


def load_corpus(path):
    """
    Load a preprocessed corpus from a JSON file and build a set of normalized word tokens.

    Normalization:
      - Split each token into word and punctuation parts using the same regex as in the
        artificial word insertion scripts (r'\\w+|[^\\w\\s]').
      - Keep only the "word-like" parts (\\w+).
      - Lowercase those parts before adding them to the set.

    This makes the presence check robust to casing and punctuation attached to words,
    e.g. 'Papango,' or '(papango)' will both contribute 'papango' to the token set.
    """
    print(f"Loading corpus from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    token_set = set()

    for doc in corpus:
        for token in doc:
            # Split token into word and punctuation parts
            parts = re.findall(r'\w+|[^\w\s]', token, re.UNICODE)
            for part in parts:
                # Keep only word-like parts and normalize to lowercase
                if re.fullmatch(r'\w+', part):
                    token_set.add(part.lower())

    print(f"  Corpus contains {len(token_set)} distinct normalized word tokens.")
    return token_set


def check_name_in_corpus(pairs, corpus_words):
    """
    Return a list of (original_name, found: bool) for each row.

    Matching is done in a normalized space:
      - The 'name' from the CSV is stripped and lowercased.
      - It is then checked against the normalized, lowercase token set built from the corpus.
    """
    results = []
    for row in pairs:
        original_name = row.get("name", "")
        name = (original_name or "").strip()

        if not name:
            # Still record it, but mark as NOT FOUND explicitly
            results.append((original_name, False))
        else:
            normalized = name.lower()
            results.append((original_name, normalized in corpus_words))
    return results


def write_summary(pairs, corpus_words, dataset_name, input_csv_path, output_csv_path):
    """
    Write a summary text file listing, for each homonym name, whether it is
    found in the corpus.

    The summary file is named by replacing '.csv' in the output path with
    '_summary.txt'.
    """
    txt_path = output_csv_path.replace(".csv", "_summary.txt")
    print(f"Writing summary to {txt_path}...")

    name_checks = check_name_in_corpus(pairs, corpus_words)
    total_rows = len(name_checks)
    found_count = sum(1 for _, present in name_checks if present)
    not_found_count = total_rows - found_count

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Homonym Name Corpus Check – {dataset_name}\n")
        f.write("=" * (len(dataset_name) + 28) + "\n\n")

        f.write(f"Input CSV : {input_csv_path}\n")
        f.write(f"Output CSV: {output_csv_path}\n\n")

        f.write(f"Total pairs              : {total_rows}\n")
        f.write(f"Names FOUND in corpus    : {found_count}\n")
        f.write(f"Names NOT FOUND in corpus: {not_found_count}\n\n")

        f.write("Details:\n")
        for name, present in name_checks:
            label = "FOUND" if present else "NOT FOUND"
            # Handle the (unlikely) case of an empty name explicitly
            display_name = name if name else "<EMPTY_NAME>"
            f.write(f"  {display_name}: {label}\n")


def shuffle_pairs_file(input_path, output_path):
    """
    Keeps the first row in place and shuffles the remaining rows,
    then saves to a new CSV file.

    The shuffle is done with a fixed random_state for reproducibility.
    """
    print(f"Shuffling pairs from {input_path} -> {output_path}...")
    df = pd.read_csv(input_path)

    if len(df) <= 1:
        # Nothing to shuffle
        df.to_csv(output_path, index=False)
        print("  File has 0 or 1 row; nothing to shuffle.")
        return

    top_row = df.iloc[0:1]        # first row as DataFrame
    remaining_rows = df.iloc[1:]  # all rows except the first

    shuffled_rows = remaining_rows.sample(frac=1, random_state=42).reset_index(drop=True)
    reordered_df = pd.concat([top_row, shuffled_rows], ignore_index=True)
    reordered_df.to_csv(output_path, index=False)
    print(f"  Saved shuffled output to: {output_path}")


def process_dataset(config):
    """
    Full processing pipeline for one dataset (TinyStories or RecipeNLG):

    1. Load the base corpus and flatten to a normalized token set.
    2. Load the named homonym pairs from the input CSV.
    3. Write a summary indicating whether each homonym name is found in the corpus.
    4. Shuffle all rows except the first and write the final CSV.
    """
    print(f"\nProcessing dataset: {config['name']}")
    corpus_words = load_corpus(config["corpus_path"])
    pairs = load_pairs(config["input_csv"])

    write_summary(
        pairs,
        corpus_words,
        dataset_name=config["name"],
        input_csv_path=config["input_csv"],
        output_csv_path=config["output_csv"],
    )

    shuffle_pairs_file(config["input_csv"], config["output_csv"])

    print(f"Done processing {config['name']}.")


def main():
    for cfg in DATASETS:
        process_dataset(cfg)


if __name__ == "__main__":
    main()