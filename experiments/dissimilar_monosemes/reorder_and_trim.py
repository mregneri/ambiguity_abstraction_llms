"""
reorder_and_trim.py
=======================

This script processes monoseme word pairs selected for experiments with artificial homonyms.
It ensures that each pair is ordered by word frequency, removes any pair containing the word
"farmers", trims the list to a specified number of most dissimilar pairs (based on cosine similarity),
and checks whether the constructed artificial homonym names appear in the corresponding corpus.

Main Features:
--------------
- Reorders each monoseme pair so the more frequent word comes first.
- Excludes pairs that contain the word "farmers".
- Keeps only the top N most dissimilar pairs according to cosine similarity.
- Verifies whether each homonym name exists in the target corpus.
- Saves the final selected pairs and a summary of removed pairs.

Inputs:
-------
- Two CSV files with monoseme pair candidates for RecipeNLG and TinyStories.
- Two preprocessed corpus files in JSON format.

Outputs:
--------
- CSV files with reordered and trimmed monoseme pairs.
- Text summary files listing removed pairs and corpus name checks.

Usage:
------
$ python reorder_and_trim.py
"""

import csv
import json
import os

# ==================== CONFIGURATION ====================

RECIPE_FILE = "RecipeNLG_final_nameproposal.csv"   # Input file for RecipeNLG pairs
TINY_FILE = "tiny_final_nameproposal-2.csv"           # Input file for TinyStories pairs
RECIPE_OUTPUT_FILE = "recipe_artificial_homonyms.csv"  # Output file for RecipeNLG
TINY_OUTPUT_FILE = "tiny_artificial_homonyms.csv"         # Output file for TinyStories
RECIPE_CORPUS_PATH = "/home/nina/ambiguity/word2vec_pipelines/word2vec_recipe_base/preprocessed_corpus.json"
TINY_CORPUS_PATH = "/home/nina/ambiguity/word2vec_pipelines/word2vec_tiny_base/preprocessed_corpus.json"
TRIM_TO = 50   # Number of pairs to keep

# ==================== FUNCTIONS ====================

def load_pairs(filepath):
    """Load monoseme pairs from a CSV file."""
    print(f"Loading pairs from {filepath}...")
    with open(filepath, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def remove_pairs_containing_word(pairs, word):
    """Remove any pairs that contain the specified word."""
    original_len = len(pairs)
    filtered = [row for row in pairs if word != row['word1'] and word != row['word2']]
    removed_count = original_len - len(filtered)
    if removed_count:
        print(f"Removed {removed_count} pair(s) containing the word '{word}'.")
    return filtered

def reorder_pair(row):
    """
    Ensures that in each pair, word1 is the more frequent word.
    Swaps word1 and word2 along with their associated frequency and docfreq fields if needed.
    """
    if int(row['word1_freq']) < int(row['word2_freq']):
        # Swap values between word1 and word2 (including metadata)
        row['word1'], row['word2'] = row['word2'], row['word1']
        row['word1_freq'], row['word2_freq'] = row['word2_freq'], row['word1_freq']
        row['word1_docfreq'], row['word2_docfreq'] = row['word2_docfreq'], row['word1_docfreq']
    return row

def trim_pairs(pairs, max_pairs):
    """Keep only the top N most dissimilar pairs based on cosine similarity."""
    print(f"Trimming to top {max_pairs} most dissimilar pairs...")
    sorted_pairs = sorted(pairs, key=lambda x: float(x['cosine_similarity']))
    kept = sorted_pairs[:max_pairs]
    removed = sorted_pairs[max_pairs:]
    return kept, removed

def load_corpus(path):
    """Load a preprocessed corpus from a JSON file and flatten into a single word set."""
    print(f"Loading corpus from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return set(word for doc in corpus for word in doc)

def save_pairs(pairs, filename):
    """Save updated pairs to a CSV file."""
    print(f"Saving reordered and trimmed pairs to {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
        writer.writeheader()
        writer.writerows(pairs)

def check_name_in_corpus(pairs, corpus_words):
    """Return a list of (name, found: bool) for each pair."""
    return [(row['name'], row['name'] in corpus_words) for row in pairs]

def write_summary(removed_pairs, kept_pairs, corpus_words, filename):
    """Save details of removed pairs and presence of names in corpus to a text file."""
    txt_path = filename.replace(".csv", "_summary.txt")
    print(f"Writing summary to {txt_path}...")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Removed {len(removed_pairs)} pairs:\n\n")
        for row in removed_pairs:
            name = row['name']
            w1 = row['word1']
            w2 = row['word2']
            sim = row['cosine_similarity']
            f.write(f"- {name}: {w1} ({row['word1_freq']}), {w2} ({row['word2_freq']}), similarity={sim}\n")

        f.write("\n=== Corpus Check: Removed Pairs ===\n")
        for name, present in check_name_in_corpus(removed_pairs, corpus_words):
            f.write(f"{name}: {'FOUND' if present else 'NOT FOUND'} in corpus\n")

        f.write("\n=== Corpus Check: Kept Pairs ===\n")
        for name, present in check_name_in_corpus(kept_pairs, corpus_words):
            f.write(f"{name}: {'FOUND' if present else 'NOT FOUND'} in corpus\n")

# ==================== MAIN ====================

def process_dataset(input_csv_path, output_csv_path, corpus_path):
    """Main processing pipeline for one dataset."""
    print(f"\nProcessing dataset: {input_csv_path}")
    pairs = load_pairs(input_csv_path)
    print(f"Loaded {len(pairs)} pairs.")
    pairs = remove_pairs_containing_word(pairs, "farmers")
    pairs = [reorder_pair(row) for row in pairs]
    kept, removed = trim_pairs(pairs, TRIM_TO)
    corpus_words = load_corpus(corpus_path)
    save_pairs(kept, output_csv_path)
    write_summary(removed, kept, corpus_words, output_csv_path)
    print(f"Done processing {input_csv_path}. Kept {len(kept)} pairs, removed {len(removed)}.\n")

if __name__ == "__main__":
    process_dataset(RECIPE_FILE, RECIPE_OUTPUT_FILE, RECIPE_CORPUS_PATH)
    process_dataset(TINY_FILE, TINY_OUTPUT_FILE, TINY_CORPUS_PATH)