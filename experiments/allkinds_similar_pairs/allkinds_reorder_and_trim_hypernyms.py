"""
allkinds_reorder_and_trim_hypernyms.py
======================================

This script processes word pairs selected for experiments with artificial hypernyms
in the "all word kinds" setup (M/P/H). It:

- Reorders each pair so the more frequent word comes first.
- Removes any pair containing the word "farmers".
- Trims the list to a specified number of most similar pairs (based on cosine similarity).
- Checks whether the constructed artificial hypernym names appear in the corresponding corpus.
- Produces a shuffled version of the list with the most similar pair at the top.

Inputs:
-------
- recipe_hypernym_names_allkinds.csv
- tiny_hypernym_names_allkinds.csv
- Preprocessed corpus JSON files for RecipeNLG and TinyStories.

Outputs:
--------
- recipe_trimmed_hypernyms_sorted_allkinds.csv
- tiny_trimmed_hypernyms_sorted_allkinds.csv
- recipe_final_hypernyms_shuffled_allkinds.csv
- tiny_final_hypernyms_shuffled_allkinds.csv
- Text summaries for each corpus.

Usage:
------
$ python allkinds_reorder_and_trim_hypernyms.py
"""

import csv
import json
import pandas as pd

# ==================== CONFIGURATION ====================

RECIPE_FILE = "recipe_hypernym_names_allkinds.csv"
TINY_FILE = "tiny_hypernym_names_allkinds.csv"

RECIPE_OUTPUT_FILE_SORTED = "recipe_trimmed_hypernyms_sorted_allkinds.csv"
TINY_OUTPUT_FILE_SORTED = "tiny_trimmed_hypernyms_sorted_allkinds.csv"

RECIPE_OUTPUT_FILE_SHUFFLED = "recipe_final_hypernyms_shuffled_allkinds.csv"
TINY_OUTPUT_FILE_SHUFFLED = "tiny_final_hypernyms_shuffled_allkinds.csv"

RECIPE_CORPUS_PATH = "/home/nina/ambiguity/trained_models/word2vec_recipe_base/preprocessed_corpus.json"
TINY_CORPUS_PATH = "/home/nina/ambiguity/trained_models/word2vec_tiny_base/preprocessed_corpus.json"

TRIM_TO = 50   # Number of pairs to keep

# ==================== FUNCTIONS ====================

def load_pairs(filepath):
    """Load pairs from a CSV file."""
    print(f"Loading pairs from {filepath}...")
    with open(filepath, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def remove_pairs_containing_word(pairs, word):
    """Separate out any pairs that contain the specified word."""
    kept = []
    removed = []
    for row in pairs:
        if row['word1'] == word or row['word2'] == word:
            removed.append(row)
        else:
            kept.append(row)
    if removed:
        print(f"Removed {len(removed)} pair(s) containing the word '{word}'.")
    return kept, removed

def reorder_pair(row):
    """
    Ensures that in each pair, word1 is the more frequent word.
    Swaps word1 and word2 along with their associated frequency and docfreq fields if needed.
    """
    if int(row['word1_freq']) < int(row['word2_freq']):
        row['word1'], row['word2'] = row['word2'], row['word1']
        row['word1_freq'], row['word2_freq'] = row['word2_freq'], row['word1_freq']
        row['word1_docfreq'], row['word2_docfreq'] = row['word2_docfreq'], row['word1_docfreq']
    return row

def trim_pairs(pairs, max_pairs):
    """Keep only the top N most similar pairs based on cosine similarity."""
    print(f"Trimming to top {max_pairs} most similar pairs...")
    sorted_pairs = sorted(pairs, key=lambda x: float(x['cosine_similarity']), reverse=True)
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
    """Return a list of (hypernym_name, found: bool) for each pair."""
    return [(row['hypernym_name'], row['hypernym_name'] in corpus_words) for row in pairs]

def write_summary(removed_pairs, kept_pairs, corpus_words, filename, forbidden_pairs=None):
    """Save details of removed and excluded pairs and presence of names in corpus to a text file."""
    txt_path = filename.replace(".csv", "_summary.txt")
    print(f"Writing summary to {txt_path}...")
    with open(txt_path, 'w', encoding='utf-8') as f:

        if forbidden_pairs:
            f.write(f"=== Pairs Excluded Due to Forbidden Word (\"farmers\") ===\n\n")
            for row in forbidden_pairs:
                name = row['hypernym_name']
                w1 = row['word1']
                w2 = row['word2']
                sim = row['cosine_similarity']
                f.write(f"- {name}: {w1}, {w2}, similarity={sim}\n")
            f.write("\n")

        f.write(f"Removed {len(removed_pairs)} pairs (due to trimming):\n\n")
        for row in removed_pairs:
            name = row['hypernym_name']
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

def shuffle_pairs_file(input_path, output_path):
    """
    Keeps the most similar pair at the top (highest cosine similarity),
    shuffles the rest, and saves to a new CSV file.
    """
    df = pd.read_csv(input_path)
    top_row = df.loc[df['cosine_similarity'].idxmax()]
    remaining_rows = df.drop(df['cosine_similarity'].idxmax())
    shuffled_rows = remaining_rows.sample(frac=1, random_state=42).reset_index(drop=True)
    reordered_df = pd.concat([top_row.to_frame().T, shuffled_rows], ignore_index=True)
    reordered_df.to_csv(output_path, index=False)
    print(f"Saved shuffled output to: {output_path}")

# ==================== MAIN ====================

def process_dataset(input_csv_path, output_csv_path, corpus_path, shuffled_output_path):
    """Main processing pipeline for one dataset."""
    print(f"\nProcessing dataset: {input_csv_path}")
    pairs = load_pairs(input_csv_path)
    print(f"Loaded {len(pairs)} pairs.")
    pairs, forbidden = remove_pairs_containing_word(pairs, "farmers")
    pairs = [reorder_pair(row) for row in pairs]
    kept, removed = trim_pairs(pairs, TRIM_TO)
    corpus_words = load_corpus(corpus_path)
    save_pairs(kept, output_csv_path)
    write_summary(removed, kept, corpus_words, output_csv_path, forbidden_pairs=forbidden)

    # Shuffle and save to separate file
    shuffle_pairs_file(output_csv_path, shuffled_output_path)

    print(f"Done processing {input_csv_path}. Kept {len(kept)} pairs, removed {len(removed)}.\n")

if __name__ == "__main__":
    # RecipeNLG (all kinds)
    process_dataset(RECIPE_FILE, RECIPE_OUTPUT_FILE_SORTED,
                    RECIPE_CORPUS_PATH, RECIPE_OUTPUT_FILE_SHUFFLED)

    # TinyStories (all kinds)
    process_dataset(TINY_FILE, TINY_OUTPUT_FILE_SORTED,
                    TINY_CORPUS_PATH, TINY_OUTPUT_FILE_SHUFFLED)