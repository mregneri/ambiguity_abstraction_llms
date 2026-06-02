"""
allkinds_select_hypernym_pairs.py
=================================

This script selects a set of most similar non-overlapping word pairs from CSV files
created by `allkinds_hypernym_analysis.py`, ensuring that each word appears in only
one pair.

These pairs are intended for the creation of artificial hypernyms in the
"all word kinds" experiment (using M/P/H words).

In addition to producing the full set of selected non-overlapping pairs, this script
also creates a second output file per corpus containing only the top N most similar
pairs (sorted by cosine similarity). This makes it easier to manually construct
artificial hypernym names for a smaller, most relevant subset.

Selection Strategy:
-------------------
- Count how many times each word appears across all candidate pairs.
- Sort words in ascending order by this count (rarest words first).
- For each word in that order:
  - Skip it if it has already been used in a selected pair.
  - Look through all candidate pairs that include the word.
  - Filter to pairs where both words are still unused.
  - From the valid options, select the pair with the highest cosine similarity
    (i.e., the most semantically similar).
  - Mark both words as used.
- Continue until no more non-overlapping pairs can be formed.
- Finally, sort the selected pairs by similarity and keep only the top N for the
  additional trimmed output files.

Inputs:
-------
- tiny_base_similar_pairs_allkinds.csv
- recipe_base_similar_pairs_allkinds.csv

Outputs:
--------
TinyStories:
  - hypernym_pairs_tiny_allkinds.csv         (all selected non-overlapping pairs)
  - hypernym_pairs_tiny_allkinds_top55.csv   (top 55 by cosine similarity)

RecipeNLG:
  - hypernym_pairs_recipe_allkinds.csv       (all selected non-overlapping pairs)
  - hypernym_pairs_recipe_allkinds_top55.csv (top 55 by cosine similarity)

Usage:
------
$ python allkinds_select_hypernym_pairs.py
"""

import csv
from collections import defaultdict

# ==================== CONFIGURATION ====================

TRIM_TO = 55  # Number of most similar pairs to keep in the "top" files

PAIR_FILES = [
    {
        'name': 'TinyStories (all kinds)',
        'input_csv': 'tiny_base_similar_pairs_allkinds.csv',
        'output_csv_full': 'hypernym_pairs_tiny_allkinds.csv',
        'output_csv_top': 'hypernym_pairs_tiny_allkinds_top55.csv',
    },
    {
        'name': 'RecipeNLG (all kinds)',
        'input_csv': 'recipe_base_similar_pairs_allkinds.csv',
        'output_csv_full': 'hypernym_pairs_recipe_allkinds.csv',
        'output_csv_top': 'hypernym_pairs_recipe_allkinds_top55.csv',
    }
]

# ==================== LOAD AND INDEX ====================

def load_pairs(csv_path):
    """
    Loads candidate pairs from CSV and returns a list of dictionaries for each row.
    Expects the columns:
      word1, word2, cosine_similarity, word1_freq, word1_docfreq, word2_freq, word2_docfreq
    """
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['cosine_similarity'] = float(row['cosine_similarity'])
            row['word1_freq'] = int(row['word1_freq'])
            row['word1_docfreq'] = int(row['word1_docfreq'])
            row['word2_freq'] = int(row['word2_freq'])
            row['word2_docfreq'] = int(row['word2_docfreq'])
            pairs.append(row)
    return pairs


def build_word_counts(pairs):
    """Returns a dictionary counting how many times each word appears across all pairs."""
    counts = defaultdict(int)
    for row in pairs:
        counts[row['word1']] += 1
        counts[row['word2']] += 1
    return counts


def build_pair_lookup(pairs):
    """Creates a lookup from each word to all pairs it occurs in."""
    lookup = defaultdict(list)
    for row in pairs:
        lookup[row['word1']].append(row)
        lookup[row['word2']].append(row)
    return lookup

# ==================== SELECTION LOGIC ====================

def select_pairs(pairs):
    """
    Greedily selects non-overlapping pairs by prioritizing words with fewest options.
    Each word appears in at most one selected pair.
    """
    word_counts = build_word_counts(pairs)
    pair_lookup = build_pair_lookup(pairs)

    # Sort words by how often they appear in pairs (ascending)
    sorted_words = sorted(word_counts, key=lambda w: word_counts[w])

    used_words = set()
    selected_pairs = []

    for word in sorted_words:
        if word in used_words:
            continue

        # Look through all candidate pairs containing this word
        candidate_pairs = pair_lookup[word]

        # Filter to pairs where both words are unused
        valid_pairs = [
            p for p in candidate_pairs
            if p['word1'] not in used_words and p['word2'] not in used_words
        ]

        if not valid_pairs:
            continue

        # Pick the most similar pair (highest cosine_similarity)
        best_pair = max(valid_pairs, key=lambda p: p['cosine_similarity'])

        selected_pairs.append(best_pair)
        used_words.add(best_pair['word1'])
        used_words.add(best_pair['word2'])

    return selected_pairs


def trim_pairs_by_similarity(pairs, max_pairs):
    """
    Sort pairs by cosine_similarity (descending) and keep only the top `max_pairs`.

    Returns:
        kept  : list of top pairs (up to max_pairs)
        others: list of remaining pairs (not used here, but returned for completeness)
    """
    if not pairs:
        return [], []

    sorted_pairs = sorted(pairs, key=lambda p: p['cosine_similarity'], reverse=True)
    kept = sorted_pairs[:max_pairs]
    others = sorted_pairs[max_pairs:]
    return kept, others

# ==================== SAVE RESULTS ====================

def save_selected_pairs(pairs, output_path):
    """Saves the selected non-overlapping pairs to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word1', 'word2', 'cosine_similarity',
            'word1_freq', 'word1_docfreq', 'word2_freq', 'word2_docfreq'
        ])
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)

# ==================== MAIN ====================

def main():
    for config in PAIR_FILES:
        print(f"\nProcessing: {config['name']}")
        print(f"Loading candidate pairs from: {config['input_csv']}")

        all_pairs = load_pairs(config['input_csv'])
        print(f"Loaded {len(all_pairs)} candidate pairs.")

        print("Selecting non-overlapping pairs (all kinds)...")
        selected = select_pairs(all_pairs)
        print(f"Selected {len(selected)} non-overlapping pairs in total.")

        # Save full set of selected pairs
        full_path = config['output_csv_full']
        print(f"Saving full set of selected pairs to: {full_path}")
        save_selected_pairs(selected, full_path)

        # Create and save top-N subset sorted by similarity
        top_path = config['output_csv_top']
        top_pairs, _ = trim_pairs_by_similarity(selected, TRIM_TO)
        print(f"Saving top {len(top_pairs)} most similar pairs to: {top_path}")
        save_selected_pairs(top_pairs, top_path)

if __name__ == '__main__':
    main()