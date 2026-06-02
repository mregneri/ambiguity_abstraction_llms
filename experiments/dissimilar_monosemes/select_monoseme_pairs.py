"""
select_monoseme_pairs.py
==============================

This script selects a set of dissimilar monoseme pairs from CSV files created by
`monoseme_analysis.py`, ensuring that each monoseme appears in only one pair.

Selection Strategy:
-------------------
- Count how many times each monoseme appears across all candidate pairs.
- Sort monosemes in ascending order by this count (rarest monosemes first).
- For each monoseme in that order:
  - Skip it if it has already been used in a selected pair.
  - Look through all candidate pairs that include the monoseme.
  - Filter to pairs where both monosemes are still unused.
  - From the valid options, select the pair with the lowest cosine similarity
    (i.e., the most semantically dissimilar).
  - Mark both monosemes as used.
- Continue until no more non-overlapping pairs can be formed.

Inputs:
-------
- CSV files of candidate monoseme pairs with the following columns:
  word1, word2, cosine_similarity, word1_freq, word1_docfreq, word2_freq, word2_docfreq

Outputs:
--------
- CSV files of selected non-overlapping monoseme pairs, preserving the same columns.

Usage:
------
$ python select_monoseme_pairs.py

To add or modify corpora, update the PAIR_FILES list in the script.
"""

import csv
from collections import defaultdict

# ==================== CONFIGURATION ====================

PAIR_FILES = [
    {
        'name': 'TinyStories',
        'input_csv': 'base_tiny_dissimilar_pairs.csv',
        'output_csv': 'final_pairs_tiny.csv'
    },
    {
        'name': 'RecipeNLG',
        'input_csv': 'base_recipe_dissimilar_pairs.csv',
        'output_csv': 'final_pairs_recipe.csv'
    }
]

# ==================== LOAD AND INDEX ====================

def load_pairs(csv_path):
    """Loads candidate pairs from CSV and returns a list of dictionaries for each row."""
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

def build_monoseme_counts(pairs):
    """Returns a dictionary counting how many times each monoseme appears across all pairs."""
    counts = defaultdict(int)
    for row in pairs:
        counts[row['word1']] += 1
        counts[row['word2']] += 1
    return counts

def build_pair_lookup(pairs):
    """Creates a lookup from each monoseme to all pairs it occurs in."""
    lookup = defaultdict(list)
    for row in pairs:
        lookup[row['word1']].append(row)
        lookup[row['word2']].append(row)
    return lookup

# ==================== SELECTION LOGIC ====================

def select_pairs(pairs):
    """Greedily selects non-overlapping pairs by prioritizing monosemes with fewest options."""
    monoseme_counts = build_monoseme_counts(pairs)
    pair_lookup = build_pair_lookup(pairs)

    # Sort monosemes by how often they appear in pairs (ascending)
    sorted_monosemes = sorted(monoseme_counts, key=lambda w: monoseme_counts[w])

    used_words = set()
    selected_pairs = []

    for monoseme in sorted_monosemes:
        if monoseme in used_words:
            continue

        # Look through all candidate pairs containing this monoseme
        candidate_pairs = pair_lookup[monoseme]

        # Filter to pairs where both words are unused
        valid_pairs = [p for p in candidate_pairs if p['word1'] not in used_words and p['word2'] not in used_words]

        if not valid_pairs:
            continue

        # Pick the most dissimilar pair (lowest cosine_similarity)
        best_pair = min(valid_pairs, key=lambda p: p['cosine_similarity'])

        selected_pairs.append(best_pair)
        used_words.add(best_pair['word1'])
        used_words.add(best_pair['word2'])

    return selected_pairs

# ==================== SAVE RESULTS ====================

def save_selected_pairs(pairs, output_path):
    """Saves the selected non-overlapping monoseme pairs to CSV."""
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

        print("Selecting non-overlapping monoseme pairs...")
        selected = select_pairs(all_pairs)

        print(f"Selected {len(selected)} non-overlapping pairs.")
        print(f"Saving results to: {config['output_csv']}")
        save_selected_pairs(selected, config['output_csv'])

if __name__ == '__main__':
    main()