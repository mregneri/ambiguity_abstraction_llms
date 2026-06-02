"""
inspect_inserted_homonyms.py
=============================

This script searches a modified train and test corpus for tokens that contain
any of the artificial homonyms (as substrings). It collects up to 100 matching
tokens per split and saves them to a single text file with separate sections.

The artificial homonyms are derived from the replacement map JSON used to
insert homonyms into the corpora.

Output:
-------
- homonym_matches.txt: tokens from the trainval and test sets that include any homonym

Each section contains up to 100 unique matched tokens (case-insensitive),
one per line.
"""

import json
from tqdm import tqdm

# === Hard-coded paths ===
TRAINVAL_PATH = "/home/nina/ambiguity/datasets/allkinds/recipe_50_HOM_trainval.json"
TEST_PATH = "/home/nina/ambiguity/datasets/allkinds/recipe_50_HOM_test.json"
REPLACEMENTS_PATH = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/recipe_50_HOM_map.json"
OUTPUT_PATH = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/logs/recipe_50_HOM_homonym_matches.txt"

def load_json(filepath):
    """Load a JSON file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_homonyms(replacements_path):
    """Extract the artificial homonyms from the replacements JSON."""
    with open(replacements_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.values())

def find_homonym_matches(corpus, homonyms, max_matches=100):
    """
    Find tokens in the corpus that contain any artificial homonym as a substring.

    Matching is done case-insensitively.
    Stops when `max_matches` unique tokens have been found.

    Returns:
        A sorted list of matching tokens (original casing preserved).
    """
    matches = set()
    homonyms_lower = [h.lower() for h in homonyms]

    for doc in tqdm(corpus, desc="Searching for homonyms"):
        for token in doc:
            token_lower = token.lower()
            for hom in homonyms_lower:
                if hom in token_lower:
                    matches.add(token)
                    break  # Stop checking once a match is found
            if len(matches) >= max_matches:
                return sorted(matches)
    return sorted(matches)

def save_combined_matches(trainval_matches, test_matches, output_path):
    """Save both trainval and test matches into a single text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Trainval Matches:\n")
        f.write("=================\n")
        for token in trainval_matches:
            f.write(token + "\n")

        f.write("\nTest Matches:\n")
        f.write("=============\n")
        for token in test_matches:
            f.write(token + "\n")

def main():
    print("Loading replacement map...")
    homonyms = extract_homonyms(REPLACEMENTS_PATH)

    print(f"\nProcessing trainval set: {TRAINVAL_PATH}")
    trainval_corpus = load_json(TRAINVAL_PATH)
    trainval_matches = find_homonym_matches(trainval_corpus, homonyms)
    print(f"Found {len(trainval_matches)} matches in trainval set.")

    print(f"\nProcessing test set: {TEST_PATH}")
    test_corpus = load_json(TEST_PATH)
    test_matches = find_homonym_matches(test_corpus, homonyms)
    print(f"Found {len(test_matches)} matches in test set.")

    save_combined_matches(trainval_matches, test_matches, OUTPUT_PATH)
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()