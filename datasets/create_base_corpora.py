import os
import json
import pandas as pd
import random

"""
Creates base corpora for RecipeNLG and TinyStories.

Steps for both datasets:
1. Loads raw data.
   - RecipeNLG: CSV with 'title' and 'directions'
   - TinyStories: two TXT files, joined together
2. Cleans text using a character replacement table.
3. Replaces disallowed non-ASCII characters with spaces.
4. Normalizes whitespace.
5. Filters out entries that are too short.
6. Removes duplicates and reports how many were removed.
7. Shuffles data, splits into:
   - test set (1%)
   - remaining 99%: reserved as train+val (to be split later)
8. Tokenizes each entry to a list of words.
9. Saves:
   - JSON files: trainval and test for each corpus.
   - Log file: `create_base_corpora_log.txt`
"""

# === Parameters ===
RECIPE_CSV = "RecipeNLG_dataset.csv"
TINYSTORIES_TRAIN = "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID = "TinyStoriesV2-GPT4-valid.txt"

MIN_LENGTH_RECIPE = 180  # Minimum char length for RecipeNLG
MIN_LENGTH_TINYSTORIES = 180  # Minimum char length for TinyStories
SEED = 42  # Random seed for reproducibility
END_OF_TEXT = "<|endoftext|>"
ALLOWED_NON_ASCII = {'\u2014'}  # em-dash

# === Text cleaning replacements ===
replacement_table = {
    '\u200b': '',
    '\xa0': '',
    '\xad': '',
    '\\': '',
    '*': '',
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '\n': ' ',
    '---': ' — ',
    '--': ' — ',
    ' - ': ' — ',
    '–': ' — ',
    '…': '...'
}

# === Helpers ===
def clean_text(text):
    """Applies character replacements and normalizes whitespace."""
    for k, v in replacement_table.items():
        text = text.replace(k, v)
    return " ".join(text.split()).strip()

def replace_disallowed_non_ascii(text):
    """Replaces disallowed non-ASCII characters with spaces."""
    return ''.join(
        ch if ch.isascii() or ch in ALLOWED_NON_ASCII else ' '
        for ch in text
    )

# === RecipeNLG processing ===
def load_and_process_recipes(csv_path):
    """
    Loads and cleans recipes from CSV.
    Filters out entries that are too short or malformed.
    Returns tokenized list and skip counts.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} recipes from CSV")

    cleaned = []
    skipped = {
        "missing_title_or_directions": 0,
        "invalid_json": 0,
        "not_a_list": 0,
        "too_short": 0
    }

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        directions_raw = row.get("directions", "")

        if not title or not directions_raw or pd.isna(directions_raw):
            skipped["missing_title_or_directions"] += 1
            continue

        try:
            directions_list = json.loads(directions_raw)
            if not isinstance(directions_list, list):
                skipped["not_a_list"] += 1
                continue
            directions_text = " ".join(directions_list)
        except json.JSONDecodeError:
            skipped["invalid_json"] += 1
            continue

        title_clean = clean_text(title)
        directions_clean = clean_text(directions_text)
        full_text = f"Title: {title_clean}\nDirections: {directions_clean}"
        full_text = replace_disallowed_non_ascii(full_text)
        full_text = " ".join(full_text.split()).strip()

        if len(full_text) < MIN_LENGTH_RECIPE:
            skipped["too_short"] += 1
            continue

        tokens = full_text.split()
        cleaned.append(tokens)

    print(f"Kept {len(cleaned)} recipes after cleaning.")
    return cleaned, skipped

# === TinyStories processing ===
def load_and_process_stories(train_path, valid_path):
    """
    Loads and cleans stories from two TinyStories TXT files.
    Filters out stories that are too short.
    Returns tokenized list and skip counts.
    """
    with open(train_path, 'r', encoding='utf-8') as f:
        train_raw = f.read()
    with open(valid_path, 'r', encoding='utf-8') as f:
        valid_raw = f.read()

    combined_raw = train_raw + END_OF_TEXT + valid_raw
    raw_stories = [s.strip() for s in combined_raw.split(END_OF_TEXT) if s.strip()]
    print(f"Loaded {len(raw_stories)} raw stories from both files")

    cleaned = []
    skipped = {"too_short": 0}

    for story in raw_stories:
        s = clean_text(story)
        s = replace_disallowed_non_ascii(s)
        s = " ".join(s.split()).strip()

        if len(s) < MIN_LENGTH_TINYSTORIES:
            skipped["too_short"] += 1
            continue

        tokens = s.split()
        cleaned.append(tokens)

    print(f"Kept {len(cleaned)} stories after cleaning.")
    return cleaned, skipped

# === Main ===
def main():
    random.seed(SEED)
    script_dir = os.path.dirname(__file__)

    # Process RecipeNLG
    recipe_raw, recipe_skipped = load_and_process_recipes(os.path.join(script_dir, RECIPE_CSV))
    num_before = len(recipe_raw)
    recipe_unique = [list(x) for x in {tuple(x) for x in recipe_raw}]
    num_after = len(recipe_unique)
    recipe_duplicates = num_before - num_after
    print(f"Removed {recipe_duplicates} duplicate recipes.")

    random.shuffle(recipe_unique)
    recipe_test_size = max(1, int(len(recipe_unique) * 0.01))
    recipe_test = recipe_unique[:recipe_test_size]
    recipe_trainval = recipe_unique[recipe_test_size:]

    # Process TinyStories
    tinystories_raw, tinystories_skipped = load_and_process_stories(
        os.path.join(script_dir, TINYSTORIES_TRAIN),
        os.path.join(script_dir, TINYSTORIES_VALID)
    )
    num_before = len(tinystories_raw)
    tinystories_unique = [list(x) for x in {tuple(x) for x in tinystories_raw}]
    num_after = len(tinystories_unique)
    tinystories_duplicates = num_before - num_after
    print(f"Removed {tinystories_duplicates} duplicate stories.")

    random.shuffle(tinystories_unique)
    tinystories_test_size = max(1, int(len(tinystories_unique) * 0.01))
    tinystories_test = tinystories_unique[:tinystories_test_size]
    tinystories_trainval = tinystories_unique[tinystories_test_size:]

    # Save JSON files
    def save_json(data, name):
        with open(os.path.join(script_dir, name), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_json(recipe_trainval, 'base_recipe_trainval.json')
    save_json(recipe_test, 'base_recipe_test.json')
    save_json(tinystories_trainval, 'base_tiny_trainval.json')
    save_json(tinystories_test, 'base_tiny_test.json')
    print("Saved all JSON files.")

    # === Count total words and percentages ===
    def count_words(data):
        return sum(len(doc) for doc in data)

    recipe_trainval_words = count_words(recipe_trainval)
    recipe_test_words = count_words(recipe_test)
    recipe_total_words = recipe_trainval_words + recipe_test_words

    tinystories_trainval_words = count_words(tinystories_trainval)
    tinystories_test_words = count_words(tinystories_test)
    tinystories_total_words = tinystories_trainval_words + tinystories_test_words

    def pct(part, whole):
        return (part / whole) * 100 if whole > 0 else 0.0

    # === Write log ===
    log_path = os.path.join(script_dir, 'create_base_corpora_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== RecipeNLG ===\n")
        f.write(f"Kept: {len(recipe_unique)} (after deduplication)\n")
        for k, v in recipe_skipped.items():
            f.write(f"Skipped {k}: {v}\n")
        f.write(f"Duplicates removed: {recipe_duplicates}\n")
        f.write(f"Train/Val: {len(recipe_trainval)}\n")
        f.write(f"Test: {len(recipe_test)}\n")
        f.write(f"Train/Val word count: {recipe_trainval_words:,} "
                f"({pct(recipe_trainval_words, recipe_total_words):.4f}%)\n")
        f.write(f"Test word count: {recipe_test_words:,} "
                f"({pct(recipe_test_words, recipe_total_words):.4f}%)\n\n")

        f.write("=== TinyStories ===\n")
        f.write(f"Kept: {len(tinystories_unique)} (after deduplication)\n")
        for k, v in tinystories_skipped.items():
            f.write(f"Skipped {k}: {v}\n")
        f.write(f"Duplicates removed: {tinystories_duplicates}\n")
        f.write(f"Train/Val: {len(tinystories_trainval)}\n")
        f.write(f"Test: {len(tinystories_test)}\n")
        f.write(f"Train/Val word count: {tinystories_trainval_words:,} "
                f"({pct(tinystories_trainval_words, tinystories_total_words):.4f}%)\n")
        f.write(f"Test word count: {tinystories_test_words:,} "
                f"({pct(tinystories_test_words, tinystories_total_words):.4f}%)\n")

    print(f"Wrote log to {log_path}")

if __name__ == "__main__":
    main()