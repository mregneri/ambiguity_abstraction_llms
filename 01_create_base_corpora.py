"""
01_create_base_corpora.py
=========================

Step 1 of the pipeline. Creates and saves base corpora for RecipeNLG and
TinyStories, then splits them into train / val / test sets.

Outputs (written to data/processed/):
    recipe_base_trainval.json  ← intermediate (kept for reproducibility)
    recipe_base_train.json
    recipe_base_val.json
    recipe_base_test.json
    tiny_base_trainval.json
    tiny_base_train.json
    tiny_base_val.json
    tiny_base_test.json

Run:
    python 01_create_base_corpora.py

Prerequisites:
    Place the following raw files in data/raw/ before running:
      • RecipeNLG_dataset.csv         (https://recipenlg.cs.put.poznan.pl/)
      • TinyStoriesV2-GPT4-train.txt  (https://huggingface.co/datasets/roneneldan/TinyStories)
      • TinyStoriesV2-GPT4-valid.txt
"""

import os
import json
import random
import pandas as pd
from datetime import datetime
from config import (
    RECIPE_CSV, TINYSTORIES_TRAIN, TINYSTORIES_VALID,
    RECIPE_TRAINVAL, RECIPE_TRAIN, RECIPE_VAL, RECIPE_TEST,
    TINY_TRAINVAL, TINY_TRAIN, TINY_VAL, TINY_TEST,
    PROCESSED_DIR,
)

# ── Parameters ───────────────────────────────────────────────────────────────
SEED               = 42
VAL_RATIO          = 0.10
TEST_RATIO         = 0.01
MIN_LENGTH_RECIPE  = 180
MIN_LENGTH_TINY    = 180
ALLOWED_NON_ASCII  = {'\u2014'}  # em-dash

REPLACEMENT_TABLE = {
    '\u200b': '', '\xa0': '', '\xad': '', '\\': '', '*': '',
    '«': '"', '»': '"', '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"', '\n': ' ',
    '---': ' — ', '--': ' — ', ' - ': ' — ', '–': ' — ', '…': '...',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    for k, v in REPLACEMENT_TABLE.items():
        text = text.replace(k, v)
    return " ".join(text.split()).strip()

def replace_disallowed_non_ascii(text: str) -> str:
    return ''.join(ch if ch.isascii() or ch in ALLOWED_NON_ASCII else ' ' for ch in text)

def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def deduplicate(data):
    before = len(data)
    unique = [list(x) for x in {tuple(x) for x in data}]
    return unique, before - len(unique)

def split_three(data, val_ratio, test_ratio, seed):
    """Shuffle and split into train / val / test."""
    random.seed(seed)
    random.shuffle(data)
    n      = len(data)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int((n - n_test) * val_ratio))
    test   = data[:n_test]
    val    = data[n_test:n_test + n_val]
    train  = data[n_test + n_val:]
    return train, val, test

# ── RecipeNLG ─────────────────────────────────────────────────────────────────
def load_recipes(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows from RecipeNLG CSV")
    cleaned, skipped = [], {"missing": 0, "bad_json": 0, "not_list": 0, "too_short": 0}
    for _, row in df.iterrows():
        title    = str(row.get("title", "")).strip()
        raw_dirs = row.get("directions", "")
        if not title or not raw_dirs or pd.isna(raw_dirs):
            skipped["missing"] += 1; continue
        try:
            dirs = json.loads(raw_dirs)
            if not isinstance(dirs, list): skipped["not_list"] += 1; continue
        except json.JSONDecodeError:
            skipped["bad_json"] += 1; continue
        text = clean_text(f"Title: {title}\nDirections: {' '.join(dirs)}")
        text = replace_disallowed_non_ascii(text)
        text = " ".join(text.split()).strip()
        if len(text) < MIN_LENGTH_RECIPE: skipped["too_short"] += 1; continue
        cleaned.append(text.split())
    print(f"  Kept {len(cleaned):,} recipes  |  skipped: {skipped}")
    return cleaned

# ── TinyStories ───────────────────────────────────────────────────────────────
def load_stories(train_path: str, valid_path: str):
    END = "<|endoftext|>"
    raw = []
    for p in (train_path, valid_path):
        with open(p, 'r', encoding='utf-8') as f:
            raw += [s.strip() for s in f.read().split(END) if s.strip()]
    print(f"  Loaded {len(raw):,} raw stories")
    cleaned, skipped = [], 0
    for story in raw:
        s = clean_text(story)
        s = replace_disallowed_non_ascii(s)
        s = " ".join(s.split()).strip()
        if len(s) < MIN_LENGTH_TINY: skipped += 1; continue
        cleaned.append(s.split())
    print(f"  Kept {len(cleaned):,} stories  |  skipped (too short): {skipped}")
    return cleaned

# ── Main ──────────────────────────────────────────────────────────────────────
def process_corpus(name, data, train_path, val_path, test_path, trainval_path, log_lines):
    data, n_dup = deduplicate(data)
    print(f"  Removed {n_dup} duplicates → {len(data):,} unique docs")
    train, val, test = split_three(data, VAL_RATIO, TEST_RATIO, SEED)

    # Save intermediate trainval for word2vec (uses full non-test set)
    save_json(train + val, trainval_path)
    save_json(train, train_path)
    save_json(val,   val_path)
    save_json(test,  test_path)

    def wc(d): return sum(len(x) for x in d)
    total = wc(train) + wc(val) + wc(test)
    log_lines.append(
        f"\n=== {name} ===\n"
        f"  Docs — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}\n"
        f"  Words— train: {wc(train):,}  val: {wc(val):,}  test: {wc(test):,}  "
        f"total: {total:,}\n"
        f"  Duplicates removed: {n_dup}\n"
    )
    print(f"  ✅ Saved {len(train):,} train / {len(val):,} val / {len(test):,} test")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    log_lines = [f"=== Base Corpora Creation Log ===\n{datetime.now()}\n"]

    print("\n── RecipeNLG ──────────────────────────────────────────")
    recipes = load_recipes(RECIPE_CSV)
    process_corpus("RecipeNLG", recipes,
                   RECIPE_TRAIN, RECIPE_VAL, RECIPE_TEST, RECIPE_TRAINVAL, log_lines)

    print("\n── TinyStories ─────────────────────────────────────────")
    stories = load_stories(TINYSTORIES_TRAIN, TINYSTORIES_VALID)
    process_corpus("TinyStories", stories,
                   TINY_TRAIN, TINY_VAL, TINY_TEST, TINY_TRAINVAL, log_lines)

    log_path = os.path.join(PROCESSED_DIR, "create_base_corpora_log.txt")
    with open(log_path, 'w') as f:
        f.writelines(log_lines)
    print(f"\n📝 Log written to {log_path}")
    print("🎉 Base corpora created successfully.")

if __name__ == "__main__":
    main()
