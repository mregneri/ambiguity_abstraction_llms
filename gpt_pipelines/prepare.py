"""
prepare.py
===========

Prepares a GPT training dataset from preprocessed corpora that are already
split into train, validation, and test JSON files.

Steps:
1. Loads pre-tokenized JSON files (lists of tokenized documents).
2. Joins tokens with spaces to form text documents.
3. Concatenates each split into one long string, separated by <|endoftext|>.
4. Tokenizes all splits using the GPT-2 BPE tokenizer (via tiktoken).
5. Saves:
   - Binary token files: train.bin, val.bin, test.bin
   - Human-readable test.txt (for inspection)
   - Log file: prepare_log.txt

Example usage:
--------------
# Base corpora
python prepare.py \
  --train /home/nina/ambiguity/datasets/base/recipe_base_train.json \
  --val   /home/nina/ambiguity/datasets/base/recipe_base_val.json \
  --test  /home/nina/ambiguity/datasets/base/recipe_base_test.json \
  --out_dir /home/nina/ambiguity/gpt_pipelines/allkinds/data/gpt_recipe_base

# Allkinds HOM/HYP example (just as a template)
python prepare.py \
  --train /home/nina/ambiguity/datasets/allkinds/recipe_50_HOM_train.json \
  --val   /home/nina/ambiguity/datasets/allkinds/recipe_50_HOM_val.json \
  --test  /home/nina/ambiguity/datasets/allkinds/recipe_50_HOM_test.json \
  --out_dir /home/nina/ambiguity/gpt_pipelines/allkinds/data/gpt_recipe_50_HOM
"""

import os
import json
import argparse
import numpy as np
import tiktoken

# === Constants ===
ENCODING = "gpt2"
END_OF_TEXT = "<|endoftext|>"


def load_json(path):
    """Load a JSON corpus file (list of token lists)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def join_documents(data):
    """Join lists of tokens into single text documents."""
    return [" ".join(doc).strip() for doc in data if len(doc) > 0]


def tokenize_texts(texts, enc):
    """Concatenate texts with END_OF_TEXT and tokenize them."""
    full_text = END_OF_TEXT.join(texts)
    return enc.encode_ordinary(full_text)


def save_bin(ids, path):
    """Save token IDs as a binary .bin file."""
    np.array(ids, dtype=np.uint16).tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Prepare GPT dataset from pre-split corpora.")
    parser.add_argument("--train", required=True, help="Path to training JSON file")
    parser.add_argument("--val", required=True, help="Path to validation JSON file")
    parser.add_argument("--test", required=True, help="Path to test JSON file")
    parser.add_argument("--out_dir", required=True, help="Output directory for processed files")
    args = parser.parse_args()

    # === Setup ===
    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding(ENCODING)
    print(f"📂 Output directory: {args.out_dir}")

    # === Load ===
    print("\n🔹 Loading corpora...")
    train_data = load_json(args.train)
    val_data = load_json(args.val)
    test_data = load_json(args.test)

    print(f"Train docs: {len(train_data):,}")
    print(f"Val docs:   {len(val_data):,}")
    print(f"Test docs:  {len(test_data):,}")

    # === Join documents ===
    print("\n🔹 Joining documents...")
    train_docs = join_documents(train_data)
    val_docs = join_documents(val_data)
    test_docs = join_documents(test_data)

    # === Tokenize ===
    print("\n🔹 Tokenizing with GPT-2 BPE...")
    train_ids = tokenize_texts(train_docs, enc)
    val_ids = tokenize_texts(val_docs, enc)
    test_ids = tokenize_texts(test_docs, enc)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")
    print(f"Test tokens:  {len(test_ids):,}")

    # === Save bin files ===
    print("\n🔹 Saving binary token files...")
    save_bin(train_ids, os.path.join(args.out_dir, "train.bin"))
    save_bin(val_ids, os.path.join(args.out_dir, "val.bin"))
    save_bin(test_ids, os.path.join(args.out_dir, "test.bin"))

    # === Save test as readable text ===
    test_text_path = os.path.join(args.out_dir, "test.txt")
    with open(test_text_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(test_docs))
    print(f"Saved readable test set → {test_text_path}")

    # === Write log ===
    print("\n📝 Writing preparation log...")
    total_tokens = len(train_ids) + len(val_ids) + len(test_ids)

    def pct(part, whole):
        return (part / whole) * 100 if whole > 0 else 0.0

    log_path = os.path.join(args.out_dir, "prepare_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== GPT Dataset Preparation Log ===\n")
        f.write(f"Train input: {args.train}\n")
        f.write(f"Val input:   {args.val}\n")
        f.write(f"Test input:  {args.test}\n\n")
        f.write(f"Output directory: {args.out_dir}\n\n")
        f.write(f"Train docs: {len(train_docs):,}\n")
        f.write(f"Val docs:   {len(val_docs):,}\n")
        f.write(f"Test docs:  {len(test_docs):,}\n\n")
        f.write(f"Train tokens: {len(train_ids):,} ({pct(len(train_ids), total_tokens):.2f}%)\n")
        f.write(f"Val tokens:   {len(val_ids):,} ({pct(len(val_ids), total_tokens):.2f}%)\n")
        f.write(f"Test tokens:  {len(test_ids):,} ({pct(len(test_ids), total_tokens):.2f}%)\n")
        f.write(f"\nTotal tokens: {total_tokens:,}\n")

    print(f"✅ Log written to {log_path}")
    print("\n🎉 Preparation complete!")


if __name__ == "__main__":
    main()