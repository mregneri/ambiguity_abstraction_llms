"""
04_tokenize_for_gpt.py
======================

Step 4 of the pipeline. Converts JSON corpora into binary token files
(train.bin / val.bin / test.bin) ready for GPT training.

Uses GPT-2 BPE tokenization via tiktoken.

Usage:
    # Tokenize base corpora
    python 04_tokenize_for_gpt.py

    # Tokenize one condition
    python 04_tokenize_for_gpt.py --corpus recipe --n 10 --kind HOM

    # Tokenize all conditions (run after step 3)
    python 04_tokenize_for_gpt.py --all

Output per model (inside data/processed/gpt_<corpus>_<setting>/):
    train.bin
    val.bin
    test.bin
    test.txt          ← human-readable test set
    prepare_log.txt
"""

import os
import json
import argparse
import numpy as np
import tiktoken

from config import (
    RECIPE_TRAIN, RECIPE_VAL, RECIPE_TEST,
    TINY_TRAIN,   TINY_VAL,   TINY_TEST,
    PROCESSED_DIR, CORPORA, COUNTS, KINDS,
    get_gpt_data_dir,
)

# ── Constants ─────────────────────────────────────────────────────────────────
ENC        = tiktoken.get_encoding("gpt2")
EOT_TOKEN  = "<|endoftext|>"

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_corpus(json_path: str) -> list:
    docs  = load_json(json_path)
    texts = [" ".join(doc).strip() for doc in docs if doc]
    full  = EOT_TOKEN.join(texts)
    return ENC.encode_ordinary(full), texts

def save_bin(ids: list, path: str) -> None:
    np.array(ids, dtype=np.uint16).tofile(path)

def prepare(out_dir: str, train_path: str, val_path: str, test_path: str) -> None:
    """Tokenize three splits and write .bin files + log."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Tokenizing …")
    train_ids, _    = tokenize_corpus(train_path)
    val_ids, _      = tokenize_corpus(val_path)
    test_ids, test_texts = tokenize_corpus(test_path)

    save_bin(train_ids, os.path.join(out_dir, "train.bin"))
    save_bin(val_ids,   os.path.join(out_dir, "val.bin"))
    save_bin(test_ids,  os.path.join(out_dir, "test.bin"))

    # Human-readable test
    with open(os.path.join(out_dir, "test.txt"), 'w', encoding='utf-8') as f:
        f.write("\n\n".join(test_texts))

    total = len(train_ids) + len(val_ids) + len(test_ids)
    pct   = lambda p: (p / total * 100) if total > 0 else 0.0

    with open(os.path.join(out_dir, "prepare_log.txt"), 'w') as f:
        f.write("=== GPT Dataset Preparation Log ===\n")
        f.write(f"Train input:  {train_path}\n")
        f.write(f"Val input:    {val_path}\n")
        f.write(f"Test input:   {test_path}\n")
        f.write(f"Output:       {out_dir}\n\n")
        f.write(f"Train tokens: {len(train_ids):,} ({pct(len(train_ids)):.2f}%)\n")
        f.write(f"Val tokens:   {len(val_ids):,} ({pct(len(val_ids)):.2f}%)\n")
        f.write(f"Test tokens:  {len(test_ids):,} ({pct(len(test_ids)):.2f}%)\n")
        f.write(f"Total tokens: {total:,}\n")

    print(f"  ✅ {len(train_ids):,} train / {len(val_ids):,} val / {len(test_ids):,} test tokens → {out_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n", type=int)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--all", action="store_true",
                        help="Tokenize all condition corpora (run after step 3)")
    args = parser.parse_args()

    def base_paths(corpus):
        if corpus == "recipe":
            return RECIPE_TRAIN, RECIPE_VAL, RECIPE_TEST
        return TINY_TRAIN, TINY_VAL, TINY_TEST

    def condition_paths(corpus, n, kind):
        prefix = os.path.join(PROCESSED_DIR, f"{corpus}_{n}_{kind}")
        return (prefix + "_train.json",
                prefix + "_val.json",
                prefix + "_test.json")

    if args.all:
        # Base
        for corpus in CORPORA:
            print(f"\n── gpt_{corpus}_base ──────────────────────────────────")
            prepare(get_gpt_data_dir(corpus, "base"), *base_paths(corpus))
        # Conditions
        for corpus in CORPORA:
            for n in COUNTS:
                for kind in KINDS:
                    tp, vp, tep = condition_paths(corpus, n, kind)
                    if not os.path.exists(tp):
                        continue
                    print(f"\n── gpt_{corpus}_{n}_{kind} ──────────────────────────")
                    prepare(get_gpt_data_dir(corpus, str(n), kind), tp, vp, tep)

    elif args.corpus and args.n and args.kind:
        corpus, n, kind = args.corpus, args.n, args.kind
        print(f"\n── gpt_{corpus}_{n}_{kind} ──────────────────────────")
        prepare(get_gpt_data_dir(corpus, str(n), kind),
                *condition_paths(corpus, n, kind))

    else:
        # Default: base corpora only
        for corpus in CORPORA:
            print(f"\n── gpt_{corpus}_base ──────────────────────────────────")
            prepare(get_gpt_data_dir(corpus, "base"), *base_paths(corpus))

    print("\n🎉 GPT tokenization complete.")

if __name__ == "__main__":
    main()
