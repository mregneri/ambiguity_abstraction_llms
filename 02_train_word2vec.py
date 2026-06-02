"""
02_train_word2vec.py
=====================

Step 2 of the pipeline. Trains Word2Vec models on base corpora and, later,
on each modified corpus produced in step 3. Call this script once after step 1
(for base models) and again after step 3 (for condition-specific models).

Usage:
    # Train base models only
    python 02_train_word2vec.py

    # Train a specific model (e.g. after corpus prep)
    python 02_train_word2vec.py --corpus recipe --n 10 --kind HOM

    # Train all condition models (batch, run after step 3)
    python 02_train_word2vec.py --all

Output per model:
    trained_models/word2vec_<corpus>_<setting>/
        word2vec.model        ← Gensim format
        word2vec.bin          ← binary Word2Vec format
        training_log.txt
"""

import os
import re
import json
import time
import logging
import argparse
from tqdm import tqdm
from gensim.models import Word2Vec

from config import (
    RECIPE_TRAINVAL, TINY_TRAINVAL, PROCESSED_DIR,
    TRAINED_MODELS_DIR, CORPORA, COUNTS, KINDS,
    get_word2vec_model_dir,
)

# ── Parameters ────────────────────────────────────────────────────────────────
W2V_PARAMS = dict(vector_size=100, window=5, min_count=5, sg=1, negative=5, workers=4)
EXAMPLE_WORDS = ['happy', 'dog', 'tree', 'run', 'oven', 'bake', 'chicken', 'blueberry']

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_corpus(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    corpus = []
    for doc in tqdm(raw, desc="Preprocessing", leave=False):
        if isinstance(doc, str):
            tokens = re.findall(r'\b\w+\b', doc.lower())
        elif isinstance(doc, list):
            tokens = [re.sub(r'\W+', '', w.lower()) for w in doc if w.strip()]
        else:
            continue
        if tokens:
            corpus.append(tokens)
    return corpus

def train_and_save(input_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Loading corpus from {input_path}")
    corpus = load_corpus(input_path)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.WARNING)
    print(f"  Training Word2Vec on {len(corpus):,} docs …")
    t0 = time.time()
    model = Word2Vec(sentences=corpus, **W2V_PARAMS)
    elapsed = time.time() - t0

    model.save(os.path.join(out_dir, 'word2vec.model'))
    model.wv.save_word2vec_format(os.path.join(out_dir, 'word2vec.bin'), binary=True)

    # Write log
    total_tokens = sum(len(d) for d in corpus)
    with open(os.path.join(out_dir, 'training_log.txt'), 'w') as f:
        f.write("=== Word2Vec Training Log ===\n")
        f.write(f"Input:       {input_path}\n")
        f.write(f"Output:      {out_dir}\n")
        f.write(f"Documents:   {len(corpus):,}\n")
        f.write(f"Tokens:      {total_tokens:,}\n")
        f.write(f"Vocab size:  {len(model.wv):,}\n")
        f.write(f"Train time:  {elapsed:.1f}s\n\n")
        f.write("=== Example neighbours ===\n")
        for word in EXAMPLE_WORDS:
            if word in model.wv:
                sim = ", ".join(f"{w}({s:.3f})" for w, s in model.wv.most_similar(word))
                f.write(f"  {word}: {sim}\n")
            else:
                f.write(f"  {word}: not in vocabulary\n")

    print(f"  ✅ Saved to {out_dir}  ({elapsed:.1f}s)")

def corpus_input_path(corpus: str, n_or_base, kind: str = None) -> str:
    """Return the trainval JSON used to train a word2vec model."""
    if n_or_base == "base":
        return RECIPE_TRAINVAL if corpus == "recipe" else TINY_TRAINVAL
    # Modified corpus — trainval is train+val concatenated from step 3
    name = f"{corpus}_{n_or_base}_{kind}_trainval.json"
    return os.path.join(PROCESSED_DIR, name)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA, help="Corpus name")
    parser.add_argument("--n", type=int, help="Number of pseudoword pairs")
    parser.add_argument("--kind", choices=KINDS, help="Condition kind")
    parser.add_argument("--all", action="store_true",
                        help="Train all condition models (run after step 3)")
    args = parser.parse_args()

    if args.all:
        targets = [("base",)] + [
            (c, n, k) for c in CORPORA for n in COUNTS for k in KINDS
        ]
    elif args.corpus and args.n and args.kind:
        targets = [(args.corpus, args.n, args.kind)]
    else:
        # Default: base models only
        targets = [("base",)]

    for t in targets:
        if t == ("base",):
            for corpus in CORPORA:
                print(f"\n── word2vec_{corpus}_base ──────────────────────────")
                inp = RECIPE_TRAINVAL if corpus == "recipe" else TINY_TRAINVAL
                out = get_word2vec_model_dir(corpus, "base")
                train_and_save(inp, out)
        else:
            corpus, n, kind = t
            print(f"\n── word2vec_{corpus}_{n}_{kind} ──────────────────────")
            inp = corpus_input_path(corpus, n, kind)
            if not os.path.exists(inp):
                print(f"  ⚠️  Input not found: {inp} — skipping")
                continue
            out = get_word2vec_model_dir(corpus, str(n), kind)
            train_and_save(inp, out)

    print("\n🎉 Word2Vec training complete.")

if __name__ == "__main__":
    main()
