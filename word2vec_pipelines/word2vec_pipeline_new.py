import os
import json
import re
import argparse
import logging
import time
from typing import List, Tuple, Optional
from tqdm import tqdm
from gensim.models import Word2Vec

"""
word2vec_pipeline_new.py
=====================
Train a Word2Vec model on a preprocessed corpus with distinct train and val splits.

This script requires two inputs (`--train` and `--val`) and always trains on their
union (train + val). The test split, if any, is reserved for later evaluation and 
is not used here.

Expected input format
---------------------
Each JSON file is a list of documents, where each document is either:
- a string (will be tokenized via a word regex and lowercased), or
- a list of strings (tokens), which will be cleaned and lowercased.

Pipeline Steps
--------------
1. Load and preprocess the train and val corpora (JSON).
2. Combine both into a single training corpus.
3. Save the combined corpus as `preprocessed_trainval.json`.
4. Train a Word2Vec model (skip-gram with negative sampling).
5. Save the model in:
   - Gensim format (.model)
   - Binary Word2Vec format (.bin)
6. Test the model with example words (quick sanity check).
7. Write a detailed log including per-split and combined statistics.

Inputs
------
--train PATH    Path to the training split (JSON)
--val PATH      Path to the validation split (JSON)
--out_dir PATH  Output directory for model and log files

Outputs
-------
- <out_dir>/preprocessed_trainval.json
- <out_dir>/word2vec.model
- <out_dir>/word2vec.bin
- <out_dir>/training_log.txt

Usage
-----
$ python word2vec_pipeline_new.py \
    --train /path/to/train.json \
    --val   /path/to/val.json \
    --out_dir /path/to/output_folder

Example
-------
python word2vec_pipeline_new.py \
    --train ../datasets/recipe_1_HOM_train.json \
    --val   ../datasets/recipe_1_HOM_val.json \
    --out_dir ../trained_models/word2vec_recipe_1_HOM
"""

# === Small probe for a quick qualitative sanity check ===
EXAMPLE_WORDS = ['happy', 'dog', 'tree', 'run', 'oven', 'bake', 'chicken', 'blueberry']


def load_and_preprocess_corpus(path: str) -> List[List[str]]:
    """
    Load a JSON corpus where each item is either a raw string or a list of tokens.
    Returns a list of lowercase token lists.

    - If a document is a string, it is tokenized via the regex r'\\b\\w+\\b' and lowercased.
    - If a document is a list, each token is cleaned by removing non-word characters (\\W+)
      and lowercased; empty tokens are skipped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from {path}: {e}") from e

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a list at top level in {path}, got {type(raw_data)}")

    preprocessed: List[List[str]] = []
    for doc in tqdm(raw_data, desc=f"Preprocessing documents from {os.path.basename(path)}"):
        if isinstance(doc, str):
            tokens = re.findall(r'\b\w+\b', doc.lower())
        elif isinstance(doc, list):
            tokens = [re.sub(r'\W+', '', str(w).lower()) for w in doc if str(w).strip()]
            tokens = [t for t in tokens if t]  # drop empties that may result from cleanup
        else:
            # Skip unexpected types silently to match previous behavior
            continue

        if tokens:
            preprocessed.append(tokens)

    if not preprocessed:
        raise ValueError(f"After preprocessing, no documents remained in {path}. "
                         f"Check the input format or tokenization rules.")

    return preprocessed


def corpus_stats(corpus: List[List[str]]) -> Tuple[int, int, float]:
    """
    Compute (#docs, total_tokens, avg_tokens_per_doc) for a tokenized corpus.
    """
    num_docs = len(corpus)
    total_tokens = sum(len(doc) for doc in corpus)
    avg_tokens = (total_tokens / num_docs) if num_docs > 0 else 0.0
    return num_docs, total_tokens, avg_tokens


def train_word2vec(corpus: List[List[str]]) -> Word2Vec:
    """
    Train a Word2Vec model (skip-gram with negative sampling) on the provided tokenized corpus.
    The caller is responsible for providing the combined train+val corpus.
    """
    print("Training Word2Vec model with skip-gram and negative sampling...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,           # skip-gram
        negative=5,     # negative sampling
        workers=4
    )
    return model


def test_model(model: Word2Vec):
    """
    Run a small qualitative probe of most-similar words for a fixed set of example tokens.
    Returns a list of (query_word, results_or_None).
    """
    print("Testing Word2Vec model...")
    results = []
    for word in EXAMPLE_WORDS:
        if word in model.wv:
            similar = model.wv.most_similar(word)
            results.append((word, similar))
            print(f"Words similar to '{word}': {similar}")
        else:
            results.append((word, None))
            print(f"'{word}' not found in vocabulary.")
    return results


def write_log(
    log_path: str,
    train_path: str,
    val_path: str,
    out_dir: str,
    train_stats_tuple: Tuple[int, int, float],
    val_stats_tuple: Tuple[int, int, float],
    combined_stats_tuple: Tuple[int, int, float],
    model: Word2Vec,
    test_results,
    training_time_sec: float,
    combined_json_path: str
) -> None:
    """
    Write a detailed training log including per-split and combined stats (documents, total tokens,
    mean tokens per document), vocabulary size, model paths, probe results, and training time.
    """
    train_docs, train_tokens, train_avg = train_stats_tuple
    val_docs, val_tokens, val_avg = val_stats_tuple
    comb_docs, comb_tokens, comb_avg = combined_stats_tuple
    vocab_size = len(model.wv)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Word2Vec Training Log ===\n\n")
        f.write("Inputs:\n")
        f.write(f"  Train file: {train_path}\n")
        f.write(f"  Val file:   {val_path}\n")
        f.write(f"  Output dir: {out_dir}\n\n")

        f.write("Corpus Statistics (after preprocessing):\n")
        f.write(f"  Train      - documents: {train_docs:,} | total tokens: {train_tokens:,} | avg tokens/doc: {train_avg:.2f}\n")
        f.write(f"  Val        - documents: {val_docs:,}   | total tokens: {val_tokens:,}   | avg tokens/doc: {val_avg:.2f}\n")
        f.write(f"  Train+Val  - documents: {comb_docs:,}  | total tokens: {comb_tokens:,}  | avg tokens/doc: {comb_avg:.2f}\n\n")

        f.write("Artifacts:\n")
        f.write(f"  Combined corpus (train+val): {combined_json_path}\n")
        f.write(f"  Model (Gensim): {os.path.join(out_dir, 'word2vec.model')}\n")
        f.write(f"  Model (Binary): {os.path.join(out_dir, 'word2vec.bin')}\n\n")

        f.write(f"Vocabulary size: {vocab_size:,}\n\n")

        f.write("=== Probe: Most-Similar Words ===\n")
        for word, similar in test_results:
            if similar is None:
                f.write(f"{word}: not in vocabulary\n")
            else:
                sim_str = ", ".join([f"{w} ({score:.3f})" for w, score in similar])
                f.write(f"{word}: {sim_str}\n")

        f.write(f"\nTraining time: {training_time_sec:.2f} seconds\n")


def parse_args() -> argparse.Namespace:
    """
    Parse required CLI arguments. This pipeline intentionally requires --train and --val
    (no single-file mode) and always trains on train ∪ val.
    """
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on the combined train+val corpus (test is not used here)."
    )
    parser.add_argument('--train', required=True, help="Path to JSON file for the training split.")
    parser.add_argument('--val', required=True, help="Path to JSON file for the validation split.")
    parser.add_argument('--out_dir', required=True, help="Directory to save model, corpus, and logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load & preprocess both splits (use identical rules for consistency)
    train_corpus = load_and_preprocess_corpus(args.train)
    val_corpus = load_and_preprocess_corpus(args.val)

    # 2) Compute per-split stats for logging
    train_stats_tuple = corpus_stats(train_corpus)
    val_stats_tuple = corpus_stats(val_corpus)

    # 3) Combine corpora: training strictly uses train ∪ val (train first, then val)
    combined_corpus = train_corpus + val_corpus
    combined_stats_tuple = corpus_stats(combined_corpus)

    # Save the exact combined corpus used for training (reproducibility)
    combined_json_path = os.path.join(args.out_dir, 'preprocessed_trainval.json')
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_corpus, f)
    print(f"Saved combined training corpus to '{combined_json_path}'")

    # 4) Train model
    start_time = time.time()
    model = train_word2vec(combined_corpus)
    training_time = time.time() - start_time

    # 5) Save model files
    model_path = os.path.join(args.out_dir, 'word2vec.model')
    model_bin_path = os.path.join(args.out_dir, 'word2vec.bin')
    model.save(model_path)
    model.wv.save_word2vec_format(model_bin_path, binary=True)
    print(f"Model saved to '{model_path}' and '{model_bin_path}'")

    # 6) Quick sanity probe
    test_results = test_model(model)

    # 7) Detailed log (per-split + combined stats)
    log_path = os.path.join(args.out_dir, 'training_log.txt')
    write_log(
        log_path=log_path,
        train_path=args.train,
        val_path=args.val,
        out_dir=args.out_dir,
        train_stats_tuple=train_stats_tuple,
        val_stats_tuple=val_stats_tuple,
        combined_stats_tuple=combined_stats_tuple,
        model=model,
        test_results=test_results,
        training_time_sec=training_time,
        combined_json_path=combined_json_path
    )
    print(f"Wrote training log to '{log_path}'")


if __name__ == '__main__':
    main()