import os
import json
import re
import argparse
import logging
import time
from tqdm import tqdm
from gensim.models import Word2Vec

"""
word2vec_pipeline.py
=====================
This script trains a Word2Vec model on a preprocessed text corpus in JSON format.

Each document in the input corpus should be a list of tokens (i.e., words or subwords),
and the corpus should be saved as a JSON file (e.g., one document = one recipe or story).

Pipeline Steps:
---------------
1. Load a pre-tokenized corpus from a JSON file (list of token lists).
2. Optionally re-tokenize with lowercasing and word-based regex, if not already tokenized.
3. Save the final preprocessed corpus to the output folder for reference.
4. Train a Word2Vec model using skip-gram and negative sampling.
5. Save the model in:
   - Gensim format (.model)
   - Binary Word2Vec format (.bin)
6. Test the model by printing similar words for a few examples.
7. Write all key info to a log file in the output folder.

Input:
------
- A JSON file containing a list of documents (strings or token lists).

Output:
-------
- <out_dir>/preprocessed_corpus.json: cleaned, tokenized documents
- <out_dir>/word2vec.model: trained model in Gensim format
- <out_dir>/word2vec.bin: trained model in binary Word2Vec format
- <out_dir>/training_log.txt: summary of training and evaluation

Usage:
------
$ python word2vec_pipeline.py \
    --input path/to/preprocessed_corpus.json \
    --out_dir path/to/output_folder

Example:
--------
$ python word2vec_pipeline.py --input ../datasets/recipe_base_trainval.json --out_dir ../trained_models/word2vec_recipe_base
"""

# === Parameters ===
EXAMPLE_WORDS = ['happy', 'dog', 'tree', 'run', 'oven', 'bake', 'chicken', 'blueberry']

def load_and_preprocess_corpus(input_path):
    """Loads and ensures each document is tokenized into a list of lowercase words."""
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    preprocessed = []
    for doc in tqdm(raw_data, desc="Preprocessing documents"):
        if isinstance(doc, str):
            tokens = re.findall(r'\b\w+\b', doc.lower())
        elif isinstance(doc, list):
            tokens = [re.sub(r'\W+', '', w.lower()) for w in doc if w.strip()]
        else:
            continue
        if tokens:
            preprocessed.append(tokens)

    return preprocessed

def train_word2vec(corpus):
    """Trains and returns a Word2Vec model."""
    print("Training Word2Vec model with skip-gram and negative sampling...")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,
        negative=5,
        workers=4
    )
    return model

def test_model(model):
    """Returns similar words for selected examples."""
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

def write_log(log_path, input_path, out_dir, corpus, model, test_results, training_time_sec):
    total_tokens = sum(len(doc) for doc in corpus)
    avg_tokens = total_tokens / len(corpus)
    vocab_size = len(model.wv)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=== Word2Vec Training Log ===\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output folder: {out_dir}\n\n")
        f.write(f"Documents: {len(corpus):,}\n")
        f.write(f"Total tokens: {total_tokens:,}\n")
        f.write(f"Average tokens per document: {avg_tokens:.2f}\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n\n")
        f.write(f"Model (Gensim): {os.path.join(out_dir, 'word2vec.model')}\n")
        f.write(f"Model (Binary): {os.path.join(out_dir, 'word2vec.bin')}\n\n")
        f.write("=== Test Words ===\n")
        for word, similar in test_results:
            if similar is None:
                f.write(f"{word}: not in vocabulary\n")
            else:
                sim_str = ", ".join([f"{w} ({score:.3f})" for w, score in similar])
                f.write(f"{word}: {sim_str}\n")
        f.write(f"\nTraining time: {training_time_sec:.2f} seconds\n")

def main():
    parser = argparse.ArgumentParser(description="Train Word2Vec on a tokenized corpus.")
    parser.add_argument('--input', required=True, help="Path to JSON file with pre-tokenized documents.")
    parser.add_argument('--out_dir', required=True, help="Directory to save model and processed corpus.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load and preprocess
    corpus = load_and_preprocess_corpus(args.input)

    # Save preprocessed corpus
    preprocessed_path = os.path.join(args.out_dir, 'preprocessed_corpus.json')
    with open(preprocessed_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f)
    print(f"Saved preprocessed corpus to '{preprocessed_path}'")

    # Train model
    start_time = time.time()
    model = train_word2vec(corpus)
    training_time = time.time() - start_time

    # Save model
    model_path = os.path.join(args.out_dir, 'word2vec.model')
    model_bin_path = os.path.join(args.out_dir, 'word2vec.bin')
    model.save(model_path)
    model.wv.save_word2vec_format(model_bin_path, binary=True)
    print(f"Model saved to '{model_path}' and '{model_bin_path}'")

    # Test and log
    test_results = test_model(model)
    log_path = os.path.join(args.out_dir, 'training_log.txt')
    write_log(log_path, args.input, args.out_dir, corpus, model, test_results, training_time)
    print(f"Wrote training log to '{log_path}'")

if __name__ == '__main__':
    main()