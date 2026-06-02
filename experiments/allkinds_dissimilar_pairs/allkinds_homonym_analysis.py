"""
allkinds_homonym_analysis.py
=============================

This script identifies and compares the most dissimilar pairs of words (of all types:
monosemes, polysemes, and homonyms) in two Word2Vec embedding spaces:
TinyStories and RecipeNLG. It includes both total word frequencies and the number
of distinct documents each word appears in, providing rich information for
constructing artificial homonyms in the "all word kinds" experiment.

Pipeline Steps:
---------------
1. Load words of all types (M, P, H) from the Beekhuizen wordlist.
2. Load the preprocessed corpora (TinyStories and RecipeNLG).
3. Count how many documents each word appears in.
4. Filter words:
   - TinyStories: keep only those appearing in at least 1000 documents.
   - RecipeNLG: keep those appearing in at least 300 documents.
5. Compute all pairwise cosine similarities and select the 1000 most dissimilar pairs
   (those with the lowest cosine similarity) where word frequencies differ by less than
   a factor of 50.
6. Save extended results (word pair, cosine similarity, frequencies, document counts)
   to CSV files with `_allkinds` suffix.
7. Save a summary of statistics and top results to a .txt file for each corpus.

Input:
------
- Wordlist CSV (2-column format: word,label)
  - The label is one of {'M', 'P', 'H'}, but all are used here.
- Preprocessed corpus JSON for TinyStories
- Preprocessed corpus JSON for RecipeNLG
- Word2Vec models trained on TinyStories and RecipeNLG

Output:
-------
- Two CSV files listing the 1000 most dissimilar pairs (all word types):
  - tiny_base_dissimilar_pairs_allkinds.csv
  - recipe_base_dissimilar_pairs_allkinds.csv

- Two summary .txt files with corpus statistics:
  - tiny_base_dissimilar_pairs_allkinds_summary.txt
  - recipe_base_dissimilar_pairs_allkinds_summary.txt

Usage:
------
Run the script directly from its folder:
    $ python allkinds_homonym_analysis.py
"""

import csv
import json
import itertools
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# ==================== CONFIGURATION ====================

# Beekhuizen's wordlist (word,label format). Adjust the relative path if needed.
WORDLIST_CSV_PATH = '../data/beekhuizen_all_words.csv'

# Optional: save the full list of all words (M/P/H) to a text file for inspection.
ALLKINDS_TXT_PATH = 'allkinds_words.txt'

# TinyStories inputs
TINY_MODEL_PATH = '/home/nina/ambiguity/trained_models/word2vec_tiny_base/word2vec.model'
TINY_CORPUS_PATH = '/home/nina/ambiguity/trained_models/word2vec_tiny_base/preprocessed_corpus.json'
TINY_CSV_PATH = 'tiny_base_dissimilar_pairs_allkinds.csv'
MIN_DOC_FREQ_TINY = 1000  # TinyStories: minimum number of documents a word must appear in

# RecipeNLG inputs
RECIPE_MODEL_PATH = '/home/nina/ambiguity/trained_models/word2vec_recipe_base/word2vec.model'
RECIPE_CORPUS_PATH = '/home/nina/ambiguity/trained_models/word2vec_recipe_base/preprocessed_corpus.json'
RECIPE_CSV_PATH = 'recipe_base_dissimilar_pairs_allkinds.csv'
MIN_DOC_FREQ_RECIPE = 300  # RecipeNLG: minimum number of documents a word must appear in

# Shared config
TOP_N = 1000              # Number of most dissimilar pairs to keep per model
MAX_FREQ_RATIO = 50       # Max frequency ratio between two words in a pair (to avoid extreme imbalance)

# ==================== FUNCTIONS ====================

def extract_all_word_kinds(csv_path):
    """
    Extracts all words labeled as monosemes (M), polysemes (P), or homonyms (H)
    from the Beekhuizen wordlist CSV (word,label).
    """
    words = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
            word, label = parts
            if label in ('M', 'P', 'H'):
                words.append(word.lower())
    return words


def save_words_to_txt(words, output_path):
    """Saves the list of words to a plain text file (one per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')


def compute_document_frequencies(corpus, words):
    """
    Counts how many documents each word appears in.

    Parameters
    ----------
    corpus : list[list[str]]
        List of documents, each a list of tokens.
    words : list[str]
        Candidate words whose document frequencies should be computed.

    Returns
    -------
    dict[str, int]
        Mapping from word to the number of documents in which it appears.
    """
    doc_freq = {word: 0 for word in words}
    for doc in corpus:
        unique_tokens = set(doc)
        for word in doc_freq:
            if word in unique_tokens:
                doc_freq[word] += 1
    return doc_freq


def filter_by_doc_frequency(doc_freqs, min_df):
    """
    Filters words by minimum document frequency.

    Parameters
    ----------
    doc_freqs : dict[str, int]
        Mapping word -> document frequency.
    min_df : int
        Minimum number of documents a word must appear in.

    Returns
    -------
    list[str]
        List of words that meet the minimum document frequency.
    """
    return [word for word, count in doc_freqs.items() if count >= min_df]


def compute_dissimilar_pairs(model, words, max_ratio=MAX_FREQ_RATIO, top_n=TOP_N):
    """
    Computes cosine similarity for all word pairs that meet the frequency ratio constraint
    and returns the pairs with the lowest cosine similarity (most dissimilar).

    Parameters
    ----------
    model : gensim.models.Word2Vec
        Trained Word2Vec model.
    words : list[str]
        Words to consider as candidates for pairing (must be in the model's vocab).
    max_ratio : float
        Maximum allowed frequency ratio between the two words in a pair.
    top_n : int
        Number of most dissimilar pairs to return.

    Returns
    -------
    list[tuple[(str, str), float]]
        List of ((word1, word2), similarity) tuples, sorted by increasing similarity
        (most dissimilar first), truncated to top_n.
    """
    pairs = list(itertools.combinations(words, 2))
    results = []

    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            freq1 = model.wv.get_vecattr(w1, 'count')
            freq2 = model.wv.get_vecattr(w2, 'count')

            # Guard against division by zero, though counts should be > 0 for in-vocab words
            if freq1 == 0 or freq2 == 0:
                continue

            ratio = max(freq1, freq2) / min(freq1, freq2)
            if ratio <= max_ratio:
                sim = 1 - cosine(model.wv[w1], model.wv[w2])  # cosine similarity
                results.append(((w1, w2), sim))

    # Sort by similarity ascending: smallest similarity = most dissimilar
    results.sort(key=lambda x: x[1])
    return results[:top_n]


def save_extended_results(results, model, doc_freqs, output_path):
    """
    Saves each word pair with its similarity and frequency information to a CSV file.

    Columns:
        word1, word2, cosine_similarity,
        word1_freq, word1_docfreq, word2_freq, word2_docfreq
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'word1', 'word2', 'cosine_similarity',
            'word1_freq', 'word1_docfreq', 'word2_freq', 'word2_docfreq'
        ])
        for (w1, w2), sim in results:
            freq1 = model.wv.get_vecattr(w1, 'count')
            freq2 = model.wv.get_vecattr(w2, 'count')
            df1 = doc_freqs.get(w1, 0)
            df2 = doc_freqs.get(w2, 0)
            writer.writerow([w1, w2, round(sim, 6), freq1, df1, freq2, df2])


def print_results(title, results):
    """Prints the top 10 most dissimilar word pairs (lowest cosine similarity) for quick inspection."""
    print(f"\n=== {title} ===")
    for (w1, w2), sim in results[:10]:
        print(f"{w1} - {w2}: similarity = {sim:.4f}")


def write_summary(title, total_count, retained_count, results, output_path, min_doc_freq):
    """
    Saves summary statistics and top pairs to a .txt file.

    The summary file has the same basename as the CSV, with '_summary.txt' appended.
    """
    txt_path = output_path.replace('.csv', '_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{title} Summary (all word kinds)\n")
        f.write("=" * (len(title) + 23) + "\n\n")
        f.write(f"Total words loaded (M/P/H): {total_count}\n")
        f.write(f"Words retained (≥ {min_doc_freq} documents): {retained_count}\n")
        f.write(f"Pairs saved: {len(results)}\n\n")
        f.write("Top 10 most dissimilar pairs (lowest cosine similarity):\n")
        for (w1, w2), sim in results[:10]:
            f.write(f"  {w1} - {w2}: similarity = {sim:.4f}\n")
    print(f"Summary saved to: {txt_path}")


# ==================== MAIN WORKFLOW ====================

def main():
    # Load all word kinds (M, P, H)
    words_all = extract_all_word_kinds(WORDLIST_CSV_PATH)
    print(f"Loaded {len(words_all)} words (M/P/H) from CSV.")
    save_words_to_txt(words_all, ALLKINDS_TXT_PATH)
    print(f"Saved full word list to: {ALLKINDS_TXT_PATH}")

    # Load models
    print("\nLoading Word2Vec models...")
    model_tiny = Word2Vec.load(TINY_MODEL_PATH)
    model_recipe = Word2Vec.load(RECIPE_MODEL_PATH)

    # Load corpora
    print("Loading corpora...")
    with open(TINY_CORPUS_PATH, 'r', encoding='utf-8') as f:
        tiny_corpus = json.load(f)
    with open(RECIPE_CORPUS_PATH, 'r', encoding='utf-8') as f:
        recipe_corpus = json.load(f)

    # ========== TinyStories analysis (all word kinds) ==========
    print("\n=== TinyStories (all word kinds) ===")
    doc_freqs_tiny = compute_document_frequencies(tiny_corpus, words_all)
    words_tiny = filter_by_doc_frequency(doc_freqs_tiny, MIN_DOC_FREQ_TINY)
    print(f"{len(words_tiny)} words retained for TinyStories (≥ {MIN_DOC_FREQ_TINY} documents).")

    tiny_results = compute_dissimilar_pairs(
        model_tiny,
        words_tiny,
        max_ratio=MAX_FREQ_RATIO,
        top_n=TOP_N
    )
    save_extended_results(tiny_results, model_tiny, doc_freqs_tiny, TINY_CSV_PATH)
    print_results("TinyStories - Most Dissimilar Pairs (all word kinds)", tiny_results)
    write_summary(
        "TinyStories",
        len(words_all),
        len(words_tiny),
        tiny_results,
        TINY_CSV_PATH,
        MIN_DOC_FREQ_TINY
    )
    print(f"Results saved to: {TINY_CSV_PATH}")

    # ========== RecipeNLG analysis (all word kinds) ==========
    print("\n=== RecipeNLG (all word kinds) ===")
    doc_freqs_recipe = compute_document_frequencies(recipe_corpus, words_all)
    words_recipe = filter_by_doc_frequency(doc_freqs_recipe, MIN_DOC_FREQ_RECIPE)
    print(f"{len(words_recipe)} words retained for RecipeNLG (≥ {MIN_DOC_FREQ_RECIPE} documents).")

    recipe_results = compute_dissimilar_pairs(
        model_recipe,
        words_recipe,
        max_ratio=MAX_FREQ_RATIO,
        top_n=TOP_N
    )
    save_extended_results(recipe_results, model_recipe, doc_freqs_recipe, RECIPE_CSV_PATH)
    print_results("RecipeNLG - Most Dissimilar Pairs (all word kinds)", recipe_results)
    write_summary(
        "RecipeNLG",
        len(words_all),
        len(words_recipe),
        recipe_results,
        RECIPE_CSV_PATH,
        MIN_DOC_FREQ_RECIPE
    )
    print(f"Results saved to: {RECIPE_CSV_PATH}")


if __name__ == '__main__':
    main()