"""
monoseme_hypernym_analysis.py
==============================

This script identifies and compares the most similar pairs of monosemes in two Word2Vec embedding spaces:
TinyStories and RecipeNLG. It includes both total word frequencies and the number of distinct documents
each word appears in, providing rich information for constructing artificial hypernyms.

Pipeline Steps:
---------------
1. Load monosemes from a wordlist (labeled "M" in CSV).
2. Load the preprocessed corpora (TinyStories and RecipeNLG).
3. Count how many documents each word appears in.
4. Filter monosemes:
   - TinyStories: keep only those appearing in at least 1000 documents.
   - RecipeNLG: keep those appearing in at least 300 documents.
5. Compute all pairwise cosine similarities and select the 1000 **most similar** pairs
   where word frequencies differ by less than a factor of 50.
6. Save extended results (word pair, cosine similarity, frequencies, document counts) to CSV files.
7. Save a summary of statistics and top results to a .txt file for each corpus.

Input:
------
- Wordlist CSV (2-column format: word, category)
- Preprocessed corpus JSON for TinyStories
- Preprocessed corpus JSON for RecipeNLG
- Word2Vec models trained on TinyStories and RecipeNLG

Output:
-------
- Two CSV files listing the 1000 most similar monoseme pairs:
  - base_tiny_similar_pairs.csv
  - base_recipe_similar_pairs.csv

- Two summary .txt files with corpus statistics:
  - base_tiny_similar_pairs_summary.txt
  - base_recipe_similar_pairs_summary.txt

Usage:
------
Run the script directly:
$ python monoseme_hypernym_analysis.py
"""

import csv
import json
import itertools
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# ==================== CONFIGURATION ====================

WORDLIST_CSV_PATH = '../data/beekhuizen_all_words.csv'  # Beehuizen's wordlist (word,label format)
MONOSEME_TXT_PATH = 'monosemes_only.txt'     # Output: plain text list of monosemes (one per line)

# TinyStories inputs
TINY_MODEL_PATH = '/home/nina/ambiguity/trained_models/word2vec_tiny_base/word2vec.model'
TINY_CORPUS_PATH = '/home/nina/ambiguity/trained_models/word2vec_tiny_base/preprocessed_corpus.json'
TINY_CSV_PATH = 'tiny_base_similar_pairs.csv'
MIN_DOC_FREQ_TINY = 1000  # TinyStories: minimum number of documents a monoseme must appear in

# RecipeNLG inputs
RECIPE_MODEL_PATH = '/home/nina/ambiguity/trained_models/word2vec_recipe_base/word2vec.model'
RECIPE_CORPUS_PATH = '/home/nina/ambiguity/trained_models/word2vec_recipe_base/preprocessed_corpus.json'
RECIPE_CSV_PATH = 'recipe_base_similar_pairs.csv'
MIN_DOC_FREQ_RECIPE = 300  # RecipeNLG: minimum number of documents a monoseme must appear in

# Shared config
TOP_N = 1000              # Number of most similar pairs to keep per model
MAX_FREQ_RATIO = 50       # Max frequency ratio between two words in a pair (to avoid extreme imbalance)

# ==================== FUNCTIONS ====================

def extract_monosemes(csv_path):
    """Extracts monosemes (category 'M') from the wordlist CSV."""
    monosemes = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2 and parts[1] == 'M':
                monosemes.append(parts[0].lower())
    return monosemes

def save_monosemes_to_txt(monosemes, output_path):
    """Saves the list of monosemes to a plain text file (one per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(monosemes):
            f.write(word + '\n')

def compute_document_frequencies(corpus, words):
    """Counts how many documents (recipes or stories) each word appears in."""
    doc_freq = {word: 0 for word in words}
    for doc in corpus:
        unique_tokens = set(doc)
        for word in doc_freq:
            if word in unique_tokens:
                doc_freq[word] += 1
    return doc_freq

def filter_by_doc_frequency(doc_freqs, min_df):
    """Filters words by minimum document frequency."""
    return [word for word, count in doc_freqs.items() if count >= min_df]

def compute_similar_pairs(model, words, doc_freqs, max_ratio=MAX_FREQ_RATIO, top_n=TOP_N):
    """Computes cosine similarity for all word pairs that meet frequency ratio constraint."""
    pairs = list(itertools.combinations(words, 2))
    results = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            freq1 = model.wv.get_vecattr(w1, 'count')
            freq2 = model.wv.get_vecattr(w2, 'count')
            ratio = max(freq1, freq2) / min(freq1, freq2)
            if ratio <= max_ratio:
                sim = 1 - cosine(model.wv[w1], model.wv[w2])
                results.append(((w1, w2), sim))
    results.sort(key=lambda x: x[1], reverse=True)  # Most similar first
    return results[:top_n]

def save_extended_results(results, model, doc_freqs, output_path):
    """Saves each word pair with its similarity and frequency information to a CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'word1', 'word2', 'cosine_similarity',
            'word1_freq', 'word1_docfreq', 'word2_freq', 'word2_docfreq'])
        for (w1, w2), sim in results:
            freq1 = model.wv.get_vecattr(w1, 'count')
            freq2 = model.wv.get_vecattr(w2, 'count')
            df1 = doc_freqs.get(w1, 0)
            df2 = doc_freqs.get(w2, 0)
            writer.writerow([w1, w2, round(sim, 6), freq1, df1, freq2, df2])

def print_results(title, results):
    """Prints the top 10 most similar word pairs for quick inspection."""
    print(f"\n=== {title} ===")
    for (w1, w2), sim in results[:10]:
        print(f"{w1} - {w2}: similarity = {sim:.4f}")

def write_summary(title, monoseme_count, retained_count, results, output_path, min_doc_freq):
    """Saves summary statistics and top pairs to a .txt file."""
    txt_path = output_path.replace('.csv', '_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{title} Summary\n")
        f.write("=" * (len(title) + 8) + "\n\n")
        f.write(f"Total monosemes loaded: {monoseme_count}\n")
        f.write(f"Monosemes retained (≥ {min_doc_freq} documents): {retained_count}\n")
        f.write(f"Pairs saved: {len(results)}\n\n")
        f.write("Top 10 most similar pairs:\n")
        for (w1, w2), sim in results[:10]:
            f.write(f"  {w1} - {w2}: similarity = {sim:.4f}\n")
    print(f"Summary saved to: {txt_path}")

# ==================== MAIN WORKFLOW ====================

def main():
    # Load monosemes
    monosemes = extract_monosemes(WORDLIST_CSV_PATH)
    print(f"Loaded {len(monosemes)} monosemes from CSV.")
    save_monosemes_to_txt(monosemes, MONOSEME_TXT_PATH)

    # Load models
    print("Loading Word2Vec models...")
    model_tiny = Word2Vec.load(TINY_MODEL_PATH)
    model_recipe = Word2Vec.load(RECIPE_MODEL_PATH)

    # Load corpora
    with open(TINY_CORPUS_PATH, 'r') as f:
        tiny_corpus = json.load(f)
    with open(RECIPE_CORPUS_PATH, 'r') as f:
        recipe_corpus = json.load(f)

    # TinyStories analysis
    doc_freqs_tiny = compute_document_frequencies(tiny_corpus, monosemes)
    monosemes_tiny = filter_by_doc_frequency(doc_freqs_tiny, MIN_DOC_FREQ_TINY)
    print(f"{len(monosemes_tiny)} monosemes retained for TinyStories (≥ {MIN_DOC_FREQ_TINY} documents).")
    tiny_results = compute_similar_pairs(model_tiny, monosemes_tiny, doc_freqs_tiny, top_n=TOP_N)
    save_extended_results(tiny_results, model_tiny, doc_freqs_tiny, TINY_CSV_PATH)
    print_results("TinyStories - Most Similar Monoseme Pairs", tiny_results)
    write_summary("TinyStories", len(monosemes), len(monosemes_tiny), tiny_results, TINY_CSV_PATH, MIN_DOC_FREQ_TINY)
    print(f"Results saved to: {TINY_CSV_PATH}")

    # RecipeNLG analysis
    doc_freqs_recipe = compute_document_frequencies(recipe_corpus, monosemes)
    monosemes_recipe = filter_by_doc_frequency(doc_freqs_recipe, MIN_DOC_FREQ_RECIPE)
    print(f"{len(monosemes_recipe)} monosemes retained for RecipeNLG (≥ {MIN_DOC_FREQ_RECIPE} documents).")
    recipe_results = compute_similar_pairs(model_recipe, monosemes_recipe, doc_freqs_recipe, top_n=TOP_N)
    save_extended_results(recipe_results, model_recipe, doc_freqs_recipe, RECIPE_CSV_PATH)
    print_results("RecipeNLG - Most Similar Monoseme Pairs", recipe_results)
    write_summary("RecipeNLG", len(monosemes), len(monosemes_recipe), recipe_results, RECIPE_CSV_PATH, MIN_DOC_FREQ_RECIPE)
    print(f"Results saved to: {RECIPE_CSV_PATH}")

if __name__ == '__main__':
    main()