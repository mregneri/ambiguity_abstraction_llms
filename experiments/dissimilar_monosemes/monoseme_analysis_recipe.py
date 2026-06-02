"""
monoseme_analysis_recipe.py
==============================

This script analyzes the most dissimilar monoseme pairs in the RecipeNLG Word2Vec embedding space.
It filters monosemes by document frequency and selects the most dissimilar pairs under a frequency ratio constraint.
Then, it selects a subset of non-overlapping pairs by prioritizing monosemes with the fewest available options.

Output includes:
- A CSV file with all top dissimilar candidate pairs.
- A CSV file with the final non-overlapping subset of pairs.
- A summary text file with key stats and a preview of the top results.

Usage:
$ python monoseme_analysis_recipe.py
"""

import csv
import json
import itertools
from collections import defaultdict
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# ==================== CONFIGURATION ====================

WORDLIST_CSV_PATH = '../data/all_words.csv'  # CSV with all words and their labels (H, M, P)
MONOSEME_TX_PATH = 'monosemes_only.txt'      # Output: plain text list of monosemes

RECIPE_MODEL_PATH = '/home/nina/ambiguity/word2vec_pipelines/word2vec_base_recipe/word2vec.model'
RECIPE_CORPUS_PATH = '/home/nina/ambiguity/word2vec_pipelines/word2vec_base_recipe/preprocessed_corpus.json'
BASE_PAIR_CSV_PATH = 'prelim_recipe_candidate_pairs.csv'      # Output: all dissimilar candidate pairs
FINAL_PAIR_CSV_PATH = 'prelim_pairs_recipe.csv'               # Output: selected non-overlapping pairs

TOP_N = 1000               # Number of most dissimilar pairs to keep
MIN_DOC_FREQ = 1000         # Minimum documents a monoseme must appear in
MAX_FREQ_RATIO = 50        # Max frequency ratio between words in a pair

# ==================== MONOSEME ANALYSIS ====================

def extract_monosemes(csv_path):
    monosemes = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 2 and parts[1] == 'M':
                monosemes.append(parts[0].lower())
    return monosemes

def save_monosemes_to_txt(monosemes, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(monosemes):
            f.write(word + '\n')

def compute_document_frequencies(corpus, words):
    doc_freq = {word: 0 for word in words}
    for doc in corpus:
        unique_tokens = set(doc)
        for word in doc_freq:
            if word in unique_tokens:
                doc_freq[word] += 1
    return doc_freq

def filter_by_doc_frequency(doc_freqs, min_df):
    return [word for word, count in doc_freqs.items() if count >= min_df]

def compute_dissimilar_pairs(model, words, doc_freqs):
    pairs = list(itertools.combinations(words, 2))
    results = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            freq1 = model.wv.get_vecattr(w1, 'count')
            freq2 = model.wv.get_vecattr(w2, 'count')
            ratio = max(freq1, freq2) / min(freq1, freq2)
            if ratio <= MAX_FREQ_RATIO:
                sim = 1 - cosine(model.wv[w1], model.wv[w2])
                results.append(((w1, w2), sim))
    results.sort(key=lambda x: x[1])
    return results[:TOP_N]

def save_extended_results(results, model, doc_freqs, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'word1', 'word2', 'cosine_similarity',
            'word1_freq', 'word1_docfreq', 'word2_freq', 'word2_docfreq'])
        for (w1, w2), sim in results:
            writer.writerow([
                w1, w2, round(sim, 6),
                model.wv.get_vecattr(w1, 'count'), doc_freqs.get(w1, 0),
                model.wv.get_vecattr(w2, 'count'), doc_freqs.get(w2, 0)
            ])

def write_summary(monoseme_count, retained_count, results, output_path):
    txt_path = output_path.replace('.csv', '_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("RecipeNLG Summary\n")
        f.write("==================\n\n")
        f.write(f"Total monosemes loaded: {monoseme_count}\n")
        f.write(f"Monosemes retained (≥ {MIN_DOC_FREQ} documents): {retained_count}\n")
        f.write(f"Pairs saved: {len(results)}\n\n")
        f.write("Top 10 most dissimilar pairs:\n")
        for (w1, w2), sim in results[:10]:
            f.write(f"  {w1} - {w2}: similarity = {sim:.4f}\n")
    print(f"Summary saved to: {txt_path}")

# ==================== SELECTION STEP ====================

def load_pairs_for_selection(csv_path):
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['cosine_similarity'] = float(row['cosine_similarity'])
            row['word1_freq'] = int(row['word1_freq'])
            row['word1_docfreq'] = int(row['word1_docfreq'])
            row['word2_freq'] = int(row['word2_freq'])
            row['word2_docfreq'] = int(row['word2_docfreq'])
            pairs.append(row)
    return pairs

def build_monoseme_counts(pairs):
    counts = defaultdict(int)
    for row in pairs:
        counts[row['word1']] += 1
        counts[row['word2']] += 1
    return counts

def build_pair_lookup(pairs):
    lookup = defaultdict(list)
    for row in pairs:
        lookup[row['word1']].append(row)
        lookup[row['word2']].append(row)
    return lookup

def select_non_overlapping_pairs(pairs):
    monoseme_counts = build_monoseme_counts(pairs)
    pair_lookup = build_pair_lookup(pairs)
    sorted_monosemes = sorted(monoseme_counts, key=lambda w: monoseme_counts[w])

    used_words = set()
    selected_pairs = []

    for monoseme in sorted_monosemes:
        if monoseme in used_words:
            continue

        candidate_pairs = pair_lookup[monoseme]
        valid_pairs = [p for p in candidate_pairs if p['word1'] not in used_words and p['word2'] not in used_words]

        if not valid_pairs:
            continue

        best_pair = min(valid_pairs, key=lambda p: p['cosine_similarity'])
        selected_pairs.append(best_pair)
        used_words.add(best_pair['word1'])
        used_words.add(best_pair['word2'])

    return selected_pairs

def save_selected_pairs(pairs, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word1', 'word2', 'cosine_similarity',
            'word1_freq', 'word1_docfreq', 'word2_freq', 'word2_docfreq'
        ])
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)

# ==================== MAIN ====================

def main():
    # Load monosemes
    monosemes = extract_monosemes(WORDLIST_CSV_PATH)
    print(f"Loaded {len(monosemes)} monosemes from CSV.")
    save_monosemes_to_txt(monosemes, MONOSEME_TX_PATH)

    # Load and filter corpus
    with open(RECIPE_CORPUS_PATH, 'r') as f:
        recipe_corpus = json.load(f)

    doc_freqs = compute_document_frequencies(recipe_corpus, monosemes)
    monosemes_filtered = filter_by_doc_frequency(doc_freqs, MIN_DOC_FREQ)
    print(f"{len(monosemes_filtered)} monosemes retained for RecipeNLG (≥ {MIN_DOC_FREQ} documents).")

    print("Loading Word2Vec model...")
    model = Word2Vec.load(RECIPE_MODEL_PATH)

    # Dissimilarity computation
    results = compute_dissimilar_pairs(model, monosemes_filtered, doc_freqs)
    save_extended_results(results, model, doc_freqs, BASE_PAIR_CSV_PATH)
    write_summary(len(monosemes), len(monosemes_filtered), results, BASE_PAIR_CSV_PATH)
    print(f"Candidate pair results saved to: {BASE_PAIR_CSV_PATH}")

    # Final pair selection
    print("\nSelecting non-overlapping final pairs...")
    all_pairs = load_pairs_for_selection(BASE_PAIR_CSV_PATH)
    selected = select_non_overlapping_pairs(all_pairs)
    save_selected_pairs(selected, FINAL_PAIR_CSV_PATH)
    print(f"Final selected pairs: {len(selected)}")
    print(f"Saved to: {FINAL_PAIR_CSV_PATH}")

if __name__ == '__main__':
    main()
