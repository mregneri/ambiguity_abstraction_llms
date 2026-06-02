"""
03_prepare_corpora.py
=====================

Step 3 of the pipeline. For each experimental condition, inserts pseudowords
(artificial homonyms or hypernyms) into the base corpora and saves the
modified train/val/test splits.

For HOM conditions:  two semantically DISSIMILAR words → merged pseudoword
For HYP conditions:  two semantically SIMILAR words    → merged pseudoword
_HALF variants:      only one meaning's occurrences are replaced

Usage:
    # All conditions (runs a while)
    python 03_prepare_corpora.py

    # Single condition
    python 03_prepare_corpora.py --corpus recipe --n 10 --kind HOM

Output (written to data/processed/):
    {corpus}_{n}_{kind}_train.json
    {corpus}_{n}_{kind}_val.json
    {corpus}_{n}_{kind}_test.json
    {corpus}_{n}_{kind}_trainval.json   ← train+val combined for word2vec

Also writes per-condition summary logs alongside the output files.
"""

import os
import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from config import (
    RECIPE_TRAIN, RECIPE_VAL, RECIPE_TEST,
    TINY_TRAIN,   TINY_VAL,   TINY_TEST,
    PROCESSED_DIR, CORPORA, COUNTS, KINDS,
    get_pseudoword_map, get_inflection_map,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():   return replacement.upper()
    if original.istitle():   return replacement.capitalize()
    return replacement

def build_inflection_lookup(master_map: dict, replacements_map: dict, half: bool) -> dict:
    """
    Build {form.lower(): (base, pseudoword)} from the master inflection map
    and the per-experiment replacement map.

    For _HALF conditions, only the FIRST component word is replaced.
    """
    lookup = {}
    for pair, pseudoword in replacements_map.items():
        words = [w.strip() for w in pair.split(',')]
        if half:
            words = words[:1]   # only replace the first component

        entry = master_map.get(pseudoword)
        if not entry:
            print(f"  ⚠️  '{pseudoword}' not in master map — skipping")
            continue
        for base in words:
            for form in entry["forms"].get(base, []):
                lookup[form.lower()] = (base, pseudoword)
    return lookup

def replace_in_corpus(corpus: list, lookup: dict):
    """Replace word forms in-place, preserving punctuation and casing."""
    modified, counts = [], defaultdict(int)
    for doc in tqdm(corpus, desc="  Replacing", leave=False):
        new_doc = []
        for token in doc:
            parts = re.findall(r'\w+|[^\w\s]', token, re.UNICODE)
            new_parts = []
            for part in parts:
                if re.fullmatch(r'\w+', part):
                    key = part.lower()
                    if key in lookup:
                        base, pseudo = lookup[key]
                        new_parts.append(preserve_case(part, pseudo))
                        counts[base] += 1
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            new_doc.append(''.join(new_parts))
        modified.append(new_doc)
    return modified, counts

# ── Per-condition processing ──────────────────────────────────────────────────
def process_condition(corpus: str, n: int, kind: str) -> None:
    """Prepare one experimental condition and save all three splits."""
    half      = kind.endswith("_HALF")
    base_kind = kind.replace("_HALF", "")  # HOM or HYP

    # Paths
    base_train = RECIPE_TRAIN if corpus == "recipe" else TINY_TRAIN
    base_val   = RECIPE_VAL   if corpus == "recipe" else TINY_VAL
    base_test  = RECIPE_TEST  if corpus == "recipe" else TINY_TEST

    map_path        = get_pseudoword_map(corpus, n, kind)
    inflection_path = get_inflection_map(corpus, base_kind)

    prefix    = os.path.join(PROCESSED_DIR, f"{corpus}_{n}_{kind}")
    train_out = prefix + "_train.json"
    val_out   = prefix + "_val.json"
    test_out  = prefix + "_test.json"
    trainval_out = prefix + "_trainval.json"
    log_out   = prefix + "_log.txt"

    print(f"\n  ── {corpus}_{n}_{kind} ──")

    repl_map   = load_json(map_path)
    master_map = load_json(inflection_path)
    lookup     = build_inflection_lookup(master_map, repl_map, half)
    print(f"  {len(lookup)} word forms → {len(repl_map)} pseudowords")

    all_counts = {}
    for split_name, base_path, out_path in [
        ("train", base_train, train_out),
        ("val",   base_val,   val_out),
        ("test",  base_test,  test_out),
    ]:
        corpus_data          = load_json(base_path)
        modified, counts     = replace_in_corpus(corpus_data, lookup)
        save_json(modified, out_path)
        all_counts[split_name] = counts

    # trainval = train + val (for word2vec)
    train_data = load_json(train_out)
    val_data   = load_json(val_out)
    save_json(train_data + val_data, trainval_out)

    # Write log
    with open(log_out, 'w') as f:
        f.write(f"=== Corpus Preparation Log: {corpus}_{n}_{kind} ===\n\n")
        f.write(f"Map:          {map_path}\n")
        f.write(f"Inflections:  {inflection_path}\n")
        f.write(f"HALF mode:    {half}\n\n")
        for pair, pseudo in repl_map.items():
            words = [w.strip() for w in pair.split(',')]
            f.write(f"{', '.join(words)} → {pseudo}\n")
            for sp in ("train", "val", "test"):
                cnt = sum(all_counts[sp].get(w, 0) for w in words)
                f.write(f"  {sp}: {cnt} replacements\n")
            f.write("\n")
    print(f"  ✅ Done — log: {log_out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--n", type=int)
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if args.corpus and args.n and args.kind:
        targets = [(args.corpus, args.n, args.kind)]
    else:
        targets = [(c, n, k) for c in CORPORA for n in COUNTS for k in KINDS]

    for corpus, n, kind in targets:
        map_path = get_pseudoword_map(corpus, n, kind)
        if not os.path.exists(map_path):
            print(f"  ⚠️  Map not found for {corpus}_{n}_{kind} — skipping")
            continue
        process_condition(corpus, n, kind)

    print("\n🎉 Corpus preparation complete.")

if __name__ == "__main__":
    main()
