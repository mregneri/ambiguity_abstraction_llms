"""
insert_artificial_hypernyms_half.py
===================================

This script inserts artificial hypernyms into tokenized training, validation,
and test corpora — but only replaces **every other occurrence** of each original word
(≈50%). The alternation is fully deterministic and repeatable.

It supports multiple hypernym pairs and uses pre-generated, manually verified
master inflection maps instead of calling `lemminflect` at runtime. Each comma-separated
word pair is expanded to include all its inflected forms, and replacements occur only
on full-word matches after punctuation is removed and later restored.

Expected Config Format (Python)
-------------------------------
config = {
    'train_in': 'path/to/train.json',
    'val_in': 'path/to/val.json',
    'test_in': 'path/to/test.json',
    'train_out': 'path/to/output_train.json',
    'val_out': 'path/to/output_val.json',
    'test_out': 'path/to/output_test.json',
    'replacements': 'path/to/hypernym_map.json',
    'master_inflections': 'path/to/master_inflections.json',
    'summary_log': 'path/to/summary_log.txt'
}

Usage
-----
python insert_artificial_hypernyms_half.py --config config/insert_recipe_50_HYP_HALF.py

Output
------
- Modified train, val, and test corpora as JSON
- Grouped summary log showing replacement counts and all inflected forms
"""

import os
import re
import json
import argparse
import importlib.util
from tqdm import tqdm
from collections import defaultdict


# === Utility loaders ===
def load_config(config_path):
    """Load a Python config file and return its `config` dict."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def load_json(path):
    """Load JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Save object as JSON with pretty formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# === Data preparation ===
def load_replacement_map(replacements_path):
    """Flatten CSV-style hypernym map: 'w1,w2' → 'artificial'  → dict base→artificial."""
    data = load_json(replacements_path)
    mapping = {}
    for pair, artificial in data.items():
        w1, w2 = map(str.strip, pair.split(","))
        mapping[w1.lower()] = artificial
        mapping[w2.lower()] = artificial
    return mapping


def load_master_inflections(master_path, base_to_artificial):
    """
    Build a flattened inflection map (form → (base, artificial)) from master file.
    Logs warnings if artificials are missing or extra.
    """
    master_data = load_json(master_path)
    expected_artificials = set(base_to_artificial.values())
    available_artificials = set(master_data.keys())

    missing = expected_artificials - available_artificials
    extra = available_artificials - expected_artificials

    if missing:
        print(f"⚠️  Warning: {len(missing)} artificials from the replacements map "
              f"are missing in {master_path}: {sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"ℹ️  Note: {len(extra)} artificials exist in {master_path} "
              f"but are not used in this experiment.")

    inflection_map = {}
    for artificial, info in master_data.items():
        if artificial not in expected_artificials:
            continue
        for base, forms in info["forms"].items():
            for form in forms:
                inflection_map[form.lower()] = (base, artificial)
    return inflection_map


# === Replacement helpers ===
def preserve_case(original, replacement):
    """Match the replacement's case style to the original token."""
    if original.isupper():
        return replacement.upper()
    elif original.istitle():
        return replacement.capitalize()
    return replacement


def replace_tokens_half(corpus, inflection_map):
    """
    Replace half (every other) occurrence of each base across the corpus.
    Preserves punctuation and casing.
    """
    modified = []
    replacement_counts = defaultdict(int)
    artificial_counts = defaultdict(int)
    counters = defaultdict(int)

    for doc in tqdm(corpus, desc="Replacing words (50%)"):
        new_doc = []
        for token in doc:
            parts = re.findall(r"\w+|[^\w\s]", token, re.UNICODE)
            new_parts = []
            for part in parts:
                if re.fullmatch(r"\w+", part):
                    key = part.lower()
                    if key in inflection_map:
                        base, artificial = inflection_map[key]
                        counters[base] += 1
                        if counters[base] % 2 == 1:  # replace every 2nd
                            replaced = preserve_case(part, artificial)
                            replacement_counts[base] += 1
                            artificial_counts[artificial] += 1
                            new_parts.append(replaced)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            new_doc.append("".join(new_parts))
        modified.append(new_doc)

    return modified, replacement_counts, artificial_counts


# === Logging ===
def save_replacement_log(train_repl, val_repl, test_repl,
                         train_artif, val_artif, test_artif,
                         log_filename, inflection_map, base_map,
                         train_in, val_in, test_in,
                         train_out, val_out, test_out,
                         replacements_path, master_path):
    """Save grouped summary log with inflected forms and per-split counts."""
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    base_to_forms = defaultdict(set)
    for form, (base, _) in inflection_map.items():
        base_to_forms[base].add(form)

    pair_to_bases = defaultdict(list)
    for base, art in base_map.items():
        pair_to_bases[art].append(base)

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("Partial Hypernym Replacement Summary (50%)\n")
        f.write("=========================================\n")
        f.write(f"Train input:  {train_in}\n")
        f.write(f"Val input:    {val_in}\n")
        f.write(f"Test input:   {test_in}\n")
        f.write(f"Train output: {train_out}\n")
        f.write(f"Val output:   {val_out}\n")
        f.write(f"Test output:  {test_out}\n")
        f.write(f"Replacements: {replacements_path}\n")
        f.write(f"Master inflections: {master_path}\n\n")

        for artificial, bases in sorted(pair_to_bases.items()):
            f.write(f"=== {', '.join(bases)} → {artificial} ===\n")
            f.write("Inflected forms:\n")
            for base in bases:
                for form in sorted(base_to_forms.get(base, [])):
                    f.write(f"  - {form}\n")

            train_count = sum(train_repl.get(b, 0) for b in bases)
            val_count = sum(val_repl.get(b, 0) for b in bases)
            test_count = sum(test_repl.get(b, 0) for b in bases)
            f.write(f"\nTrain replacements: {train_count}\n")
            f.write(f"Val replacements:   {val_count}\n")
            f.write(f"Test replacements:  {test_count}\n\n")

    print(f"\nReplacement log saved to: {log_filename}")


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Insert artificial hypernyms (50%) using a config file.")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loading replacements: {cfg['replacements']}")
    base_map = load_replacement_map(cfg["replacements"])
    inflection_map = load_master_inflections(cfg["master_inflections"], base_map)

    print(f"\nLoading training corpus: {cfg['train_in']}")
    train = load_json(cfg["train_in"])
    train_mod, train_repl, train_artif = replace_tokens_half(train, inflection_map)
    save_json(train_mod, cfg["train_out"])
    print(f"✅ Saved: {cfg['train_out']}")

    print(f"\nLoading validation corpus: {cfg['val_in']}")
    val = load_json(cfg["val_in"])
    val_mod, val_repl, val_artif = replace_tokens_half(val, inflection_map)
    save_json(val_mod, cfg["val_out"])
    print(f"✅ Saved: {cfg['val_out']}")

    print(f"\nLoading test corpus: {cfg['test_in']}")
    test = load_json(cfg["test_in"])
    test_mod, test_repl, test_artif = replace_tokens_half(test, inflection_map)
    save_json(test_mod, cfg["test_out"])
    print(f"✅ Saved: {cfg['test_out']}")

    save_replacement_log(train_repl, val_repl, test_repl,
                         train_artif, val_artif, test_artif,
                         cfg["summary_log"], inflection_map, base_map,
                         cfg["train_in"], cfg["val_in"], cfg["test_in"],
                         cfg["train_out"], cfg["val_out"], cfg["test_out"],
                         cfg["replacements"], cfg["master_inflections"])

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()