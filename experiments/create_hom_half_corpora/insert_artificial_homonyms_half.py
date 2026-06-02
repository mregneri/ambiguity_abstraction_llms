"""
insert_artificial_homonyms_half.py
==================================

This script inserts artificial homonyms into tokenized training, validation,
and test corpora — but replaces **only every other occurrence** of each word
(≈50% of all possible replacements).

It supports replacing multiple word pairs with new artificial words in all three
corpora and uses pre-generated, manually verified master inflection maps instead
of generating forms at runtime. Each comma-separated word pair will be expanded
to include all inflected forms (from the master maps) and replaced only when a
full word match occurs after removing punctuation separators. Casing is preserved,
and original punctuation is restored.

Expected Config Format (Python):
--------------------------------
config = {
    'train_in': 'path/to/train.json',
    'val_in': 'path/to/val.json',
    'test_in': 'path/to/test.json',
    'train_out': 'path/to/output_train.json',
    'val_out': 'path/to/output_val.json',
    'test_out': 'path/to/output_test.json',
    'replacements': 'path/to/homonym_map.json',
    'master_inflections': 'path/to/master_inflections.json',
    'summary_log': 'path/to/summary_log.txt'
}

Replacement File Format (JSON):
-------------------------------
{
    "bathrobe,teamwork": "bathwork",
    "moisture,sundae": "moistdae"
}

Master Inflection Map Format (JSON):
------------------------------------
{
  "bathwork": {
    "bases": ["bathrobe", "teamwork"],
    "forms": {
      "bathrobe": ["bathrobe", "bathrobes"],
      "teamwork": ["teamwork"]
    }
  },
  ...
}

Usage:
------
python insert_artificial_homonyms_half.py --config config/insert_recipe_1_HOM_HALF.py

Output:
-------
- Modified train, val, and test corpora as JSON
- A grouped summary log with replacement statistics and all inflected forms
"""

import os
import json
import argparse
import importlib.util
import re
from tqdm import tqdm
from collections import defaultdict


# === CONFIG LOADING ===
def load_config(config_path):
    """Load a Python config file and return its `config` dictionary."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def load_corpus(file_path):
    """Load a JSON corpus file (list of token lists)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_corpus(corpus, output_path):
    """Save a corpus to a JSON file with pretty formatting."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


# === MAP LOADING ===
def load_replacement_map(replacements_path):
    """Load the replacement map (homonym pairs → artificial word)."""
    with open(replacements_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    base_to_artificial = {}
    for pair, artificial in data.items():
        w1, w2 = map(str.strip, pair.split(','))
        base_to_artificial[w1.lower()] = artificial
        base_to_artificial[w2.lower()] = artificial
    return base_to_artificial


def load_master_inflections(master_path, base_to_artificial):
    """
    Build a flattened inflection map (form → (base, artificial)) from the master inflection file.
    Logs warnings if artificials are missing or extra.
    """
    with open(master_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    expected_artificials = set(base_to_artificial.values())
    available_artificials = set(master_data.keys())

    missing_in_master = expected_artificials - available_artificials
    extra_in_master = available_artificials - expected_artificials

    if missing_in_master:
        print(f"⚠️  Warning: {len(missing_in_master)} artificials from the replacements map "
              f"are missing in {master_path}: {sorted(list(missing_in_master))[:5]}{'...' if len(missing_in_master) > 5 else ''}")

    if extra_in_master:
        print(f"ℹ️  Note: {len(extra_in_master)} artificials exist in {master_path} "
              f"but are not used in this experiment.")

    inflection_map = {}
    for artificial, info in master_data.items():
        if artificial not in expected_artificials:
            continue  # skip unused artificials
        for base, forms in info["forms"].items():
            for form in forms:
                inflection_map[form.lower()] = (base, artificial)
    return inflection_map


# === TOKEN REPLACEMENT ===
def preserve_case(original, replacement):
    """Adjust the casing of the replacement to match the original token."""
    if original.isupper():
        return replacement.upper()
    elif original.istitle():
        return replacement.capitalize()
    else:
        return replacement


def replace_tokens_half(corpus, inflection_map):
    """
    Replace inflected word forms in tokens, preserving punctuation and casing.
    Only every second occurrence of each base word is replaced (≈50% deterministic).

    Returns:
    - modified corpus
    - replacement_counts (base → count)
    - artificial_counts (artificial → count)
    """
    modified = []
    replacement_counts = defaultdict(int)
    artificial_counts = defaultdict(int)
    occurrence_counters = defaultdict(int)

    for doc in tqdm(corpus, desc="Replacing words (50%)"):
        new_doc = []
        for token in doc:
            parts = re.findall(r'\w+|[^\w\s]', token, re.UNICODE)
            new_parts = []
            for part in parts:
                if re.fullmatch(r'\w+', part):  # word
                    key = part.lower()
                    if key in inflection_map:
                        base, artificial = inflection_map[key]
                        occurrence_counters[base] += 1
                        if occurrence_counters[base] % 2 == 1:  # replace every other
                            replaced = preserve_case(part, artificial)
                            replacement_counts[base] += 1
                            artificial_counts[artificial] += 1
                            new_parts.append(replaced)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)  # punctuation
            new_doc.append(''.join(new_parts))
        modified.append(new_doc)
    return modified, replacement_counts, artificial_counts


# === LOGGING ===
def save_replacement_log(train_repl, val_repl, test_repl, base_map,
                         train_artif, val_artif, test_artif, log_filename,
                         train_in, val_in, test_in,
                         train_out, val_out, test_out,
                         replacements_path, inflection_map):
    """
    Save a detailed grouped replacement summary log.
    """
    base_to_forms = defaultdict(set)
    for form, (base, _) in inflection_map.items():
        base_to_forms[base].add(form)

    pair_to_bases = defaultdict(list)
    for base, art in base_map.items():
        pair_to_bases[art].append(base)

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("Homonym Replacement Summary Log (50%)\n")
        f.write("=====================================\n")
        f.write(f"Train input:  {train_in}\n")
        f.write(f"Val input:    {val_in}\n")
        f.write(f"Test input:   {test_in}\n")
        f.write(f"Train output: {train_out}\n")
        f.write(f"Val output:   {val_out}\n")
        f.write(f"Test output:  {test_out}\n")
        f.write(f"Replacements: {replacements_path}\n\n")

        for artificial, bases in sorted(pair_to_bases.items()):
            f.write(f"=== {', '.join(bases)} → {artificial} ===\n")
            f.write("Inflected forms:\n")
            for base in bases:
                for form in sorted(base_to_forms.get(base, [])):
                    f.write(f"  - {form}\n")
            train_count = sum(train_repl.get(base, 0) for base in bases)
            val_count = sum(val_repl.get(base, 0) for base in bases)
            test_count = sum(test_repl.get(base, 0) for base in bases)
            f.write(f"\nTrain replacements: {train_count}\n")
            f.write(f"Val replacements:   {val_count}\n")
            f.write(f"Test replacements:  {test_count}\n\n\n")

    print(f"\nReplacement log saved to: {log_filename}")


# === MAIN ===
def main():
    parser = argparse.ArgumentParser(
        description="Insert artificial homonyms (50%) into corpora using a config file."
    )
    parser.add_argument('--config', required=True, help="Path to Python config file defining corpus paths")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading replacement map from: {config['replacements']}")
    base_map = load_replacement_map(config['replacements'])

    print(f"Loading master inflections from: {config['master_inflections']}")
    inflection_map = load_master_inflections(config['master_inflections'], base_map)

    print(f"\nLoading training corpus from: {config['train_in']}")
    train = load_corpus(config['train_in'])
    train_mod, train_repl, train_artif = replace_tokens_half(train, inflection_map)
    save_corpus(train_mod, config['train_out'])
    print(f"Modified training set saved to: {config['train_out']}")

    print(f"\nLoading validation corpus from: {config['val_in']}")
    val = load_corpus(config['val_in'])
    val_mod, val_repl, val_artif = replace_tokens_half(val, inflection_map)
    save_corpus(val_mod, config['val_out'])
    print(f"Modified validation set saved to: {config['val_out']}")

    print(f"\nLoading test corpus from: {config['test_in']}")
    test = load_corpus(config['test_in'])
    test_mod, test_repl, test_artif = replace_tokens_half(test, inflection_map)
    save_corpus(test_mod, config['test_out'])
    print(f"Modified test set saved to: {config['test_out']}")

    save_replacement_log(
        train_repl, val_repl, test_repl, base_map,
        train_artif, val_artif, test_artif, config['summary_log'],
        config['train_in'], config['val_in'], config['test_in'],
        config['train_out'], config['val_out'], config['test_out'],
        config['replacements'], inflection_map
    )

    print("Done!")


if __name__ == "__main__":
    main()