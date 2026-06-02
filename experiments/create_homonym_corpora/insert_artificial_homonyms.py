"""
insert_artificial_homonyms.py
=============================

This script inserts artificial homonyms into tokenized training, validation, and test corpora.
It supports replacing multiple word pairs with new artificial words across all splits,
saving a detailed log of replacement counts and inflected forms.

All input/output paths and parameters are specified in a Python config file
that is passed using the --config argument.

Instead of calling `lemminflect` at runtime, the script uses pre-generated, manually verified
master inflection maps. Each comma-separated word pair is expanded to include all its inflected
forms (from the master map) and replaced only when a full word match occurs after separating
punctuation. Casing is preserved, and original punctuation is restored.

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
python insert_artificial_homonyms.py --config config/insert_recipe_10_HOM.py

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


# === Helper functions ===

def load_config(config_path):
    """Load a Python config file and return its `config` dictionary."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def load_json(path):
    """Load a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    """Save a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preserve_case(original, replacement):
    """Match the casing style of the original token."""
    if original.isupper():
        return replacement.upper()
    elif original.istitle():
        return replacement.capitalize()
    else:
        return replacement


def build_inflection_map_from_master(master_map, replacements_map):
    """
    Build a lookup from inflected forms to (base, artificial).

    Uses the manually edited master inflection maps to include all variants.
    Returns dict: {form.lower(): (base, artificial)}
    """
    inflection_map = {}
    for pair, artificial in replacements_map.items():
        w1, w2 = map(str.strip, pair.split(','))
        entry = master_map.get(artificial)
        if not entry:
            print(f"⚠️  Warning: '{artificial}' not found in master map. Skipping inflections.")
            continue

        for base in entry["bases"]:
            forms = entry["forms"].get(base, [])
            for form in forms:
                inflection_map[form.lower()] = (base, artificial)
    return inflection_map


def replace_tokens_exact(corpus, inflection_map):
    """
    Replace inflected word forms in tokens, preserving punctuation and casing.

    Uses regex to separate punctuation from words. Only replaces tokens that match
    full word forms from the inflection map. Punctuation is restored after replacement.
    """
    modified = []
    replacement_counts = defaultdict(int)
    artificial_counts = defaultdict(int)

    for doc in tqdm(corpus, desc="Replacing words"):
        new_doc = []
        for token in doc:
            parts = re.findall(r'\w+|[^\w\s]', token, re.UNICODE)
            new_parts = []
            for part in parts:
                if re.fullmatch(r'\w+', part):
                    key = part.lower()
                    if key in inflection_map:
                        base, artificial = inflection_map[key]
                        replaced = preserve_case(part, artificial)
                        replacement_counts[base] += 1
                        artificial_counts[artificial] += 1
                        new_parts.append(replaced)
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            new_doc.append(''.join(new_parts))
        modified.append(new_doc)

    return modified, replacement_counts, artificial_counts


def save_replacement_log(train_repl, val_repl, test_repl, replacements_map,
                         train_artif, val_artif, test_artif, log_filename,
                         config, inflection_map, master_map):
    """
    Save a grouped replacement summary log, including all inflected forms.
    """
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write("Homonym Replacement Summary Log\n")
        f.write("===============================\n")
        for key in ["train_in", "val_in", "test_in",
                    "train_out", "val_out", "test_out",
                    "replacements", "master_inflections"]:
            f.write(f"{key}: {config[key]}\n")
        f.write("\n")

        for pair, artificial in replacements_map.items():
            w1, w2 = map(str.strip, pair.split(','))
            f.write(f"=== {w1}, {w2} → {artificial} ===\n")

            entry = master_map.get(artificial, {})
            base_forms = entry.get("forms", {}) if entry else {}

            f.write("Inflected forms:\n")
            for base, forms in base_forms.items():
                f.write(f"  {base}: {forms}\n")

            train_count = sum(train_repl.get(b, 0) for b in (w1, w2))
            val_count = sum(val_repl.get(b, 0) for b in (w1, w2))
            test_count = sum(test_repl.get(b, 0) for b in (w1, w2))
            f.write(f"\nTrain replacements: {train_count}\n")
            f.write(f"Val replacements:   {val_count}\n")
            f.write(f"Test replacements:  {test_count}\n\n")

    print(f"\n📝 Replacement log saved to: {log_filename}")


# === Main ===
def main():
    parser = argparse.ArgumentParser(
        description="Insert artificial homonyms into corpora using master inflection maps."
    )
    parser.add_argument('--config', required=True,
                        help="Path to Python config file defining corpus paths.")
    args = parser.parse_args()

    # Load config and data
    config = load_config(args.config)
    replacements_map = load_json(config["replacements"])
    master_map = load_json(config["master_inflections"])

    print(f"Loaded master inflection map with {len(master_map)} entries.")
    print(f"Loaded replacements map with {len(replacements_map)} pairs.")

    # Build inflection lookup
    inflection_map = build_inflection_map_from_master(master_map, replacements_map)
    print(f"Constructed inflection lookup with {len(inflection_map)} word forms.\n")

    # Process corpora
    for split in ["train", "val", "test"]:
        print(f"Processing {split} corpus...")
        corpus = load_json(config[f"{split}_in"])
        modified, repl_counts, artif_counts = replace_tokens_exact(corpus, inflection_map)
        save_json(modified, config[f"{split}_out"])
        print(f"✅ Saved modified {split} corpus → {config[f'{split}_out']}")

        if split == "train":
            train_repl, train_artif = repl_counts, artif_counts
        elif split == "val":
            val_repl, val_artif = repl_counts, artif_counts
        else:
            test_repl, test_artif = repl_counts, artif_counts

    # Log summary
    save_replacement_log(
        train_repl, val_repl, test_repl, replacements_map,
        train_artif, val_artif, test_artif,
        config["summary_log"], config, inflection_map, master_map
    )

    print("\n🎉 All splits processed successfully!")


if __name__ == "__main__":
    main()