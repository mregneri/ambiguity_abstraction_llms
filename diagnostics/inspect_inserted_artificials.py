#!/usr/bin/env python3
"""
inspect_inserted_artificials.py
===============================

Inspects corpora that contain artificial homonyms or hypernyms.

This version mirrors insert_artificial_homonyms.py matching logic:
- splits tokens into word/punctuation parts via: re.findall(r'\\w+|[^\\w\\s]')
- replaces/matches ONLY full word parts (re.fullmatch(r'\\w+'))
- uses the master inflections JSON (manually edited) to define which forms belong
  to each base/component word for a given artificial.

It verifies that inserted artificial words appear as expected, collects sample matches,
and counts frequencies of:
- original base/component words (counting any of their inflected forms)
- artificial words (exact word match)

Usage:
------
python inspect_inserted_artificials.py --config path/to/your/config.py
"""

import json
import argparse
import importlib.util
import os
import re
from tqdm import tqdm
from collections import defaultdict


# -------------------------
# IO / config
# -------------------------

def load_config(config_path):
    """Load a Python config file and return its `config` dictionary."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus(file_path):
    """Load a JSON corpus file (list of token lists)."""
    return load_json(file_path)


def load_replacement_pairs(replacements_path):
    """
    Load a replacement map from JSON.

    Expected format:
      {
        "w1,w2": "artificial",
        ...
      }

    Returns:
      pairs: list of tuples ([w1, w2], artificial)
      replacements_map: dict raw (pair_str -> artificial)
    """
    replacements_map = load_json(replacements_path)
    pairs = []
    for pair, artificial in replacements_map.items():
        w1, w2 = map(str.strip, pair.split(","))
        pairs.append(([w1, w2], artificial))
    return pairs, replacements_map


# -------------------------
# Matching logic (same as insertion script)
# -------------------------

WORD_OR_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def iter_word_parts(token):
    """
    Yield (part, is_word) for each part produced by the insertion-style split.
    """
    parts = WORD_OR_PUNCT_RE.findall(token)
    for part in parts:
        is_word = bool(re.fullmatch(r"\w+", part))
        yield part, is_word


# -------------------------
# Build lookups from master inflections
# -------------------------

def build_inspection_lookups(master_map, replacements_map):
    """
    Build:
      - form_to_base: dict form_lower -> base (string)
        where form comes from master_map[artificial]["forms"][base] for bases in that entry
      - artificial_set: set of artificial_lower
    Only includes artificials that are present in replacements_map.
    """
    form_to_base = {}
    artificial_set = set()

    for pair, artificial in replacements_map.items():
        artificial_lower = artificial.lower()
        artificial_set.add(artificial_lower)

        entry = master_map.get(artificial)
        if not entry:
            print(f"⚠️  Warning: artificial '{artificial}' not found in master map. Skipping its base-form lookup.")
            continue

        forms_by_base = entry.get("forms", {})
        # entry["bases"] is available too, but "forms" is the authoritative list of what to match
        for base, forms in forms_by_base.items():
            for form in forms:
                form_to_base[str(form).lower()] = base  # base key exactly as in master map

    return form_to_base, artificial_set


# -------------------------
# Scanning
# -------------------------

def scan_split(corpus, form_to_base, artificial_set, keys_for_samples, max_matches=100):
    """
    Scan corpus and return:
      matches: dict key -> list(samples)
      counts:  dict key -> int

    keys:
      - base keys are exactly the base strings from master_map entries (e.g. "earthquake")
        counts accumulate across all of their forms.
      - artificial keys are the artificial string as it appears in the map (we store counts under
        that spelling, but matching uses lowercased membership in artificial_set).

    keys_for_samples: iterable of keys we want to store samples for (bases + artificials)
    """
    matches = {k: set() for k in keys_for_samples}
    counts = defaultdict(int)

    # To map artificial_lower back to the canonical key spelling we want in output,
    # we’ll accept that keys_for_samples contains the canonical artificials (case from map).
    # Build a reverse for artificials:
    artificial_lower_to_key = {k.lower(): k for k in keys_for_samples if k.lower() in artificial_set}

    for doc in tqdm(corpus, desc="Scanning corpus"):
        for token in doc:
            for part, is_word in iter_word_parts(token):
                if not is_word:
                    continue

                p_l = part.lower()

                # base form match
                if p_l in form_to_base:
                    base = form_to_base[p_l]
                    counts[base] += 1
                    if base in matches and len(matches[base]) < max_matches:
                        matches[base].add(token)

                # artificial match
                if p_l in artificial_set:
                    art_key = artificial_lower_to_key.get(p_l, p_l)
                    counts[art_key] += 1
                    if art_key in matches and len(matches[art_key]) < max_matches:
                        matches[art_key].add(token)

    matches = {k: list(v) for k, v in matches.items()}
    return matches, counts


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect inserted artificial words in corpora using insertion-style matching + master inflections."
    )
    parser.add_argument("--config", required=True, help="Path to Python config file defining corpus paths")
    parser.add_argument("--max_matches", type=int, default=100, help="Max unique sample tokens stored per key")
    args = parser.parse_args()

    config = load_config(args.config)

    replacements_path = config["replacements"]
    master_inflections_path = config["master_inflections"]

    train_path = config["train_out"]
    val_path = config["val_out"]
    test_path = config["test_out"]

    out_dir = "/home/nina/ambiguity/diagnostics/allkinds/inspection_logs"
    os.makedirs(out_dir, exist_ok=True)

    # Derive output name from the actual inspected corpus (distinguishes HOM vs HOM_HALF, etc.)
    train_out_basename = os.path.basename(train_path)  # e.g. recipe_30_HOM_HALF_train.json
    model_name = train_out_basename.replace("_train.json", "")
    out_path = os.path.join(out_dir, f"{model_name}_inspection.txt")

    print(f"Replacement map:      {replacements_path}")
    print(f"Master inflections:   {master_inflections_path}")
    print(f"Inspecting train set: {train_path}")
    print(f"Inspecting val set:   {val_path}")
    print(f"Inspecting test set:  {test_path}")
    print(f"Results will be saved to: {out_path}")

    pairs, replacements_map = load_replacement_pairs(replacements_path)
    master_map = load_json(master_inflections_path)

    form_to_base, artificial_set = build_inspection_lookups(master_map, replacements_map)

    print(f"Loaded master map with {len(master_map)} artificials.")
    print(f"Loaded replacements map with {len(replacements_map)} pairs.")
    print(f"Constructed base-form lookup with {len(form_to_base)} word forms.")
    print(f"Tracking {len(artificial_set)} artificial words.\n")

    # Keys we want counts/samples for:
    # - bases: from master_map entries for included artificials (not just w1/w2 strings)
    bases_in_scope = set(form_to_base.values())
    artificials_in_scope = [art for _, art in pairs]  # canonical spelling from map
    keys_for_samples = sorted(bases_in_scope) + artificials_in_scope

    # Load corpora
    train_corpus = load_corpus(train_path)
    val_corpus = load_corpus(val_path)
    test_corpus = load_corpus(test_path)

    # Scan splits
    print("Processing training set...")
    train_matches, train_counts = scan_split(
        train_corpus, form_to_base, artificial_set, keys_for_samples, max_matches=args.max_matches
    )

    print("\nProcessing validation set...")
    val_matches, val_counts = scan_split(
        val_corpus, form_to_base, artificial_set, keys_for_samples, max_matches=args.max_matches
    )

    print("\nProcessing test set...")
    test_matches, test_counts = scan_split(
        test_corpus, form_to_base, artificial_set, keys_for_samples, max_matches=args.max_matches
    )

    # Write report grouped by replacement pair
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Inspection Results (Grouped by Replacement Pair)\n")
        f.write("================================================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Config: {args.config}\n")

        for key in ["train_out", "val_out", "test_out", "replacements", "master_inflections"]:
            f.write(f"{key}: {config[key]}\n")
        f.write("\n")

        for originals, artificial in pairs:
            # originals are w1/w2 from the map, but master_map has authoritative bases/forms
            entry = master_map.get(artificial, {})
            bases = entry.get("bases", originals)
            forms_by_base = entry.get("forms", {})

            f.write("------------------------------------------------\n")
            f.write(f"Pair: {', '.join(originals)} → {artificial}\n")
            f.write("------------------------------------------------\n")

            # Forms shown exactly as in master map
            f.write("\nInflected forms (from master map):\n")
            if forms_by_base:
                for base, forms in forms_by_base.items():
                    f.write(f"  {base}: {forms}\n")
            else:
                f.write("  (No master-map entry found; inflected forms unavailable.)\n")

            # Samples
            def write_samples(label, match_dict):
                f.write(f"\n{label} samples:\n")
                for base in bases:
                    samples = match_dict.get(base, [])
                    if samples:
                        f.write(f"  {base}:\n")
                        for s in samples:
                            f.write(f"    {s}\n")
                art_samples = match_dict.get(artificial, [])
                if art_samples:
                    f.write(f"  {artificial}:\n")
                    for s in art_samples:
                        f.write(f"    {s}\n")

            write_samples("Train", train_matches)
            write_samples("Validation", val_matches)
            write_samples("Test", test_matches)

            # Counts
            f.write("\nCounts:\n")
            for base in bases:
                t = train_counts.get(base, 0)
                v = val_counts.get(base, 0)
                s = test_counts.get(base, 0)
                f.write(f"  {base}: train={t}, val={v}, test={s}, total={t+v+s}\n")

            t = train_counts.get(artificial, 0)
            v = val_counts.get(artificial, 0)
            s = test_counts.get(artificial, 0)
            f.write(f"  {artificial}: train={t}, val={v}, test={s}, total={t+v+s}\n")

            f.write("\n\n")

    print(f"\n✅ Inspection results written to: {out_path}")


if __name__ == "__main__":
    main()