#!/usr/bin/env python3
"""
debug_component_triggers_ckpt1.py
================================

Debug helper for evaluate_new_comp_accuracy.py.

Scans test.bin blocks for a given model and prints example blocks whose context X
triggers the "contains_component" category (component word base/inflection forms).

For each matching block, prints:
- which component form(s) triggered (unique per block, left-to-right order)
- the decoded X text for the block (full)
- the full list of split parts seen by the matcher

No wandb logging. No model forward pass.

Usage:
    python debug_component_triggers_ckpt1.py <model_name> <map_json> <master_inflections_json>

Example:
    python debug_component_triggers_ckpt1.py gpt_recipe_1_HOM \
      /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/recipe_1_HOM_map.json \
      /home/nina/ambiguity/experiments/inflection_maps/allkinds/recipe_master_HOM_inflections_edited.json
"""

import os
import sys
import json
import torch
import numpy as np
import tiktoken
import re


# =========================
# CLI
# =========================
if len(sys.argv) < 4:
    print("Usage: python debug_component_triggers_ckpt1.py <model_name> <map_json> <master_inflections_json>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
MAP_JSON = sys.argv[2]
MASTER_INFLECTIONS_JSON = sys.argv[3]


# =========================
# Paths
# =========================
CHECKPOINT_DIR = f"/home/nina/ambiguity/trained_models/allkinds/{MODEL_NAME}"
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "ckpt1.pt")  # informational only (not required)
TEST_DATA_PATH = f"/home/nina/ambiguity/gpt_pipelines/allkinds/data/{MODEL_NAME}/test.bin"

OUT_DIR = f"/home/nina/ambiguity/evaluation/new_and_comp_accuracy/allkinds/{MODEL_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "debug_component_triggers_ckpt1.txt")


# =========================
# Settings
# =========================
BLOCK_SIZE = 256
BATCH_SIZE = 64
MAX_MATCH_BLOCKS = 25  # how many "contains_component" blocks to print


# =========================
# Data loading
# =========================
def assert_exists(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


assert_exists(TEST_DATA_PATH, "test.bin")
assert_exists(MAP_JSON, "map_json")
assert_exists(MASTER_INFLECTIONS_JSON, "master_inflections_json")

test_data = np.memmap(TEST_DATA_PATH, dtype=np.uint16, mode="r")


def get_full_batches():
    """Yield (X, Y) batches covering the full test set (Y is unused; kept for parity)."""
    for i in range(0, len(test_data) - BLOCK_SIZE, BLOCK_SIZE * BATCH_SIZE):
        x_batch = []
        y_batch = []
        for j in range(BATCH_SIZE):
            start = i + j * BLOCK_SIZE
            end = start + BLOCK_SIZE
            if end + 1 > len(test_data):
                break
            x = torch.from_numpy(test_data[start:end].astype(np.int64))
            y = torch.from_numpy(test_data[start + 1:end + 1].astype(np.int64))
            x_batch.append(x)
            y_batch.append(y)
        if x_batch and y_batch:
            yield torch.stack(x_batch), torch.stack(y_batch)


# =========================
# Load new words (original map order)
# =========================
def load_new_words_from_map(map_json_path):
    """
    Load new word surface forms from a replacements map,
    preserving their order of appearance in the JSON file.
    """
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_words = []
    seen = set()

    if isinstance(data, dict):
        for v in data.values():  # dict order preserved in Python 3.7+
            if isinstance(v, str):
                w = v.strip()
                if w and w not in seen:
                    new_words.append(w)
                    seen.add(w)

    return new_words


# =========================
# Load component forms (original master order, filtered)
# =========================
def load_component_forms_from_master(master_inflections_path, allowed_artificials):
    """
    Load component-word inflected forms from a master inflections JSON,
    filtered to only artificials used in this run.

    Preserves the order in which forms appear in the JSON file.
    Returns a list of lowercased forms.
    """
    with open(master_inflections_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    forms_list = []
    seen = set()

    if not isinstance(data, dict):
        return forms_list

    for artificial, entry in data.items():
        if artificial not in allowed_artificials:
            continue

        if not isinstance(entry, dict):
            continue

        forms = entry.get("forms", {})
        if not isinstance(forms, dict):
            continue

        for _base, form_list in forms.items():
            if not isinstance(form_list, list):
                continue
            for form in form_list:
                if isinstance(form, str):
                    form_lc = form.strip().lower()
                    if form_lc and form_lc not in seen:
                        forms_list.append(form_lc)
                        seen.add(form_lc)

    return forms_list


enc = tiktoken.get_encoding("gpt2")

NEW_WORDS = load_new_words_from_map(MAP_JSON)
ALLOWED_ARTIFICIALS = set(NEW_WORDS)  # artificials used in this run

COMPONENT_FORMS = load_component_forms_from_master(MASTER_INFLECTIONS_JSON, ALLOWED_ARTIFICIALS)
COMPONENT_FORMS_SET = set(COMPONENT_FORMS)

_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"^\w+$", re.UNICODE)


def find_component_triggers_in_x(x_row):
    """
    Return (triggers, x_text, parts) for one X row.

    triggers: list of matched component forms in left-to-right order,
              unique within the block
    parts: full list of split parts returned by r'\\w+|[^\\w\\s]'
    """
    text = enc.decode(x_row.tolist())
    parts = _SPLIT_RE.findall(text)

    triggers = []
    seen = set()
    for part in parts:
        if _WORD_RE.fullmatch(part):
            w = part.lower()
            if w in COMPONENT_FORMS_SET and w not in seen:
                triggers.append(w)
                seen.add(w)

    return triggers, text, parts


# =========================
# Main
# =========================
def main():
    header_lines = [
        f"DEBUG component triggers for {MODEL_NAME} (no model forward pass)",
        f"Checkpoint path (informational only): {CKPT_PATH}",
        f"Test data:  {TEST_DATA_PATH}",
        f"Map JSON:   {MAP_JSON}",
        f"Master infl:{MASTER_INFLECTIONS_JSON}",
        f"New words (map order, {len(NEW_WORDS)}): {NEW_WORDS}",
        f"Allowed artificials for component eval ({len(ALLOWED_ARTIFICIALS)}): {sorted(ALLOWED_ARTIFICIALS)}",
        f"Component forms (master order, lowercased, {len(COMPONENT_FORMS)}): {COMPONENT_FORMS}",
        "",
        f"Printing up to {MAX_MATCH_BLOCKS} blocks that trigger contains_component.",
        "",
    ]

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for line in header_lines:
            print(line)
            f.write(line + "\n")

        found = 0
        total_rows_seen = 0

        for X, _Y in get_full_batches():
            for b in range(X.size(0)):
                total_rows_seen += 1

                triggers, x_text, parts = find_component_triggers_in_x(X[b])
                if not triggers:
                    continue

                found += 1

                block_lines = [
                    f"=== MATCH {found} (global_row #{total_rows_seen}, batch_row={b}) ===",
                    f"triggers: {triggers}",
                    f"parts: {parts}",
                    f"X_text: {repr(x_text)}",
                    "",
                ]

                for line in block_lines:
                    print(line)
                    f.write(line + "\n")

                if found >= MAX_MATCH_BLOCKS:
                    break

            if found >= MAX_MATCH_BLOCKS:
                break

        tail = [
            "",
            f"Done. Found {found} matching blocks (scanned {total_rows_seen} rows).",
            f"Output written to: {OUT_FILE}",
        ]
        for line in tail:
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()