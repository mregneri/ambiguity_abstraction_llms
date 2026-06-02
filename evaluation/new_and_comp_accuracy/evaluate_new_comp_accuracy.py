#!/usr/bin/env python3
"""
evaluate_new_comp_accuracy.py
====================================

Minimal-modification variant of evaluate_accuracy.py that additionally reports
next-token accuracy separately for:

- blocks whose context X contains any inserted new word (decode + word/punctuation split match)
- blocks whose context X contains no inserted new word (same detection rule)
- blocks whose context X contains any component word (base or inflected form; same decode + split match)
- blocks whose context X contains no component word (same detection rule)
- blocks whose context X contains either a new word OR a component word (same detection rule)
- blocks whose context X contains neither a new word NOR a component word (same detection rule)

Usage:
    python evaluate_new_comp_accuracy.py <model_name> <map_json> <master_inflections_json>

Example:
    python evaluate_new_comp_accuracy.py gpt_recipe_1_HOM \
      /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/recipe_1_HOM_map.json \
      /home/nina/ambiguity/experiments/inflection_maps/allkinds/recipe_master_HOM_inflections_edited.json

Expected folder structure:
    /home/nina/ambiguity/trained_models/allkinds/<model_name>/ckpt*.pt
    /home/nina/ambiguity/gpt_pipelines/allkinds/data/<model_name>/test.bin

Output:
    /home/nina/ambiguity/evaluation/new_and_comp_accuracy/allkinds/<model_name>/new_comp_accuracy_results.txt
"""

import os
import sys
import json
import torch
import numpy as np
import wandb
import tiktoken
import re

# Add path to access the GPT model code
sys.path.append("/home/nina/ambiguity/gpt_pipelines")
from model import GPTConfig, GPT

# ========== PARSE ARGUMENT ==========
if len(sys.argv) < 4:
    print("Usage: python evaluate_new_comp_accuracy.py <model_name> <map_json> <master_inflections_json>")
    sys.exit(1)

# Model identifier (e.g. gpt_recipe_1_HOM or gpt_tiny_50_HYP_HALF)
MODEL_NAME = sys.argv[1]
MAP_JSON = sys.argv[2]
MASTER_INFLECTIONS_JSON = sys.argv[3]

# Define paths for checkpoints, test data, and result output
CHECKPOINT_DIR = f"/home/nina/ambiguity/trained_models/allkinds/{MODEL_NAME}"
CHECKPOINT_FILES = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
TEST_DATA_PATH = f"/home/nina/ambiguity/gpt_pipelines/allkinds/data/{MODEL_NAME}/test.bin"

# Create model-specific results directory
RESULTS_DIR = os.path.join(
    "/home/nina/ambiguity/evaluation/new_and_comp_accuracy/allkinds", MODEL_NAME
)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "new_comp_accuracy_results.txt")

# Evaluation settings
BLOCK_SIZE = 256
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== INIT WANDB ==========
# Derive clean tags from model name
model_lower = MODEL_NAME.lower()
tags = []

if "recipe" in model_lower:
    tags.append("recipe")
elif "tiny" in model_lower:
    tags.append("tiny")

if "half" in model_lower:
    tags.append("half")

tags.append("eval")       # always include eval tag
tags.append("new_comp")   # distinguish from standard + newword eval

wandb.init(
    project="ambiguity_allkinds",
    name=f"{MODEL_NAME}_new_comp_eval",
    tags=tags,
    notes="Next-token accuracy on test.bin for ckpt1–3, with new-word + component-word splits (incl. either/neither)"
)

# ========== LOAD TEST DATA ==========
test_data = np.memmap(TEST_DATA_PATH, dtype=np.uint16, mode="r")


def get_full_batches():
    """Yield batches covering the full test set."""
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
            yield torch.stack(x_batch).to(DEVICE), torch.stack(y_batch).to(DEVICE)


# ========== LOAD NEW WORDS (FROM MAP) ==========
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
        for v in data.values():  # JSON dict order is preserved in Python 3.7+
            if isinstance(v, str):
                w = v.strip()
                if w and w not in seen:
                    new_words.append(w)
                    seen.add(w)

    return new_words


def load_component_forms_from_master(master_inflections_path, allowed_artificials):
    """
    Load component-word inflected forms from a master inflections JSON,
    filtered to only artificials used in this run.

    Preserves the order in which forms appear in the JSON file.
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

        forms = entry.get("forms", {})
        if not isinstance(forms, dict):
            continue

        for base, form_list in forms.items():
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

ALLOWED_ARTIFICIALS = set(NEW_WORDS)
COMPONENT_FORMS = load_component_forms_from_master(MASTER_INFLECTIONS_JSON, ALLOWED_ARTIFICIALS)
COMPONENT_FORMS_SET = set(COMPONENT_FORMS)

# --- Use text-level detection mirroring insertion rules ---
# In the pseudoword insertion scripts, replacements happen only when a token is split into
# parts via r'\w+|[^\w\s]' and a \w+ part matches a form (case-insensitive).
# We mirror that here: decode X -> split -> if any \w+ part equals a new word (lowercased).

NEW_WORD_SET = {w.lower() for w in NEW_WORDS if w.strip()}

_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"^\w+$", re.UNICODE)


def x_contains_any_newword(x_row):
    """
    Check whether a single 1D tensor row (context X) contains any new word,
    using the same punctuation-splitting and full-word matching rule as the
    corpus insertion script.
    """
    text = enc.decode(x_row.tolist())
    parts = _SPLIT_RE.findall(text)

    for part in parts:
        if _WORD_RE.fullmatch(part) and part.lower() in NEW_WORD_SET:
            return True
    return False


def x_contains_any_component_word(x_row):
    """
    Check whether a single 1D tensor row (context X) contains any component word form
    (base words + inflections), using the same punctuation-splitting and full-word
    matching rule as the corpus insertion script.
    """
    text = enc.decode(x_row.tolist())
    parts = _SPLIT_RE.findall(text)

    for part in parts:
        if _WORD_RE.fullmatch(part) and part.lower() in COMPONENT_FORMS_SET:
            return True
    return False


# ========== EVALUATE A SINGLE CHECKPOINT ==========
def evaluate_checkpoint(path):
    """Load a model from checkpoint and compute next-token accuracy with new-word and component-word splits."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    # Overall
    total_tokens = 0
    correct_tokens = 0

    # Split: X contains new word vs not
    total_tokens_with = 0
    correct_tokens_with = 0
    total_tokens_without = 0
    correct_tokens_without = 0

    # Diagnostics
    num_blocks_with = 0
    num_blocks_without = 0

    # Component-word split: X contains any component word (base/inflected) vs not
    total_tokens_component = 0
    correct_tokens_component = 0
    num_blocks_component = 0
    total_tokens_no_component = 0
    correct_tokens_no_component = 0
    num_blocks_no_component = 0

    # Joint categories: either / neither
    total_tokens_either = 0
    correct_tokens_either = 0
    num_blocks_either = 0
    total_tokens_neither = 0
    correct_tokens_neither = 0
    num_blocks_neither = 0

    with torch.no_grad():
        for X, Y in get_full_batches():
            logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)

            # Per-row correct counts
            correct_per_row = (preds == Y).sum(dim=1)  # [B]
            total_per_row = torch.full(
                (Y.size(0),), Y.size(1), device=Y.device, dtype=torch.long
            )  # [B]

            # Determine which rows contain a new word in X
            mask_with = []
            for b in range(X.size(0)):
                mask_with.append(x_contains_any_newword(X[b]))
            mask_with = torch.tensor(mask_with, device=Y.device, dtype=torch.bool)
            mask_without = ~mask_with

            # Determine which rows contain a component word (base/inflected) in X
            mask_component = []
            for b in range(X.size(0)):
                mask_component.append(x_contains_any_component_word(X[b]))
            mask_component = torch.tensor(mask_component, device=Y.device, dtype=torch.bool)
            mask_no_component = ~mask_component

            mask_either = mask_with | mask_component
            mask_neither = ~mask_either  # exact complement

            # Sanity: either vs neither is an exact partition
            assert (mask_either | mask_neither).all()
            assert (mask_either.sum() + mask_neither.sum()) == X.size(0)

            # Update diagnostics
            num_blocks_with += mask_with.sum().item()
            num_blocks_without += mask_without.sum().item()
            num_blocks_component += mask_component.sum().item()
            num_blocks_no_component += mask_no_component.sum().item()
            num_blocks_either += mask_either.sum().item()
            num_blocks_neither += mask_neither.sum().item()

            # Aggregate totals (overall)
            correct = correct_per_row.sum().item()
            total = total_per_row.sum().item()
            correct_tokens += correct
            total_tokens += total

            # Aggregate totals (split)
            if mask_with.any():
                correct_tokens_with += correct_per_row[mask_with].sum().item()
                total_tokens_with += total_per_row[mask_with].sum().item()
            if mask_without.any():
                correct_tokens_without += correct_per_row[mask_without].sum().item()
                total_tokens_without += total_per_row[mask_without].sum().item()
            if mask_component.any():
                correct_tokens_component += correct_per_row[mask_component].sum().item()
                total_tokens_component += total_per_row[mask_component].sum().item()
            if mask_no_component.any():
                correct_tokens_no_component += correct_per_row[mask_no_component].sum().item()
                total_tokens_no_component += total_per_row[mask_no_component].sum().item()
            if mask_either.any():
                correct_tokens_either += correct_per_row[mask_either].sum().item()
                total_tokens_either += total_per_row[mask_either].sum().item()
            if mask_neither.any():
                correct_tokens_neither += correct_per_row[mask_neither].sum().item()
                total_tokens_neither += total_per_row[mask_neither].sum().item()

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else float("nan")
    accuracy_with = correct_tokens_with / total_tokens_with if total_tokens_with > 0 else float("nan")
    accuracy_without = correct_tokens_without / total_tokens_without if total_tokens_without > 0 else float("nan")
    accuracy_component = (
        correct_tokens_component / total_tokens_component if total_tokens_component > 0 else float("nan")
    )
    accuracy_no_component = (
        correct_tokens_no_component / total_tokens_no_component if total_tokens_no_component > 0 else float("nan")
    )
    accuracy_either = (
        correct_tokens_either / total_tokens_either if total_tokens_either > 0 else float("nan")
    )
    accuracy_neither = (
        correct_tokens_neither / total_tokens_neither if total_tokens_neither > 0 else float("nan")
    )

    return {
        "accuracy_overall": accuracy,
        "accuracy_contains_new": accuracy_with,
        "accuracy_no_new": accuracy_without,
        "accuracy_contains_component": accuracy_component,
        "accuracy_no_component": accuracy_no_component,
        "accuracy_contains_either": accuracy_either,
        "accuracy_contains_neither": accuracy_neither,
        "n_tokens_overall": total_tokens,
        "n_tokens_contains_new": total_tokens_with,
        "n_tokens_no_new": total_tokens_without,
        "n_tokens_contains_component": total_tokens_component,
        "n_tokens_no_component": total_tokens_no_component,
        "n_tokens_contains_either": total_tokens_either,
        "n_tokens_contains_neither": total_tokens_neither,
        "n_blocks_contains_new": num_blocks_with,
        "n_blocks_no_new": num_blocks_without,
        "n_blocks_contains_component": num_blocks_component,
        "n_blocks_no_component": num_blocks_no_component,
        "n_blocks_contains_either": num_blocks_either,
        "n_blocks_contains_neither": num_blocks_neither,
    }


# ========== MAIN EVALUATION LOOP ==========
results = {}
with open(RESULTS_FILE, "w") as f:
    f.write(f"New-word and component-word split next-token accuracy evaluation for {MODEL_NAME}:\n")
    f.write(f"Test data: {TEST_DATA_PATH}\n")
    f.write(f"Map JSON:  {MAP_JSON}\n")
    f.write(f"Master inflections JSON: {MASTER_INFLECTIONS_JSON}\n")
    f.write(f"New words (original map order, {len(NEW_WORDS)}): {NEW_WORDS}\n")
    f.write(f"Number of tracked artificials for component eval: {len(ALLOWED_ARTIFICIALS)}\n")
    f.write(f"Tracked component forms (original master order, {len(COMPONENT_FORMS)}): {COMPONENT_FORMS}\n")
    f.write("Detection: decode X -> split with r'\\w+|[^\\w\\s]' -> match \\w+ parts (case-insensitive) for new words and component forms\n\n")

    for ckpt_file in CHECKPOINT_FILES:
        path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        out = evaluate_checkpoint(path)

        # Store for wandb
        results[f"{ckpt_file}_accuracy_overall"] = out["accuracy_overall"]

        results[f"{ckpt_file}_accuracy_contains_new"] = out["accuracy_contains_new"]
        results[f"{ckpt_file}_accuracy_no_new"] = out["accuracy_no_new"]
        results[f"{ckpt_file}_n_tokens_contains_new"] = out["n_tokens_contains_new"]
        results[f"{ckpt_file}_n_tokens_no_new"] = out["n_tokens_no_new"]
        results[f"{ckpt_file}_n_blocks_contains_new"] = out["n_blocks_contains_new"]
        results[f"{ckpt_file}_n_blocks_no_new"] = out["n_blocks_no_new"]

        results[f"{ckpt_file}_accuracy_contains_component"] = out["accuracy_contains_component"]
        results[f"{ckpt_file}_n_tokens_contains_component"] = out["n_tokens_contains_component"]
        results[f"{ckpt_file}_n_blocks_contains_component"] = out["n_blocks_contains_component"]
        results[f"{ckpt_file}_accuracy_no_component"] = out["accuracy_no_component"]
        results[f"{ckpt_file}_n_tokens_no_component"] = out["n_tokens_no_component"]
        results[f"{ckpt_file}_n_blocks_no_component"] = out["n_blocks_no_component"]

        results[f"{ckpt_file}_accuracy_contains_either"] = out["accuracy_contains_either"]
        results[f"{ckpt_file}_n_tokens_contains_either"] = out["n_tokens_contains_either"]
        results[f"{ckpt_file}_n_blocks_contains_either"] = out["n_blocks_contains_either"]

        results[f"{ckpt_file}_accuracy_contains_neither"] = out["accuracy_contains_neither"]
        results[f"{ckpt_file}_n_tokens_contains_neither"] = out["n_tokens_contains_neither"]
        results[f"{ckpt_file}_n_blocks_contains_neither"] = out["n_blocks_contains_neither"]

        line = (
            f"\n{ckpt_file}:\n"
            f"  overall:              {out['accuracy_overall']:.6f} (n_tokens={out['n_tokens_overall']})\n"
            f"  contains_new:         {out['accuracy_contains_new']:.6f} (n_tokens={out['n_tokens_contains_new']}, n_blocks={out['n_blocks_contains_new']})\n"
            f"  no_new:               {out['accuracy_no_new']:.6f} (n_tokens={out['n_tokens_no_new']}, n_blocks={out['n_blocks_no_new']})\n"
            f"  contains_component:   {out['accuracy_contains_component']:.6f} (n_tokens={out['n_tokens_contains_component']}, n_blocks={out['n_blocks_contains_component']})\n"
            f"  no_component:         {out['accuracy_no_component']:.6f} (n_tokens={out['n_tokens_no_component']}, n_blocks={out['n_blocks_no_component']})\n"
            f"  either (new OR comp): {out['accuracy_contains_either']:.6f} (n_tokens={out['n_tokens_contains_either']}, n_blocks={out['n_blocks_contains_either']})\n"
            f"  neither:              {out['accuracy_contains_neither']:.6f} (n_tokens={out['n_tokens_contains_neither']}, n_blocks={out['n_blocks_contains_neither']})\n"
        )
        print(line)
        f.write(line)

    # Averages across ckpt1–3 (for convenience)
    avg_overall = np.mean([results[f"{c}_accuracy_overall"] for c in CHECKPOINT_FILES])
    avg_with = np.mean([results[f"{c}_accuracy_contains_new"] for c in CHECKPOINT_FILES])
    avg_without = np.mean([results[f"{c}_accuracy_no_new"] for c in CHECKPOINT_FILES])
    avg_component = np.mean([results[f"{c}_accuracy_contains_component"] for c in CHECKPOINT_FILES])
    avg_no_component = np.mean([results[f"{c}_accuracy_no_component"] for c in CHECKPOINT_FILES])
    avg_either = np.mean([results[f"{c}_accuracy_contains_either"] for c in CHECKPOINT_FILES])
    avg_neither = np.mean([results[f"{c}_accuracy_contains_neither"] for c in CHECKPOINT_FILES])

    results["average_accuracy_overall"] = float(avg_overall)
    results["average_accuracy_contains_new"] = float(avg_with)
    results["average_accuracy_no_new"] = float(avg_without)
    results["average_accuracy_contains_component"] = float(avg_component)
    results["average_accuracy_no_component"] = float(avg_no_component)
    results["average_accuracy_contains_either"] = float(avg_either)
    results["average_accuracy_contains_neither"] = float(avg_neither)

    summary = (
        f"\nAverage over {len(CHECKPOINT_FILES)} runs:\n"
        f"  overall:              {avg_overall:.6f}\n"
        f"  contains_new:         {avg_with:.6f}\n"
        f"  no_new:               {avg_without:.6f}\n"
        f"  contains_component:   {avg_component:.6f}\n"
        f"  no_component:         {avg_no_component:.6f}\n"
        f"  either (new OR comp): {avg_either:.6f}\n"
        f"  neither:              {avg_neither:.6f}\n"
    )
    print(summary)
    f.write(summary)

# ========== LOG TO WANDB ==========
wandb.log(results)
wandb.finish()