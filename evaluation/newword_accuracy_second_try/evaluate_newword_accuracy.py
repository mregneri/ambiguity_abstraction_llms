"""
evaluate_newword_accuracy.py
====================================

Minimal-modification variant of evaluate_accuracy.py that additionally reports
next-token accuracy separately for:

- blocks whose context X contains any inserted new word (decode + word/punctuation split match)
- blocks whose context X contains no inserted new word (same detection rule)

Usage:
    python evaluate_newword_accuracy.py <model_name> <map_json>

Example:
    python evaluate_newword_accuracy.py gpt_recipe_1_HOM /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/recipe_1_HOM_map.json

Expected folder structure:
    /home/nina/ambiguity/trained_models/allkinds/<model_name>/ckpt*.pt
    /home/nina/ambiguity/gpt_pipelines/allkinds/data/<model_name>/test.bin

Output:
    /home/nina/ambiguity/evaluation/newword_accuracy_second_try/allkinds/<model_name>/newword_accuracy_results.txt
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
if len(sys.argv) < 3:
    print("Usage: python evaluate_newword_accuracy.py <model_name> <map_json>")
    sys.exit(1)

# Model identifier (e.g. gpt_recipe_1_HOM or gpt_tiny_50_HYP_HALF)
MODEL_NAME = sys.argv[1]
MAP_JSON = sys.argv[2]

# Define paths for checkpoints, test data, and result output
CHECKPOINT_DIR = f"/home/nina/ambiguity/trained_models/allkinds/{MODEL_NAME}"
CHECKPOINT_FILES = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
TEST_DATA_PATH = f"/home/nina/ambiguity/gpt_pipelines/allkinds/data/{MODEL_NAME}/test.bin"

# Create model-specific results directory
RESULTS_DIR = os.path.join(
    "/home/nina/ambiguity/evaluation/newword_accuracy_second_try/allkinds", MODEL_NAME
)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "newword_accuracy_results.txt")

# Evaluation settings
BLOCK_SIZE = 256
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

tags.append("eval")  # always include eval tag
tags.append("newword")  # distinguish from standard eval

wandb.init(
    project="ambiguity_allkinds",
    name=f"{MODEL_NAME}_newword_eval",
    tags=tags,
    notes="Next-token accuracy on test.bin for ckpt1–3, split by whether X contains a new word"
)

# ========== LOAD TEST DATA ==========
test_data = np.memmap(TEST_DATA_PATH, dtype=np.uint16, mode='r')

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
    Load new word surface forms from a replacements map.

    Expected map format (simple dict):
        { "base1,base2": "newtoken", ... }

    We interpret the dict VALUES as the inserted new words.
    """
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_words = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str) and v.strip():
                new_words.append(v.strip())
    return sorted(set(new_words))

enc = tiktoken.get_encoding("gpt2")
NEW_WORDS = load_new_words_from_map(MAP_JSON)

# --- Use text-level detection mirroring insertion rules ---
# In the pseudoword insertion scripts, replacements happen only when a token is split into
# parts via r'\w+|[^\w\s]' and a \w+ part matches a form (case-insensitive).
# We mirror that here: decode X -> split -> if any \w+ part equals a new word (lowercased).

NEW_WORD_SET = {w.lower() for w in NEW_WORDS if w.strip()}

_SPLIT_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)
_WORD_RE = re.compile(r'^\w+$', re.UNICODE)

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

# ========== EVALUATE A SINGLE CHECKPOINT ==========
def evaluate_checkpoint(path):
    """Load a model from checkpoint and compute next-token accuracy split by whether X contains a new word."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
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

    with torch.no_grad():
        for X, Y in get_full_batches():
            logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)

            # Per-row correct counts
            correct_per_row = (preds == Y).sum(dim=1)  # [B]
            total_per_row = torch.full((Y.size(0),), Y.size(1), device=Y.device, dtype=torch.long)  # [B]

            # Determine which rows contain a new word in X
            mask_with = []
            for b in range(X.size(0)):
                mask_with.append(x_contains_any_newword(X[b]))
            mask_with = torch.tensor(mask_with, device=Y.device, dtype=torch.bool)
            mask_without = ~mask_with

            # Update diagnostics
            num_blocks_with += mask_with.sum().item()
            num_blocks_without += mask_without.sum().item()

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

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else float("nan")
    accuracy_with = correct_tokens_with / total_tokens_with if total_tokens_with > 0 else float("nan")
    accuracy_without = correct_tokens_without / total_tokens_without if total_tokens_without > 0 else float("nan")

    return {
        "accuracy_overall": accuracy,
        "accuracy_contains_new": accuracy_with,
        "accuracy_no_new": accuracy_without,
        "n_tokens_overall": total_tokens,
        "n_tokens_contains_new": total_tokens_with,
        "n_tokens_no_new": total_tokens_without,
        "n_blocks_contains_new": num_blocks_with,
        "n_blocks_no_new": num_blocks_without,
    }

# ========== MAIN EVALUATION LOOP ==========
results = {}
with open(RESULTS_FILE, "w") as f:
    f.write(f"New-word split next-token accuracy evaluation for {MODEL_NAME}:\n")
    f.write(f"Test data: {TEST_DATA_PATH}\n")
    f.write(f"Map JSON:  {MAP_JSON}\n")
    f.write(f"New words ({len(NEW_WORDS)}): {NEW_WORDS}\n")
    f.write("New word detection: decode X -> split with r'\\w+|[^\\w\\s]' -> match \\w+ parts (case-insensitive)\n\n")

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

        line = (
            f"\n{ckpt_file}:\n"
            f"  overall:       {out['accuracy_overall']:.6f} (n_tokens={out['n_tokens_overall']})\n"
            f"  contains_new:  {out['accuracy_contains_new']:.6f} (n_tokens={out['n_tokens_contains_new']}, n_blocks={out['n_blocks_contains_new']})\n"
            f"  no_new:        {out['accuracy_no_new']:.6f} (n_tokens={out['n_tokens_no_new']}, n_blocks={out['n_blocks_no_new']})\n"
        )
        print(line)
        f.write(line)

    # Averages across ckpt1–3 (for convenience)
    avg_overall = np.mean([results[f"{c}_accuracy_overall"] for c in CHECKPOINT_FILES])
    avg_with = np.mean([results[f"{c}_accuracy_contains_new"] for c in CHECKPOINT_FILES])
    avg_without = np.mean([results[f"{c}_accuracy_no_new"] for c in CHECKPOINT_FILES])

    results["average_accuracy_overall"] = float(avg_overall)
    results["average_accuracy_contains_new"] = float(avg_with)
    results["average_accuracy_no_new"] = float(avg_without)

    summary = (
        f"\nAverage over {len(CHECKPOINT_FILES)} runs:\n"
        f"  overall:      {avg_overall:.6f}\n"
        f"  contains_new: {avg_with:.6f}\n"
        f"  no_new:       {avg_without:.6f}\n"
    )
    print(summary)
    f.write(summary)

# ========== LOG TO WANDB ==========
wandb.log(results)
wandb.finish()