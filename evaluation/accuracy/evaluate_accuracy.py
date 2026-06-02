"""
evaluate_accuracy.py
=====================

This script evaluates next-token accuracy for a trained GPT model using
three saved checkpoints (ckpt1.pt, ckpt2.pt, ckpt3.pt). It:

- Loads each checkpoint from a directory named after the model (passed as an argument)
- Evaluates accuracy on a fixed test set (`test.bin`)
- Logs results to Weights & Biases (wandb)
- Saves a plain-text summary to:
  /home/nina/ambiguity/evaluation/accuracy/allkinds/<model_name>/accuracy_results.txt

Usage:
    python evaluate_accuracy.py <model_name>

Example:
    python evaluate_accuracy.py gpt_recipe_1_HOM

Expected folder structure:
    /home/nina/ambiguity/trained_models/allkinds/<model_name>/ckpt*.pt
    /home/nina/ambiguity/gpt_pipelines/allkinds/data/<model_name>/test.bin
"""

import os
import sys
import torch
import numpy as np
import wandb

# Add path to access the GPT model code
sys.path.append("/home/nina/ambiguity/gpt_pipelines")
from model import GPTConfig, GPT

# ========== PARSE ARGUMENT ==========
if len(sys.argv) < 2:
    print("Usage: python evaluate_accuracy.py <model_name>")
    sys.exit(1)

# Model identifier (e.g. gpt_recipe_1_HOM or gpt_tiny_50_HYP_HALF)
MODEL_NAME = sys.argv[1]

# Define paths for checkpoints, test data, and result output
CHECKPOINT_DIR = f"/home/nina/ambiguity/trained_models/allkinds/{MODEL_NAME}"
CHECKPOINT_FILES = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
TEST_DATA_PATH = f"/home/nina/ambiguity/gpt_pipelines/allkinds/data/{MODEL_NAME}/test.bin"

# Create model-specific results directory
RESULTS_DIR = os.path.join(
    "/home/nina/ambiguity/evaluation/accuracy/allkinds", MODEL_NAME
)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "accuracy_results.txt")

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

wandb.init(
    project="ambiguity_allkinds",
    name=f"{MODEL_NAME}_eval",
    tags=tags,
    notes="Next-token accuracy on test set for ckpt1–3"
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

# ========== EVALUATE A SINGLE CHECKPOINT ==========
def evaluate_checkpoint(path):
    """Load a model from checkpoint and compute next-token accuracy."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for X, Y in get_full_batches():
            logits, _ = model(X, Y)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == Y).sum().item()
            total = Y.numel()
            correct_tokens += correct
            total_tokens += total

    accuracy = correct_tokens / total_tokens
    return accuracy

# ========== MAIN EVALUATION LOOP ==========
accuracies = {}
with open(RESULTS_FILE, "w") as f:
    f.write(f"Next-token accuracy evaluation for {MODEL_NAME}:\n")
    for ckpt_file in CHECKPOINT_FILES:
        path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        acc = evaluate_checkpoint(path)
        key = f"{ckpt_file}_accuracy"
        accuracies[key] = acc
        line = f"Accuracy for {ckpt_file}: {acc:.4f}"
        print(line)
        f.write(line + "\n")

    avg_acc = sum(accuracies.values()) / len(accuracies)
    accuracies["average_accuracy"] = avg_acc

    summary = f"\nAverage accuracy over {len(CHECKPOINT_FILES)} runs: {avg_acc:.4f}"
    print(summary)
    f.write(summary + "\n")

# ========== LOG TO WANDB ==========
wandb.log(accuracies)
wandb.finish()