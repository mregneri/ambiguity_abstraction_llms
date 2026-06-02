"""
train_all_models.py
=========================

Sequentially trains and evaluates multiple GPT models for both homonym and hypernym datasets.
Each model is trained in 3 independent runs via `launch_training_runs.py`
and immediately evaluated using `evaluate_accuracy.py`.

Includes both full and half (50%) insertion variants.

Usage:
    python train_all_models.py
"""

import os
import subprocess

# === SETTINGS ===
model_bases = ["gpt_recipe", "gpt_tiny"]
ambiguity_types = ["HOM", "HYP"]         # covers both homonyms and hypernyms
half_variants = ["", "_HALF"]            # full + half versions
counts = [1, 10, 20, 30, 40, 50]

# Generate all combinations like:
# gpt_recipe_1_HOM, gpt_recipe_1_HOM_HALF, gpt_tiny_50_HYP, ...
model_names = [
    f"{base}_{count}_{ambig}{half}"
    for base in model_bases
    for count in counts
    for ambig in ambiguity_types
    for half in half_variants
]

# Paths to scripts
BASE_DIR = "/home/nina/ambiguity"
LAUNCH_SCRIPT = os.path.join(BASE_DIR, "gpt_pipelines/launch_training_runs.py")
EVAL_SCRIPT = os.path.join(BASE_DIR, "evaluation/accuracy/evaluate_accuracy.py")

def main():
    total = len(model_names)
    print(f"\n🚀 Starting sequential GPT trainings + evaluations for {total} models...\n")

    for i, model_name in enumerate(model_names, 1):
        print("=" * 90)
        print(f"🔹 [{i}/{total}] Training model: {model_name}")
        print("=" * 90)

        # --- Training phase ---
        train_result = subprocess.run(["python", LAUNCH_SCRIPT, model_name])
        if train_result.returncode != 0:
            print(f"❌ Training failed for {model_name}. Stopping.\n")
            break
        print(f"✅ Training completed for: {model_name}")

        # --- Evaluation phase ---
        print(f"\n📊 Evaluating model: {model_name}")
        eval_result = subprocess.run(["python", EVAL_SCRIPT, model_name])
        if eval_result.returncode != 0:
            print(f"❌ Evaluation failed for {model_name}. Stopping.\n")
            break
        print(f"✅ Evaluation completed for: {model_name}\n")

    else:
        print("\n🎉 All homonym and hypernym model trainings + evaluations completed successfully!\n")

if __name__ == "__main__":
    main()