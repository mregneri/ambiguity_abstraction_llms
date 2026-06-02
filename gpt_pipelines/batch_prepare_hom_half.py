"""
batch_prepare_hom_half.py
=========================

Runs `prepare.py` for multiple GPT datasets based on 50% artificial homonym corpora.

Assumes:
- Input JSONs are located in /home/nina/ambiguity/datasets/allkinds
- Each corpus has pre-split files:
    - {base}_{count}_HOM_HALF_train.json
    - {base}_{count}_HOM_HALF_val.json
    - {base}_{count}_HOM_HALF_test.json
- Output folders will be created inside:
    /home/nina/ambiguity/gpt_pipelines/allkinds/data
- Output naming follows: gpt_{base}_{count}_HOM_HALF

Usage:
------
python batch_prepare_hom_half.py
"""

import subprocess
import os

# === Configuration ===
model_bases = ["recipe", "tiny"]
hom_counts = [1, 10, 20, 30, 40, 50]

DATASET_DIR = "/home/nina/ambiguity/datasets/allkinds"
OUTPUT_DIR = "/home/nina/ambiguity/gpt_pipelines/allkinds/data"

# === Run preparations ===
for base in model_bases:
    for count in hom_counts:
        name = f"{base}_{count}_HOM_HALF"
        train_path = os.path.join(DATASET_DIR, f"{name}_train.json")
        val_path = os.path.join(DATASET_DIR, f"{name}_val.json")
        test_path = os.path.join(DATASET_DIR, f"{name}_test.json")
        out_dir = os.path.join(OUTPUT_DIR, f"gpt_{name}")

        print(f"\n=== Running prepare.py for: {name} ===")

        cmd = [
            "python", "prepare.py",
            "--train", train_path,
            "--val", val_path,
            "--test", test_path,
            "--out_dir", out_dir
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"⚠️ prepare.py failed for: {name}")
            break
        else:
            print(f"✅ Finished: {name}")

print("\n🎉 All HOM_HALF datasets prepared successfully!")