import os
import subprocess

"""
launch_word2vec_trainings.py
============================
Launch Word2Vec training runs by calling `word2vec_pipeline_new.py`
with *separate* train and val inputs for multiple dataset variants.

Conventions
-----------
Inputs (must exist):
  ../datasets/allkinds/{base}_{count}_{kind}_train.json
  ../datasets/allkinds/{base}_{count}_{kind}_val.json

Output dir per run:
  ../trained_models/allkinds/word2vec_{base}_{count}_{kind}

Supported dimensions:
  base  ∈ {"recipe", "tiny"}               # extend as needed
  count ∈ {1, 10, 20, 30, 40, 50}          # extend as needed
  kind  ∈ {"HOM", "HYP", "HOM_HALF", "HYP_HALF"}

Each run trains on train ∪ val and writes:
  - preprocessed_trainval.json
  - word2vec.model
  - word2vec.bin
  - training_log.txt
"""

# === Configuration (edit as needed) ===
python_cmd = "python"  # adjust if you need a specific interpreter, e.g., "python3"
pipeline_script = "word2vec_pipeline_new.py"

bases = ["recipe", "tiny"]
counts = [1, 10, 20, 30, 40, 50]
kinds = ["HOM", "HYP", "HOM_HALF", "HYP_HALF"]

datasets_dir = "../datasets/allkinds"
models_dir = "../trained_models/allkinds"

# === Launch ===
scheduled = 0
skipped = 0
failed = 0
completed = 0

for base in bases:
    for count in counts:
        for kind in kinds:
            name = f"{base}_{count}_{kind}"

            train_path = os.path.join(datasets_dir, f"{name}_train.json")
            val_path = os.path.join(datasets_dir, f"{name}_val.json")
            output_dir = os.path.join(models_dir, f"word2vec_{name}")

            print(f"\n=== Training: {name} ===")

            # Check inputs
            missing = []
            if not os.path.isfile(train_path):
                missing.append(train_path)
            if not os.path.isfile(val_path):
                missing.append(val_path)

            if missing:
                print("⚠️  Skipping run due to missing files:")
                for m in missing:
                    print(f"   - {m}")
                skipped += 1
                continue

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                python_cmd,
                pipeline_script,
                "--train", train_path,
                "--val", val_path,
                "--out_dir", output_dir
            ]

            print("Command:")
            print(" ", " ".join(cmd))

            try:
                scheduled += 1
                subprocess.run(cmd, check=True)
                completed += 1
                print(f"✅ Finished: {name}")
            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"❌ Error while training model for {name}: {e}")

print("\n=== Summary ===")
print(f"Scheduled: {scheduled}")
print(f"Completed: {completed}")
print(f"Skipped (missing inputs): {skipped}")
print(f"Failed: {failed}")