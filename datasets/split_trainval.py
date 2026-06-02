import os
import json
import random
from datetime import datetime

"""
Splits existing trainval corpora into separate train and val sets.

Input:
- recipe_base_trainval.json
- tiny_base_trainval.json

For each corpus:
- Loads the trainval set (list of tokenized documents)
- Shuffles with fixed seed
- Splits into:
  - 90% → train
  - 10% → val
- Saves:
  - recipe_base_train.json / recipe_base_val.json
  - tiny_base_train.json / tiny_base_val.json
Also writes a log file: split_trainval_log.txt
"""

# === Settings ===
SEED = 42
VAL_RATIO = 0.10

# === Paths (assumes files are in same directory as script) ===
BASE_DIR = os.path.dirname(__file__)
FILES = {
    "recipe": {
        "input": "recipe_base_trainval.json",
        "train_out": "recipe_base_train.json",
        "val_out": "recipe_base_val.json"
    },
    "tiny": {
        "input": "tiny_base_trainval.json",
        "train_out": "tiny_base_train.json",
        "val_out": "tiny_base_val.json"
    }
}

LOG_FILENAME = "split_trainval_log.txt"

# === Helpers ===
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_trainval(data, val_ratio):
    random.shuffle(data)
    val_size = max(1, int(len(data) * val_ratio))
    val = data[:val_size]
    train = data[val_size:]
    return train, val

def pct(part, whole):
    return (part / whole) * 100 if whole > 0 else 0.0

# === Main ===
def main():
    random.seed(SEED)
    log_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_lines.append(f"=== Train/Val Split Log ===\nTimestamp: {timestamp}\n")

    for name, paths in FILES.items():
        input_path = os.path.join(BASE_DIR, paths["input"])
        train_out_path = os.path.join(BASE_DIR, paths["train_out"])
        val_out_path = os.path.join(BASE_DIR, paths["val_out"])

        if not os.path.exists(input_path):
            print(f"[SKIP] {input_path} not found.")
            log_lines.append(f"[SKIP] {input_path} not found.\n")
            continue

        print(f"📂 Processing {input_path}...")
        data = load_json(input_path)
        train, val = split_trainval(data, VAL_RATIO)

        save_json(train, train_out_path)
        save_json(val, val_out_path)

        print(f"✅ Saved {len(train)} train docs → {paths['train_out']}")
        print(f"✅ Saved {len(val)} val docs   → {paths['val_out']}")
        print()

        total = len(train) + len(val)
        log_lines.append(f"Corpus: {name}\n")
        log_lines.append(f"  Input: {paths['input']}\n")
        log_lines.append(f"  Train: {len(train)} docs ({pct(len(train), total):.2f}%) → {paths['train_out']}\n")
        log_lines.append(f"  Val:   {len(val)} docs ({pct(len(val), total):.2f}%) → {paths['val_out']}\n\n")

    log_path = os.path.join(BASE_DIR, LOG_FILENAME)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.writelines(log_lines)

    print(f"📝 Wrote log to {LOG_FILENAME}")

if __name__ == "__main__":
    main()