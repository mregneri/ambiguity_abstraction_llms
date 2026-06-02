"""
generate_inflection_maps.py
===========================

Generates master inflection maps for both hypernym (HYP) and homonym (HOM)
datasets directly from their respective CSV files.

Each master map contains all inflected forms for every base word pair
appearing in the dataset-specific CSVs.

The resulting files serve as canonical sources for all later experiments.
They can be used to generate smaller per-corpus subsets automatically.

Output format (example)
-----------------------
{
  "bracelace": {
    "bases": ["necklace", "bracelet"],
    "forms": {
      "necklace": ["necklace", "necklaces"],
      "bracelet": ["bracelet", "bracelets"]
    }
  },
  ...
}

Suspicious entries (words with 0, 1, or >10 inflections) are logged to:
`generate_inflection_maps_log.txt`

Usage:
------
python generate_inflection_maps.py
"""

import os
import json
import pandas as pd
from lemminflect import getAllInflections

# Input CSV files
RECIPE_HYP_CSV = "/home/nina/ambiguity/experiments/allkinds_similar_pairs/recipe_hypernym_pairs_final_shuffled.csv"
TINY_HYP_CSV = "/home/nina/ambiguity/experiments/allkinds_similar_pairs/tiny_hypernym_pairs_final_shuffled.csv"
RECIPE_HOM_CSV = "/home/nina/ambiguity/experiments/allkinds_dissimilar_pairs/recipe_homonym_pairs_final_shuffled.csv"
TINY_HOM_CSV = "/home/nina/ambiguity/experiments/allkinds_dissimilar_pairs/tiny_homonym_pairs_final_shuffled.csv"

# Output directory and files
OUTPUT_DIR = "/home/nina/ambiguity/experiments/inflection_maps/allkinds"
RECIPE_HYP_JSON = "/home/nina/ambiguity/experiments/inflection_maps/allkinds/recipe_master_HYP_inflections_raw.json"
TINY_HYP_JSON = "/home/nina/ambiguity/experiments/inflection_maps/allkinds/tiny_master_HYP_inflections_raw.json"
RECIPE_HOM_JSON = "/home/nina/ambiguity/experiments/inflection_maps/allkinds/recipe_master_HOM_inflections_raw.json"
TINY_HOM_JSON = "/home/nina/ambiguity/experiments/inflection_maps/allkinds/tiny_master_HOM_inflections_raw.json"
LOG_PATH = "/home/nina/ambiguity/experiments/inflection_maps/allkinds/generate_inflection_maps_log.txt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Structured dataset definitions
DATASETS = {
    "HYP": {
        "recipe": {"csv": RECIPE_HYP_CSV, "out": RECIPE_HYP_JSON},
        "tiny": {"csv": TINY_HYP_CSV, "out": TINY_HYP_JSON},
    },
    "HOM": {
        "recipe": {"csv": RECIPE_HOM_CSV, "out": RECIPE_HOM_JSON},
        "tiny": {"csv": TINY_HOM_CSV, "out": TINY_HOM_JSON},
    },
}


# === HELPER FUNCTIONS ===
def get_inflections(word):
    """Return all unique inflected forms of a word using lemminflect."""
    infl = getAllInflections(word)
    forms = set()
    for form_list in infl.values():
        forms.update(form_list)
    return sorted(forms)


def log_suspicious(word, forms, log_lines):
    """Log words with missing or suspicious inflection counts."""
    if len(forms) == 0:
        log_lines.append(f"[MISSING] {word} → 0 forms\n")
    elif len(forms) == 1:
        log_lines.append(f"[FEW] {word} → 1 form: {forms}\n")
    elif len(forms) > 10:
        log_lines.append(f"[MANY] {word} → {len(forms)} forms: {forms[:10]}...\n")


def generate_master_map(dataset_name, csv_path, variant, output_path):
    """Generate a master inflection map (HYP or HOM) from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"\n📄 Loaded {len(df)} rows from {csv_path}")

    master_map = {}
    log_lines = [
        f"=== {dataset_name.upper()} {variant.upper()} MASTER INFLECTION MAP ===\n",
        f"Input CSV: {csv_path}\n",
    ]

    # Both hypernym and homonym CSVs now use the same column name for the artificial word
    cols = {"word1": "word1", "word2": "word2", "artificial": "name"}

    for _, row in df.iterrows():
        try:
            w1, w2, artificial = (
                str(row[cols["word1"]]).strip(),
                str(row[cols["word2"]]).strip(),
                str(row[cols["artificial"]]).strip(),
            )
        except KeyError as e:
            raise KeyError(f"CSV {csv_path} missing expected column: {e}")

        base_forms = {}
        for base in (w1, w2):
            forms = get_inflections(base)
            base_forms[base] = forms
            log_suspicious(base, forms, log_lines)

        master_map[artificial] = {"bases": [w1, w2], "forms": base_forms}

    # Write output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_map, f, ensure_ascii=False, indent=2)

    log_lines.append(f"Output JSON: {output_path}\n")
    log_lines.append(f"Total entries: {len(master_map)}\n\n")

    print(f"✅ Saved master inflection map: {output_path}")
    return log_lines


# === MAIN ===
def main():
    all_logs = ["MASTER INFLECTION MAP GENERATION LOG\n", "=================================\n\n"]

    for variant, datasets in DATASETS.items():
        for dataset_name, paths in datasets.items():
            csv_path = paths["csv"]
            output_path = paths["out"]

            if not os.path.exists(csv_path):
                print(f"⚠️  Skipping {dataset_name} ({variant}): file not found at {csv_path}")
                continue

            logs = generate_master_map(dataset_name, csv_path, variant, output_path)
            all_logs.extend(logs)
            all_logs.append("\n\n")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.writelines(all_logs)

    print(f"\n📝 Wrote log to {LOG_PATH}")
    print("\n🎉 All master inflection maps (HYP + HOM) generated successfully!")


if __name__ == "__main__":
    main()