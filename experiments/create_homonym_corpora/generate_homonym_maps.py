"""
generate_homonym_maps.py
==========================

Generates JSON mapping files for inserting artificial homonyms.

Each file contains the top N homonym pairs from a CSV file, in the format:
{
    "word1,word2": "blend"
}

Usage:
------
python generate_homonym_maps.py
"""

import os
import json
import pandas as pd

# === SETTINGS ===
COUNTS = [1, 10, 20, 30, 40, 50]
DATASETS = {
    "recipe": "/home/nina/ambiguity/experiments/allkinds_dissimilar_pairs/recipe_homonym_pairs_final_shuffled.csv",
    "tiny": "/home/nina/ambiguity/experiments/allkinds_dissimilar_pairs/tiny_homonym_pairs_final_shuffled.csv",
}
OUTPUT_DIR = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps"


def create_map_file(df, dataset, count):
    """
    Create a homonym map JSON file containing the top N rows from the dataset CSV.

    Each entry maps "word1,word2" → "homonym_name".
    """
    map_dict = {
        f"{row['word1']},{row['word2']}": row['name']
        for _, row in df.head(count).iterrows()
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"{dataset}_{count}_HOM_map.json")

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(map_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ Created map: {filename} ({len(map_dict)} entries)")


def main():
    """Generate homonym map files for each dataset and N-count."""
    for dataset, csv_file in DATASETS.items():
        print(f"\n📄 Processing dataset: {dataset} from {csv_file}")
        df = pd.read_csv(csv_file)
        for count in COUNTS:
            create_map_file(df, dataset, count)

    print("\n🎉 All homonym map files generated successfully!")


if __name__ == "__main__":
    main()