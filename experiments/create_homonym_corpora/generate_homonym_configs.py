"""
generate_homonym_configs.py
=============================

Generates config files for inserting artificial homonyms into tokenized corpora.

Each config references the base corpora (train, val, test), output corpora,
the homonym map, the verified master inflection map, and the summary log path.
"""

import os

# === SETTINGS ===
output_dir = "allkinds/config"
base_corpora_dir = "/home/nina/ambiguity/datasets/base"
allkinds_corpora_dir = "/home/nina/ambiguity/datasets/allkinds"
maps_base_dir = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps"
logs_base_dir = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/logs"
master_inflections_dir = "/home/nina/ambiguity/experiments/inflection_maps/allkinds"

counts = [1, 10, 20, 30, 40, 50]
datasets = ["recipe", "tiny"]

def make_config(dataset, count):
    name = f"{dataset}_{count}_HOM"

    # Choose the correct master map
    master_inflections = (
        f"{master_inflections_dir}/{dataset}_master_HOM_inflections_edited.json"
    )

    config_text = f"""# config/insert_{name}.py
config = {{
    'train_in': '{base_corpora_dir}/{dataset}_base_train.json',
    'val_in': '{base_corpora_dir}/{dataset}_base_val.json',
    'test_in': '{base_corpora_dir}/{dataset}_base_test.json',
    'train_out': '{allkinds_corpora_dir}/{name}_train.json',
    'val_out': '{allkinds_corpora_dir}/{name}_val.json',
    'test_out': '{allkinds_corpora_dir}/{name}_test.json',
    'replacements': '{maps_base_dir}/{name}_map.json',
    'master_inflections': '{master_inflections}',
    'summary_log': '{logs_base_dir}/{name}_summary.txt'
}}
"""
    filename = os.path.join(output_dir, f"insert_{name}.py")
    os.makedirs(output_dir, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"✅ Created: {filename}")

def main():
    for dataset in datasets:
        for count in counts:
            make_config(dataset, count)
    print("\n🎉 All homonym config files generated!")

if __name__ == "__main__":
    main()