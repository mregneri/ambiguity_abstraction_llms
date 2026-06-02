"""
generate_hom_half_configs.py
=============================

Generates config files for inserting artificial homonyms (50% replacements)
into tokenized training, validation, and test corpora.

Each config file specifies:
- Input and output corpus paths (train / val / test)
- The corresponding homonym replacement map
- The master inflection map (shared for all HOM corpora)
- The path for the output summary log

Usage:
------
python generate_hom_half_configs.py
"""

import os

# === SETTINGS ===
output_dir = "allkinds/config"

# Base directories
base_corpora_dir = "/home/nina/ambiguity/datasets/base"
half_corpora_dir = "/home/nina/ambiguity/datasets/allkinds"

# Use allkinds homonym maps from create_homonym_corpora
maps_base_dir = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps"
logs_base_dir = "/home/nina/ambiguity/experiments/create_hom_half_corpora/allkinds/logs"
master_inflections_dir = "/home/nina/ambiguity/experiments/inflection_maps/allkinds"

# Counts and datasets
counts = [1, 10, 20, 30, 40, 50]
datasets = ["recipe", "tiny"]

# Master inflection maps (edited, verified)
master_inflections = {
    "recipe": f"{master_inflections_dir}/recipe_master_HOM_inflections_edited.json",
    "tiny": f"{master_inflections_dir}/tiny_master_HOM_inflections_edited.json",
}


def make_config(dataset, count):
    """Create one config file for a given dataset and count."""
    name = f"{dataset}_{count}_HOM_HALF"

    config_text = f"""# config/insert_{name}.py
config = {{
    'train_in': '{base_corpora_dir}/{dataset}_base_train.json',
    'val_in': '{base_corpora_dir}/{dataset}_base_val.json',
    'test_in': '{base_corpora_dir}/{dataset}_base_test.json',

    'train_out': '{half_corpora_dir}/{name}_train.json',
    'val_out': '{half_corpora_dir}/{name}_val.json',
    'test_out': '{half_corpora_dir}/{name}_test.json',

    'replacements': '{maps_base_dir}/{dataset}_{count}_HOM_map.json',
    'master_inflections': '{master_inflections[dataset]}',
    'summary_log': '{logs_base_dir}/{name}_summary.txt'
}}
"""

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"insert_{name}.py")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"✅ Created: {filename}")


def main():
    for dataset in datasets:
        for count in counts:
            make_config(dataset, count)
    print("\n🎉 All HOM_HALF config files generated!")


if __name__ == "__main__":
    main()