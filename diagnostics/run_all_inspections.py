"""
run_all_inspections.py
=======================

Runs inspection (`inspect_inserted_artificials.py`) for all
HOM, HYP, HOM_HALF, and HYP_HALF corpora sequentially.

Config locations:
- HOM:       /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/config
- HYP:       /home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/config
- HOM_HALF:  /home/nina/ambiguity/experiments/create_hom_half_corpora/allkinds/config
- HYP_HALF:  /home/nina/ambiguity/experiments/create_hyp_half_corpora/allkinds/config

Inspector location:
- /home/nina/ambiguity/diagnostics/inspect_inserted_artificials.py

Usage:
------
python run_all_inspections.py
"""

import subprocess
import os

# === SETTINGS ===
datasets = ["recipe", "tiny"]

# Full corpora counts
full_counts = [1, 10, 20, 30, 40, 50]

# Half corpora counts
half_counts = [1, 10, 20, 30, 40, 50]

# Config directories
config_dirs = {
    "HOM":      "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/config",
    "HYP":      "/home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/config",
    "HOM_HALF": "/home/nina/ambiguity/experiments/create_hom_half_corpora/allkinds/config",
    "HYP_HALF": "/home/nina/ambiguity/experiments/create_hyp_half_corpora/allkinds/config",
}

# Inspector script path
INSPECT_SCRIPT = "/home/nina/ambiguity/diagnostics/inspect_inserted_artificials.py"

def main():
    for dataset in datasets:
        for variant, cfg_dir in config_dirs.items():
            counts = full_counts if "HALF" not in variant else half_counts

            for count in counts:
                config_name = f"insert_{dataset}_{count}_{variant}.py"
                config_path = os.path.join(cfg_dir, config_name)

                print(f"\n=== Inspecting {dataset}_{count}_{variant} ===")

                result = subprocess.run(
                    ["python", INSPECT_SCRIPT, "--config", config_path]
                )

                if result.returncode != 0:
                    print(f"❌ Inspection failed for {config_name}")
                    return  # stop on error
                else:
                    print(f"✅ Finished inspection for {config_name}")

    print("\n🎉 All inspections completed successfully!")

if __name__ == "__main__":
    main()