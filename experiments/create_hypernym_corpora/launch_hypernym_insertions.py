"""
launch_hypernym_insertions.py
==============================
This script launches insertions of artificial hypernyms into tokenized training, validation,
and test corpora for various config files sequentially, using "insert_artificial_hypernyms.py".

Usage:
    python launch_hypernym_insertions.py
"""

import subprocess

# List of config file names
model_bases = ["recipe", "tiny"]
hyp_counts = [1, 10, 20, 30, 40, 50]

config_names = [f"insert_{base}_{count}_HYP.py" for base in model_bases for count in hyp_counts]

# Path to insertion script
INSERTION_SCRIPT = "insert_artificial_hypernyms.py"

def main():
    for config_name in config_names:
        config_path = f"allkinds/config/{config_name}"
        print(f"\n=== Starting hypernym insertion for config: {config_name} ===")

        result = subprocess.run(
            ["python", INSERTION_SCRIPT, "--config", config_path]
        )

        if result.returncode != 0:
            print(f"\n❌ Error in {config_name}")
            break
        print(f"\n✅ Finished {config_name} successfully.")
    print("\n🎉 All hypernym insertions completed!")

if __name__ == "__main__":
    main()