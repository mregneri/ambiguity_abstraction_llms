"""
launch_homonym_insertions.py
=============================
This script launches insertions of artificial homonyms into tokenized training, validation,
and test corpora for various config files sequentially, using "insert_artificial_homonyms.py".

Usage:
    python launch_homonym_insertions.py
"""

import subprocess

# List of config file names
model_bases = ["recipe", "tiny"]
hom_counts = [1, 10, 20, 30, 40, 50]   # [1, 50]

config_names = [f"insert_{base}_{count}_HOM.py" for base in model_bases for count in hom_counts]

# Path to insertion script
INSERTION_SCRIPT = "insert_artificial_homonyms.py"

def main():
    for config_name in config_names:
        config_path = f"allkinds/config/{config_name}"
        print(f"\n=== Starting homonym insertion for config: {config_name} ===")

        result = subprocess.run(
            ["python", INSERTION_SCRIPT, "--config", config_path]
        )

        if result.returncode != 0:
            print(f"\n❌ Error in {config_name}")
            break
        print(f"\n✅ Finished {config_name} successfully.")
    print("\n🎉 All homonym insertions completed!")

if __name__ == "__main__":
    main()