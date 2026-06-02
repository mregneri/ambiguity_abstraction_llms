import csv
import json
import os

# ===================== CONFIG =====================

# Input CSV files (add more if needed)
CSV_FILES = [
    "recipe_artificial_homonyms_shuffled.csv",
    "tiny_artificial_homonyms_shuffled.csv"
]

# Output directory
OUTPUT_DIR = "json_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of entries per output file
CHUNK_SIZES = [10, 20, 30, 40, 50]

# ================ PROCESSING LOGIC ================

def create_json_files(csv_path):
    # Get base name prefix like "recipe" or "tiny"
    base = os.path.basename(csv_path)
    prefix = base.split("_")[0]

    with open(csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    for size in CHUNK_SIZES:
        data = {
            f"{row['word1']},{row['word2']}": row['name']
            for row in rows[:size]
        }

        output_filename = f"{prefix}_{size}_HOM_map.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=2)
        print(f"Saved: {output_path}")

# ================ MAIN =================

if __name__ == "__main__":
    for csv_file in CSV_FILES:
        create_json_files(csv_file)