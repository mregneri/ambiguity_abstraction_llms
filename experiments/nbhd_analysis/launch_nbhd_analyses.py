import os
import subprocess
import re

"""
launch_nbhd_analyses.py
========================
Launches neighborhood distance analyses for a curated set of Word2Vec models,
constructed from naming conventions.

Conventions
-----------
Models:
  word2vec_{base}_{count}_{kind}
  word2vec_{base}_base

Where
  base  ∈ {"recipe", "tiny"}
  count ∈ {1, 10, 20, 30, 40, 50}
  kind  ∈ {"HOM", "HYP", "HOM_HALF", "HYP_HALF"}

Model file (must exist):
  /home/nina/ambiguity/trained_models/allkinds/<model_name>/word2vec.bin

Maps (required for these kinds):
  HOM, HOM_HALF  → /home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/<map_name>_map.json
  HYP, HYP_HALF  → /home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/hypernym_maps/<map_name>_map.json
  BASE           → no map

Where <map_name> is <model_name> without the 'word2vec_' prefix.
"""

# === Config ===
python_cmd = "python"  # change to "python3" or a full path if needed
analyzer_script = "analyze_nbhd_distances.py"

BASE_MODEL_DIR = "/home/nina/ambiguity/trained_models/allkinds"
HOMONYM_MAP_DIR = "/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/"
HYPERNYM_MAP_DIR = "/home/nina/ambiguity/experiments/create_hypernym_corpora/allkinds/hypernym_maps/"

bases = ["recipe", "tiny"]
counts = [1, 10, 20, 30, 40, 50]
kinds = ["HOM", "HYP", "HOM_HALF", "HYP_HALF"]
include_base_models = False  # set False if you don't want to analyze base models


def model_path(model_name: str) -> str:
    return os.path.join(BASE_MODEL_DIR, model_name, "word2vec.bin")


def map_path_for_model(model_name: str) -> str | None:
    """
    Return the required map path for HOM/HYP families (including HALF variants).
    For HALF models we look up the corresponding full map name.
    """
    lower = model_name.lower()
    map_name = model_name.removeprefix("word2vec_")

    # Normalize HALF -> full for map lookup
    map_name_lookup = re.sub(r'_HOM_HALF$', '_HOM', map_name, flags=re.IGNORECASE)
    map_name_lookup = re.sub(r'_HYP_HALF$', '_HYP', map_name_lookup, flags=re.IGNORECASE)

    if lower.endswith("_hom") or lower.endswith("_hom_half"):
        return os.path.join(HOMONYM_MAP_DIR, f"{map_name_lookup}_map.json")
    if lower.endswith("_hyp") or lower.endswith("_hyp_half"):
        return os.path.join(HYPERNYM_MAP_DIR, f"{map_name_lookup}_map.json")
    if lower.endswith("_base"):
        return None
    return None


def main():
    scheduled = 0
    skipped = 0
    failed = 0
    completed = 0

    # --- Build explicit model list from conventions ---
    model_names = []

    # Base models (optional)
    if include_base_models:
        for base in bases:
            model_names.append(f"word2vec_{base}_base")

    # HOM/HYP (+ HALF) models
    for base in bases:
        for count in counts:
            for kind in kinds:
                model_names.append(f"word2vec_{base}_{count}_{kind}")

    # --- Iterate and launch ---
    for model_name in model_names:
        print(f"\n=== Neighborhood analysis: {model_name} ===")

        # Check model file
        mpath = model_path(model_name)
        if not os.path.isfile(mpath):
            print(f"⚠️  Skipping: model file not found: {mpath}")
            skipped += 1
            continue

        # Check map requirement (HOM/HYP families require maps)
        mappath = map_path_for_model(model_name)
        if mappath:
            print(f"Using resolved map path for lookup: {mappath}")
        if mappath is not None and not os.path.isfile(mappath):
            print(f"⚠️  Skipping: required map not found for {model_name}: {mappath}")
            skipped += 1
            continue

        cmd = [python_cmd, analyzer_script, model_name]
        print("Command:")
        print(" ", " ".join(cmd))

        try:
            scheduled += 1
            subprocess.run(cmd, check=True)
            completed += 1
            print(f"✅ Done: {model_name}")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"❌ Error while processing {model_name}: {e}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Scheduled: {scheduled}")
    print(f"Completed: {completed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()