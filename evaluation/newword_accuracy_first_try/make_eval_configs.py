"""
make_eval_configs.py
===================

Purpose
-------
Generate evaluation config files for the new-word accuracy evaluator
(`measure_newword_accuracy.py`). A config file is produced for either:
  • a single FULL model name (e.g., "gpt_recipe_1_HOM"), or
  • all FULL model names across a predefined grid (`--all`).

Each config contains only the *required* keys expected by the evaluator:
    kind, map_json, full_model_dir, half_model_dir, base_model_dir,
    full_test_json, half_test_json, base_test_json, out_dir

Where configs are written
-------------------------
/home/nina/ambiguity/evaluation/newword_accuracy/config/eval_<FULL_MODEL>.py

Where evaluator outputs will go
-------------------------------
/home/nina/ambiguity/evaluation/newword_accuracy/results/<FULL_MODEL>/
    ├── newword_accuracy.csv
    └── newword_accuracy.json

Assumptions / Naming conventions
--------------------------------
• FULL/ HALF/ BASE model directories:
    /home/nina/ambiguity/trained_models/<model_name>/
      └── ckpt{1,2,3}.pt

• Test JSON file names:
    FULL: /home/nina/ambiguity/datasets/<corpus>_<count>_<kind>_test.json
    HALF: /home/nina/ambiguity/datasets/<corpus>_<count>_<kind>_HALF_test.json
    BASE: /home/nina/ambiguity/datasets/<corpus>_base_test.json
  (Each JSON is a list of token lists per document.)

• Map file names:
    HOM: /home/nina/ambiguity/experiments/create_homonym_corpora/homonym_maps/<corpus>_<count>_map.json
    HYP: /home/nina/ambiguity/experiments/create_hypernym_corpora/hypernym_maps/<corpus>_<count>_map.json

• Model name format for FULL:
    gpt_<corpus>_<count>_<kind>
  e.g., gpt_recipe_1_HOM, gpt_tiny_50_HYP

Usage
-----
# Single FULL model
python make_eval_configs.py gpt_recipe_1_HOM

# All FULL models in the predefined grid
python make_eval_configs.py --all
"""

import os
import sys
from pathlib import Path
from textwrap import dedent

# --- Root paths (keep in sync with your pipeline) ---
BASE = "/home/nina/ambiguity"
EVAL_DIR = f"{BASE}/evaluation/newword_accuracy"
CONFIG_DIR = f"{EVAL_DIR}/config"
TRAINED_DIR = f"{BASE}/trained_models"
DATASETS_DIR = f"{BASE}/datasets"

HOMO_MAP_DIR = f"{BASE}/experiments/create_homonym_corpora/homonym_maps"
HYPER_MAP_DIR = f"{BASE}/experiments/create_hypernym_corpora/hypernym_maps"

Path(CONFIG_DIR).mkdir(parents=True, exist_ok=True)


def parse_model_name(full_model: str):
    """
    Parse a FULL model name like 'gpt_recipe_1_HOM' (or 'gpt_recipe_1_HOM_HALF').
    Returns:
        base_tag: 'gpt_recipe' or 'gpt_tiny'
        count   : e.g. '1', '10', ...
        kind    : 'HOM' or 'HYP'
        is_half : bool (True if *_HALF)
    Raises:
        ValueError if the format is unexpected.
    """
    parts = full_model.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected model name format: {full_model}")
    base_tag = parts[0] + "_" + parts[1]
    count = parts[2]
    kind = parts[3]
    is_half = (len(parts) >= 5 and parts[4].lower() == "half")
    return base_tag, count, kind, is_half


def normalized_pair(full_or_half: str):
    """
    Normalize any FULL/HALF input name to the canonical trio:
      FULL: gpt_<corpus>_<count>_<kind>
      HALF: <FULL>_HALF
      BASE: gpt_<corpus>_base
    Also returns: (corpus, count, kind)
    """
    base_tag, count, kind, _ = parse_model_name(full_or_half)
    full_name = f"{base_tag}_{count}_{kind}"
    half_name = f"{full_name}_HALF"
    corpus = "recipe" if "recipe" in base_tag else "tiny"
    base_name = f"gpt_{corpus}_base"
    return full_name, half_name, base_name, corpus, count, kind


def map_path(corpus: str, count: str, kind: str) -> str:
    """
    Build the path to the homonym/hypernym map file for the given (corpus, count, kind).

    Expected naming pattern:
        <corpus>_<count>_<kind>_map.json
    Example:
        recipe_1_HOM_map.json
    """
    kind = kind.upper()
    base_dir = HOMO_MAP_DIR if kind == "HOM" else HYPER_MAP_DIR
    path = os.path.join(base_dir, f"{corpus}_{count}_{kind}_map.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected map file not found: {path}")
    return path


def test_json_paths(corpus: str, count: str, kind: str):
    """
    Return the (FULL, HALF, BASE) test JSON paths following the dataset naming scheme.
    """
    full = os.path.join(DATASETS_DIR, f"{corpus}_{count}_{kind}_test.json")
    half = os.path.join(DATASETS_DIR, f"{corpus}_{count}_{kind}_HALF_test.json")
    base = os.path.join(DATASETS_DIR, f"{corpus}_base_test.json")
    return full, half, base


def write_config(full_model: str) -> str:
    """
    Create the eval config file for a given FULL model name (or *_HALF input).
    Returns:
        Path to the generated config file.
    """
    full_name, half_name, base_name, corpus, count, kind = normalized_pair(full_model)
    map_json = map_path(corpus, count, kind)
    full_test, half_test, base_test = test_json_paths(corpus, count, kind)

    # Evaluator output directory for this FULL model
    out_dir = os.path.join(EVAL_DIR, "results", full_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg_path = os.path.join(CONFIG_DIR, f"eval_{full_name}.py")
    code = dedent(f"""
    # Auto-generated eval config for {full_name}
    config = {{
        'kind': '{kind}',
        'map_json': '{map_json}',
        'full_model_dir': '{os.path.join(TRAINED_DIR, full_name)}',
        'half_model_dir': '{os.path.join(TRAINED_DIR, half_name)}',
        'base_model_dir': '{os.path.join(TRAINED_DIR, base_name)}',
        'full_test_json': '{full_test}',
        'half_test_json': '{half_test}',
        'base_test_json': '{base_test}',
        'out_dir': '{out_dir}',
    }}
    """).strip() + "\n"
    Path(cfg_path).write_text(code, encoding="utf-8")
    return cfg_path


def all_full_models():
    """
    Generator over the grid of FULL model names used in your experiments.
    Adjust lists if you add/remove counts or corpora.
    """
    bases = ["gpt_recipe", "gpt_tiny"]
    kinds = ["HOM", "HYP"]
    counts = [1, 10, 20, 30, 40, 50]
    for b in bases:
        for c in counts:
            for k in kinds:
                yield f"{b}_{c}_{k}"


def main():
    """
    CLI entry point:
      - With one positional argument → generate config for that FULL model.
      - With --all → generate configs for the whole grid.
    """
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        paths = [write_config(m) for m in all_full_models()]
        print("\nCreated configs:")
        for p in paths:
            print("  ", p)
    elif len(sys.argv) == 2:
        cfg = write_config(sys.argv[1])
        print("Created config:", cfg)
    else:
        print("Usage:")
        print("  python make_eval_configs.py gpt_recipe_1_HOM")
        print("  python make_eval_configs.py --all")


if __name__ == "__main__":
    main()