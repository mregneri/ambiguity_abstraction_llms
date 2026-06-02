#!/usr/bin/env python3
"""
run_newword_evals.py
===================

Purpose
-------
Convenience launcher that:
  1) creates evaluation config file(s) via `make_eval_configs.py`
  2) runs the evaluator `measure_newword_accuracy.py` for each config

You can target a single FULL model (e.g., "gpt_recipe_1_HOM") or run the
entire predefined grid with `--all`.

Outputs (per FULL model)
------------------------
/home/nina/ambiguity/evaluation/newword_accuracy/results/<FULL_MODEL>/
    ├── newword_accuracy.csv
    └── newword_accuracy.json

Usage
-----
# Single FULL model
python run_newword_evals.py gpt_recipe_1_HOM

# All FULL models in the grid
python run_newword_evals.py --all

Notes
-----
• This script does not alter any evaluator defaults (e.g., SPLITTER, BATCH_SIZE).
• It assumes each model directory contains ckpt1.pt, ckpt2.pt, ckpt3.pt.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE = "/home/nina/ambiguity"
EVAL_DIR = f"{BASE}/evaluation/newword_accuracy"
CONFIG_DIR = f"{EVAL_DIR}/config"
MAKE_CFG = os.path.join(EVAL_DIR, "make_eval_configs.py")
EVALUATOR = os.path.join(EVAL_DIR, "measure_newword_accuracy.py")


def run_one(model_full_name: str):
    """
    Generate a config for a single FULL model and run the evaluator on it.
    """
    # 1) Create config
    subprocess.check_call(["python", MAKE_CFG, model_full_name])
    cfg_path = os.path.join(CONFIG_DIR, f"eval_{model_full_name}.py")

    # 2) Evaluate
    subprocess.check_call(["python", EVALUATOR, cfg_path])


def run_all():
    """
    Generate configs for the whole grid and evaluate each one sequentially.
    """
    # 1) Create all configs
    subprocess.check_call(["python", MAKE_CFG, "--all"])

    # 2) Evaluate each config in lexicographic order for reproducible logs
    for cfg in sorted(Path(CONFIG_DIR).glob("eval_*.py")):
        subprocess.check_call(["python", EVALUATOR, str(cfg)])


def main():
    """
    CLI entry point:
      - With one positional argument → run for that FULL model
      - With --all → run for all predefined FULL models
    """
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        run_all()
    elif len(sys.argv) == 2:
        run_one(sys.argv[1])
    else:
        print("Usage:")
        print("  python run_newword_evals.py gpt_recipe_1_HOM")
        print("  python run_newword_evals.py --all")


if __name__ == "__main__":
    main()