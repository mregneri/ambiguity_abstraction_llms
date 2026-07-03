"""
run_pipeline.py
===============

Master script that runs the full pipeline end-to-end in the correct order.
Each step is a separate subprocess call so failures are visible immediately.

Stages
------
  1  create_base_corpora    — process raw corpora → base JSON files
  2  train_word2vec_base    — train Word2Vec on base corpora
  3  prepare_corpora        — insert pseudowords for all conditions
  4  tokenize_for_gpt       — convert corpora to .bin token files
  5  train_word2vec_cond    — train Word2Vec for all conditions
  6  word2vec_eval          — compute neighbourhood metrics
  7  train_gpt              — train GPT models (3 runs × all conditions)
  8  evaluate_gpt_overall   — next-token accuracy on test set
  9  evaluate_gpt_breakdown — accuracy split by pseudoword/component presence

Usage:
    # Full pipeline
    python run_pipeline.py

    # Single stage
    python run_pipeline.py --start 3 --end 3

    # From a given stage to the end
    python run_pipeline.py --start 5

Notes:
    - Stage 7 (GPT training) is GPU-intensive and will take hours/days.
    - Stages 3–9 can be run for a single condition:
        python run_pipeline.py --start 3 --end 4 --corpus recipe --n 10 --kind HOM
"""

import subprocess
import sys
import argparse

STAGES = {
    1: ("01_create_base_corpora.py", []),
    2: ("02_train_word2vec.py",      []),             # base models
    3: ("03_prepare_corpora.py",     ["--all"]),
    4: ("04_tokenize_for_gpt.py",    ["--all"]),
    5: ("02_train_word2vec.py",      ["--all"]),      # condition models
    6: ("06_word2vec_eval.py",       ["--all"]),
    7: ("05_train_gpt.py",           ["--all"]),
    8:  ("07_evaluate_gpt.py",           ["--all"]),
    9:  ("08_analyze_embeddings.py",     ["--all"]),
    10: ("09_analyze_weights.py",        ["--all"]),
    11: ("10_analyze_disambiguation.py", ["--all_corpora", "--compare"]),
}

STAGE_NAMES = {
    1:  "Create base corpora",
    2:  "Train Word2Vec (base)",
    3:  "Prepare modified corpora",
    4:  "Tokenize for GPT",
    5:  "Train Word2Vec (conditions)",
    6:  "Word2Vec neighbourhood eval",
    7:  "Train GPT models",
    8:  "Evaluate GPT (accuracy + perplexity, all batch categories)",
    9:  "Analyze embedding geometry (effective rank, compressibility, cosine)",
    10: "Analyze MLP W1 spectral properties",
    11: "Sense selection experiment (HALF models only, HOM vs HYP comparison)",
}

def run(script: str, extra_args: list) -> int:
    cmd = [sys.executable, script] + extra_args
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end",   type=int, default=11)
    # Pass-through args for single-condition runs on stages 3–9
    parser.add_argument("--corpus", help="Limit to one corpus (recipe or tiny)")
    parser.add_argument("--n",      type=int, help="Limit to one N value")
    parser.add_argument("--kind",   help="Limit to one kind (HOM, HYP, HOM_HALF, HYP_HALF)")
    args = parser.parse_args()

    extra = []
    if args.corpus: extra += ["--corpus", args.corpus]
    if args.n:      extra += ["--n", str(args.n)]
    if args.kind:   extra += ["--kind", args.kind]

    for stage_id in range(args.start, args.end + 1):
        script, default_args = STAGES[stage_id]
        name = STAGE_NAMES[stage_id]

        # For stages 3–9, allow per-condition override (remove --all if extra args given)
        stage_args = default_args
        if extra and stage_id >= 3:
            stage_args = [a for a in default_args if a != "--all"] + extra

        print(f"\n{'='*60}")
        print(f"  Stage {stage_id}: {name}")
        print(f"{'='*60}")
        code = run(script, stage_args)
        if code != 0:
            print(f"\n❌ Stage {stage_id} failed with code {code}. Stopping.")
            sys.exit(code)
        print(f"\n✅ Stage {stage_id} complete.")

    print("\n🎉 Pipeline finished successfully.")

if __name__ == "__main__":
    main()
