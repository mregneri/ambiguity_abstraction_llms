#!/usr/bin/env python3
"""
launch_new_comp_eval_all_models.py
=================================

Launcher for evaluate_new_comp_accuracy.py.

Runs evaluation for:
- corpora: recipe, tiny
- kinds: HOM, HYP
- variants: FULL and HALF
- indices: 1, 10, 20, 30, 40, 50

It calls:
    python evaluate_new_comp_accuracy.py <MODEL_NAME> <MAP_JSON> <MASTER_INFLECTIONS_JSON>

and creates a shared launcher log under:
    /home/nina/ambiguity/evaluation/new_and_comp_accuracy/allkinds/_launcher_logs/

Behavior:
- FAILS immediately if a map JSON, master inflections JSON, test.bin, or checkpoint dir is missing
- Uses one shared launcher log file
- Prints all output live to the console and writes it to the log

Map conventions:
- HOM maps:
    .../create_homonym_corpora/allkinds/homonym_maps/<corpus>_<count>_HOM_map.json
- HYP maps:
    .../create_hypernym_corpora/allkinds/hypernym_maps/<corpus>_<count>_HYP_map.json

Master inflections conventions:
- .../experiments/inflection_maps/allkinds/<corpus>_master_<KIND>_inflections_edited.json
  e.g., recipe_master_HOM_inflections_edited.json
"""

import os
import sys
import subprocess
from datetime import datetime

# ============================================================
# Paths
# ============================================================
BASE_DIR = "/home/nina/ambiguity"

EVAL_SCRIPT = os.path.join(BASE_DIR, "evaluation", "new_and_comp_accuracy", "evaluate_new_comp_accuracy.py")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models", "allkinds")
DATA_DIR = os.path.join(BASE_DIR, "gpt_pipelines", "allkinds", "data")

HOM_MAPS_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "create_homonym_corpora",
    "allkinds",
    "homonym_maps",
)
HYP_MAPS_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "create_hypernym_corpora",
    "allkinds",
    "hypernym_maps",
)

INFLECTION_MAPS_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "inflection_maps",
    "allkinds",
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
    "new_and_comp_accuracy",
    "allkinds",
    "_launcher_logs",
)
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LAUNCHER_LOG = os.path.join(LOG_DIR, f"launcher_new_comp_eval_{TIMESTAMP}.log")

# ============================================================
# What to run
# ============================================================
CORPORA = ["recipe", "tiny"]
KINDS = ["HOM", "HYP"]
VARIANTS = ["", "_HALF"]     # "" = full
INDICES = [1, 10, 20, 30, 40, 50]

# ============================================================
# Helpers
# ============================================================
def model_name(corpus: str, idx: int, kind: str, variant: str) -> str:
    # e.g., gpt_recipe_1_HOM or gpt_tiny_50_HYP_HALF
    return f"gpt_{corpus}_{idx}_{kind}{variant}"


def map_path(corpus: str, idx: int, kind: str) -> str:
    # IMPORTANT: map filename includes kind, but NOT _HALF
    filename = f"{corpus}_{idx}_{kind}_map.json"
    if kind == "HOM":
        return os.path.join(HOM_MAPS_DIR, filename)
    elif kind == "HYP":
        return os.path.join(HYP_MAPS_DIR, filename)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def master_inflections_path(corpus: str, kind: str) -> str:
    # Expected pattern:
    #   recipe_master_HOM_inflections_edited.json
    filename = f"{corpus}_master_{kind}_inflections_edited.json"
    return os.path.join(INFLECTION_MAPS_DIR, filename)


def testbin_path(model: str) -> str:
    return os.path.join(DATA_DIR, model, "test.bin")


def ckpt_dir(model: str) -> str:
    return os.path.join(TRAINED_MODELS_DIR, model)


def assert_exists(path: str, description: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


# ============================================================
# Main
# ============================================================
def main():
    assert_exists(EVAL_SCRIPT, "evaluation script")

    with open(LAUNCHER_LOG, "w", encoding="utf-8") as log:
        log.write(f"Launcher started at {datetime.now()}\n")
        log.write(f"Evaluation script: {EVAL_SCRIPT}\n\n")
        log.flush()

        run_idx = 0

        for corpus in CORPORA:
            for kind in KINDS:
                for variant in VARIANTS:
                    for idx in INDICES:
                        run_idx += 1
                        model = model_name(corpus, idx, kind, variant)
                        map_json = map_path(corpus, idx, kind)
                        master_json = master_inflections_path(corpus, kind)

                        header = (
                            "\n" + "=" * 80 + "\n"
                            f"RUN {run_idx}: {model}\n"
                            f"KIND: {kind}\n"
                            f"MAP:  {map_json}\n"
                            f"MASTER_INFLECTIONS: {master_json}\n"
                            + "=" * 80 + "\n"
                        )

                        print(header, end="")
                        log.write(header)
                        log.flush()

                        # ---- strict checks ----
                        assert_exists(map_json, "map JSON")
                        assert_exists(master_json, "master inflections JSON")
                        assert_exists(testbin_path(model), "test.bin")
                        assert_exists(ckpt_dir(model), "checkpoint directory")

                        cmd = [sys.executable, EVAL_SCRIPT, model, map_json, master_json]

                        cmd_line = f"COMMAND: {' '.join(cmd)}\n\n"
                        print(cmd_line, end="")
                        log.write(cmd_line)
                        log.flush()

                        # ---- run and tee output ----
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )

                        for line in proc.stdout:
                            print(line, end="")
                            log.write(line)
                            log.flush()

                        proc.wait()

                        if proc.returncode != 0:
                            raise RuntimeError(
                                f"Evaluation failed for {model} "
                                f"(exit code {proc.returncode})"
                            )

        log.write("\nLauncher finished successfully.\n")

    print("\nAll evaluations completed successfully.")
    print(f"Launcher log written to:\n{LAUNCHER_LOG}")


if __name__ == "__main__":
    main()