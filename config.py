"""
config.py — Central configuration for the ambiguity/abstraction LLM pipeline.
==============================================================================

Edit the paths in this file to match your local setup before running anything.
All other scripts import from here — no absolute paths appear elsewhere.
"""

import os

# ============================================================
# ROOT — set this to wherever you cloned / placed the project
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# RAW CORPORA  (files YOU must place here before running)
# ============================================================
RAW_DATA_DIR       = os.path.join(ROOT, "data", "raw")

# RecipeNLG CSV  →  download from https://recipenlg.cs.put.poznan.pl/
RECIPE_CSV         = os.path.join(RAW_DATA_DIR, "RecipeNLG_dataset.csv")

# TinyStories V2 (GPT-4 version) →  https://huggingface.co/datasets/roneneldan/TinyStories
TINYSTORIES_TRAIN  = os.path.join(RAW_DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
TINYSTORIES_VALID  = os.path.join(RAW_DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

# ============================================================
# PROCESSED DATA  (created by the pipeline)
# ============================================================
PROCESSED_DIR      = os.path.join(ROOT, "data", "processed")

# Base corpora — produced by step 1 (create_base_corpora.py)
RECIPE_TRAINVAL    = os.path.join(PROCESSED_DIR, "recipe_base_trainval.json")
RECIPE_TRAIN       = os.path.join(PROCESSED_DIR, "recipe_base_train.json")
RECIPE_VAL         = os.path.join(PROCESSED_DIR, "recipe_base_val.json")
RECIPE_TEST        = os.path.join(PROCESSED_DIR, "recipe_base_test.json")

TINY_TRAINVAL      = os.path.join(PROCESSED_DIR, "tiny_base_trainval.json")
TINY_TRAIN         = os.path.join(PROCESSED_DIR, "tiny_base_train.json")
TINY_VAL           = os.path.join(PROCESSED_DIR, "tiny_base_val.json")
TINY_TEST          = os.path.join(PROCESSED_DIR, "tiny_base_test.json")

# Modified corpora (per condition) — produced by step 3
# Pattern: processed/<corpus>_<N>_<KIND>_{train|val|test}.json
# e.g. processed/recipe_10_HOM_train.json

# ============================================================
# WORD PAIR / MAP FILES  (shipped with the repo)
# ============================================================
DATA_DIR           = os.path.join(ROOT, "data")

RECIPE_HOM_PAIRS   = os.path.join(DATA_DIR, "recipe_homonym_pairs_final_shuffled.csv")
TINY_HOM_PAIRS     = os.path.join(DATA_DIR, "tiny_homonym_pairs_final_shuffled.csv")
RECIPE_HYP_PAIRS   = os.path.join(DATA_DIR, "recipe_hypernym_pairs_final_shuffled.csv")
TINY_HYP_PAIRS     = os.path.join(DATA_DIR, "tiny_hypernym_pairs_final_shuffled.csv")

BEEKHUIZEN_WORDS   = os.path.join(DATA_DIR, "beekhuizen_all_words.csv")

# ============================================================
# INFLECTION MAPS  (shipped with the repo)
# ============================================================
INFLECTION_DIR     = os.path.join(ROOT, "inflection_maps")

RECIPE_HOM_INFLECTIONS = os.path.join(INFLECTION_DIR, "recipe_master_HOM_inflections_edited.json")
RECIPE_HYP_INFLECTIONS = os.path.join(INFLECTION_DIR, "recipe_master_HYP_inflections_edited.json")
TINY_HOM_INFLECTIONS   = os.path.join(INFLECTION_DIR, "tiny_master_HOM_inflections_edited.json")
TINY_HYP_INFLECTIONS   = os.path.join(INFLECTION_DIR, "tiny_master_HYP_inflections_edited.json")

def get_inflection_map(corpus: str, kind: str) -> str:
    """Return the path to the correct inflection map for a given corpus + kind."""
    return {
        ("recipe", "HOM"): RECIPE_HOM_INFLECTIONS,
        ("recipe", "HYP"): RECIPE_HYP_INFLECTIONS,
        ("tiny",   "HOM"): TINY_HOM_INFLECTIONS,
        ("tiny",   "HYP"): TINY_HYP_INFLECTIONS,
    }[(corpus, kind)]

# ============================================================
# HOMONYM / HYPERNYM MAPS  (shipped with the repo)
# ============================================================
HOM_MAPS_DIR       = os.path.join(ROOT, "homonym_hypernym_maps", "homonym_maps")
HYP_MAPS_DIR       = os.path.join(ROOT, "homonym_hypernym_maps", "hypernym_maps")

def get_pseudoword_map(corpus: str, n: int, kind: str) -> str:
    """
    Return the path to the pseudoword replacement map for a given setting.

    Parameters
    ----------
    corpus : "recipe" or "tiny"
    n      : number of pairs (1, 10, 20, 30, 40, 50)
    kind   : "HOM" or "HYP" (without _HALF suffix — maps are shared)
    """
    base_kind = kind.replace("_HALF", "")   # HOM_HALF uses the same HOM map
    maps_dir = HOM_MAPS_DIR if "HOM" in kind else HYP_MAPS_DIR
    return os.path.join(maps_dir, f"{corpus}_{n}_{base_kind}_map.json")

# ============================================================
# TRAINED MODELS  (created by the pipeline; models are large,
#                  only training logs are tracked in git)
# ============================================================
TRAINED_MODELS_DIR = os.path.join(ROOT, "trained_models")

def get_word2vec_model_dir(corpus: str, n_or_base: str, kind: str = None) -> str:
    """Return the directory for a word2vec model."""
    if n_or_base == "base":
        name = f"word2vec_{corpus}_base"
    else:
        name = f"word2vec_{corpus}_{n_or_base}_{kind}"
    return os.path.join(TRAINED_MODELS_DIR, name)

def get_gpt_model_dir(corpus: str, n_or_base: str, kind: str = None) -> str:
    """Return the directory for a GPT model (checkpoints saved here)."""
    if n_or_base == "base":
        name = f"gpt_{corpus}_base"
    else:
        name = f"gpt_{corpus}_{n_or_base}_{kind}"
    return os.path.join(TRAINED_MODELS_DIR, name)

def get_gpt_data_dir(corpus: str, n_or_base: str, kind: str = None) -> str:
    """Return the data directory (train.bin / val.bin / test.bin) for a GPT model."""
    if n_or_base == "base":
        name = f"gpt_{corpus}_base"
    else:
        name = f"gpt_{corpus}_{n_or_base}_{kind}"
    return os.path.join(PROCESSED_DIR, name)

# ============================================================
# EVALUATION OUTPUT DIRS  (created by the pipeline)
# ============================================================
EVAL_DIR           = os.path.join(ROOT, "evaluation")
ACCURACY_DIR       = os.path.join(EVAL_DIR, "accuracy")
NBHD_DIR           = os.path.join(EVAL_DIR, "nbhd")
NEW_COMP_EVAL_DIR  = os.path.join(EVAL_DIR, "new_comp_accuracy")

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================
CORPORA            = ["recipe", "tiny"]
COUNTS             = [1, 10, 20, 30, 40, 50]
KINDS              = ["HOM", "HYP", "HOM_HALF", "HYP_HALF"]

# ============================================================
# GPT MODEL HYPERPARAMETERS  (shared across all conditions)
# ============================================================
GPT_DEFAULTS = dict(
    eval_interval             = 250,
    eval_iters                = 200,
    log_interval              = 10,
    batch_size                = 64,
    block_size                = 256,
    gradient_accumulation_steps = 1,
    n_layer                   = 6,
    n_head                    = 6,
    n_embd                    = 384,
    dropout                   = 0.2,
    bias                      = False,
    learning_rate             = 1e-3,
    max_iters                 = 5000,
    lr_decay_iters            = 5000,
    min_lr                    = 1e-4,
    beta1                     = 0.9,
    beta2                     = 0.99,
    weight_decay              = 1e-1,
    warmup_iters              = 100,
    decay_lr                  = True,
    device                    = "cuda",
    dtype                     = "float16",
    compile                   = False,
    eval_only                 = False,
    grad_clip                 = 1.0,
    init_from                 = "scratch",
    wandb_log                 = True,
    wandb_project             = "ambiguity_abstraction_llms",
    always_save_checkpoint    = False,
)

# Evaluation settings
EVAL_BLOCK_SIZE = 256
EVAL_BATCH_SIZE = 64
CHECKPOINT_NAMES = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
N_RUNS = 3          # seeds: 1337, 1338, 1339
