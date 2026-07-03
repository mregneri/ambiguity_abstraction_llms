# Ambiguity and Abstraction: Analyzing Factors of Linguistic Complexity for Language Model Optimization

Code for the paper:

> **Compact languages, complex model shifts: How and Where Ambiguity
and Underspecification Affect LLMs**  
> Anonymous Submission

The pipeline trains small GPT-2-style models on two corpora (RecipeNLG and TinyStories), evaluating how linguistic ambiguity (homonymy and hypernymy) affects language model representations and next-token predictions. Artificial pseudowords with controlled ambiguity properties are injected into the training corpora at varying densities.

---

## Repository Structure

```
.
├── config.py                     ← Central config (all paths & hyperparameters)
├── run_pipeline.py               ← Master pipeline runner
│
├── 01_create_base_corpora.py     ← Process raw corpora → base JSON splits
├── 02_train_word2vec.py          ← Train Word2Vec models
├── 03_prepare_corpora.py         ← Insert pseudowords into corpora
├── 04_tokenize_for_gpt.py        ← Convert JSON → binary token files
├── 05_train_gpt.py               ← Train GPT models (3 seeds × all conditions)
├── 06_word2vec_eval.py           ← Compute neighbourhood distance metrics
├── 07_evaluate_gpt.py            ← GPT accuracy (overall + breakdown)
│
├── model.py                      ← nanoGPT model definition (GPTConfig, GPT)
│
├── data/
│   ├── raw/                      ← ⚠️ Place raw corpora here (see below)
│   ├── processed/                ← Created by the pipeline
│   ├── beekhuizen_all_words.csv  ← Word list (homonyms, polysemes, monosemes)
│   ├── recipe_homonym_pairs_final_shuffled.csv
│   ├── recipe_hypernym_pairs_final_shuffled.csv
│   ├── tiny_homonym_pairs_final_shuffled.csv
│   └── tiny_hypernym_pairs_final_shuffled.csv
│
├── inflection_maps/              ← Pre-verified inflection maps (shipped)
│   ├── recipe_master_HOM_inflections_edited.json
│   ├── recipe_master_HYP_inflections_edited.json
│   ├── tiny_master_HOM_inflections_edited.json
│   └── tiny_master_HYP_inflections_edited.json
│
├── homonym_hypernym_maps/        ← Pseudoword replacement maps (shipped)
│   ├── homonym_maps/             ← recipe_N_HOM_map.json, tiny_N_HOM_map.json
│   └── hypernym_maps/            ← recipe_N_HYP_map.json, tiny_N_HYP_map.json
│
├── trained_models/               ← Created by the pipeline (large — not in git)
│   ├── word2vec_recipe_base/     ← word2vec.model, word2vec.bin, training_log.txt
│   ├── word2vec_recipe_10_HOM/   ← (same structure for each condition)
│   ├── gpt_recipe_base/          ← ckpt1.pt, ckpt2.pt, ckpt3.pt
│   └── gpt_recipe_10_HOM/        ← (same structure for each condition)
│
└── evaluation/                   ← Created by the pipeline
    ├── accuracy/                 ← accuracy_results.txt per model
    ├── nbhd/                     ← neighbourhood distance CSVs
    └── new_comp_accuracy/        ← pseudoword/component breakdown results
```

---

## Prerequisites

### Python environment

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `gensim`, `tiktoken`, `pandas`, `numpy`, `scikit-learn`, `tqdm`, `wandb`

### Raw corpora (you must provide these)

Place the following files in `data/raw/` **before running step 1**:

| File | Source |
|------|--------|
| `RecipeNLG_dataset.csv` | [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) |
| `TinyStoriesV2-GPT4-train.txt` | [HuggingFace — TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| `TinyStoriesV2-GPT4-valid.txt` | same |

### GPU

Steps 5 (GPT training) and 7 (GPT evaluation) require a CUDA-capable GPU. All other steps run on CPU.

### Weights & Biases (optional)

Training and evaluation log to W&B by default. Set `wandb_log = False` in `config.py` to disable, or run `wandb offline` first.

---

## Running the Pipeline

### Full pipeline

```bash
python run_pipeline.py
```

This runs all 9 stages in order. GPU stages (training, evaluation) will take hours to days depending on hardware.

### Stage by stage

```bash
python run_pipeline.py --start 1 --end 1   # base corpora only
python run_pipeline.py --start 2 --end 2   # Word2Vec base models
python run_pipeline.py --start 3 --end 4   # corpus prep + GPT tokenization
python run_pipeline.py --start 5 --end 6   # Word2Vec conditions + eval
python run_pipeline.py --start 7 --end 7   # GPT training (slow)
python run_pipeline.py --start 8 --end 9   # GPT evaluation
```

### Single condition

```bash
python run_pipeline.py --start 3 --end 7 --corpus recipe --n 10 --kind HOM
```

---

## Execution Order (detailed)

| Stage | Script | What it does | Inputs | Outputs |
|-------|--------|--------------|--------|---------|
| 1 | `01_create_base_corpora.py` | Clean & split raw corpora | `data/raw/*.csv / *.txt` | `data/processed/recipe_base_*.json`, `data/processed/tiny_base_*.json` |
| 2 | `02_train_word2vec.py` | Train Word2Vec on base corpora | base JSON trainval files | `trained_models/word2vec_{corpus}_base/` |
| 3 | `03_prepare_corpora.py` | Insert pseudowords | base JSON splits + maps | `data/processed/{corpus}_{N}_{KIND}_*.json` |
| 4 | `04_tokenize_for_gpt.py` | GPT-2 BPE tokenisation | modified JSON splits | `data/processed/gpt_{corpus}_{N}_{KIND}/*.bin` |
| 5 | `02_train_word2vec.py --all` | Train Word2Vec on all conditions | modified JSON trainval files | `trained_models/word2vec_{corpus}_{N}_{KIND}/` |
| 6 | `06_word2vec_eval.py` | Neighbourhood distance metrics | Word2Vec `.bin` models | `evaluation/nbhd/{model}_nbhd_results/` |
| 7 | `05_train_gpt.py` | Train GPT (3 seeds) | `.bin` token files | `trained_models/gpt_{corpus}_{N}_{KIND}/ckpt{1,2,3}.pt` |
| 8 | `07_evaluate_gpt.py` | Accuracy + perplexity, all batch categories | checkpoints + test.bin + maps | `evaluation/new_comp_accuracy/{model}/eval_results.txt` |
| 9 | `08_analyze_embeddings.py` | Embedding geometry (effective rank, compressibility, pseudoword positioning) | checkpoints | `evaluation/embeddings/{model}/embedding_analysis.txt`, `evaluation/embeddings/embedding_summary.csv` |
| 10 | `09_analyze_weights.py` | MLP W1 spectral analysis (effective rank, explained variance Δ vs base) | checkpoints | `evaluation/weights/{model}/weight_analysis.txt`, `evaluation/weights/weight_summary.csv` |
| 11 | `10_analyze_disambiguation.py` | Sense selection: differential, sense_bias, base_dist per layer (HALF models only) | checkpoints | `evaluation/disambig/{model}_disambig.txt`, `evaluation/disambig/{model}_disambig.json` |

### Evaluation output columns (stage 8)

For every model and checkpoint, the evaluation reports these metrics **per batch category**:

| Metric | Description |
|--------|-------------|
| `accuracy` | Next-token prediction accuracy |
| `ppl_mod` | Perplexity of the modified (experimental) model |
| `ppl_base` | Perplexity of the base model on the same tokens — **valid only for `overall` and `neither`** (other categories contain pseudoword token IDs the base model was never trained on) |

Batch categories:

| Category | Batches included |
|----------|-----------------|
| `overall` | All batches |
| `contains_new` | Context X contains a pseudoword |
| `no_new` | Context X contains no pseudoword |
| `contains_component` | Context X contains an inflected component word |
| `no_component` | Context X contains no component word |
| `either` | Context X contains a pseudoword OR a component word |
| `neither` | Context X contains neither (general text) |

---

## Experimental Conditions

### Corpora

| Name | Description |
|------|-------------|
| `recipe` | RecipeNLG — cooking instructions |
| `tiny`   | TinyStories — short children's stories |

### Conditions (KIND)

| Kind | Description |
|------|-------------|
| `HOM`      | Artificial homonyms: two semantically **dissimilar** words merged into one pseudoword |
| `HYP`      | Artificial hypernyms: two semantically **similar** words merged into one pseudoword |
| `HOM_HALF` | As HOM, but only one component word's occurrences are replaced |
| `HYP_HALF` | As HYP, but only one component word's occurrences are replaced |

### N (number of pseudoword pairs)

`N ∈ {1, 10, 20, 30, 40, 50}`

Each (corpus × N × KIND) combination is a separate condition.  
Each GPT model is trained 3 times (seeds 1337, 1338, 1339), producing checkpoints `ckpt1.pt`, `ckpt2.pt`, `ckpt3.pt`.

---

## Files Shipped with the Repository

These files encode the experimental design and are committed to the repo:

- **`data/beekhuizen_all_words.csv`** — target word list for neighbourhood analysis (homonyms, polysemes, monosemes from Beekhuizen et al.)
- **`data/recipe_*.csv` / `data/tiny_*.csv`** — final shuffled word pair lists for HOM and HYP conditions
- **`inflection_maps/*.json`** — manually verified inflection maps for all component words
- **`homonym_hypernym_maps/homonym_maps/*.json`** — pseudoword replacement maps for HOM conditions
- **`homonym_hypernym_maps/hypernym_maps/*.json`** — pseudoword replacement maps for HYP conditions

---

## Files Created by the Pipeline (not in git)

| Path pattern | Created by | Description |
|-------------|-----------|-------------|
| `data/processed/*.json` | steps 1, 3 | Tokenized corpus files |
| `data/processed/gpt_*/*.bin` | step 4 | Binary token files for GPT |
| `trained_models/word2vec_*/` | step 2, 5 | Word2Vec models + logs |
| `trained_models/gpt_*/ckpt*.pt` | step 7 | GPT checkpoints (large) |
| `evaluation/*/` | steps 6, 8, 9 | Evaluation results |

---

## Configuration

All paths and hyperparameters are centralised in `config.py`. Edit this file to:

- Change the root directory (`ROOT`)
- Point to different raw data locations
- Adjust GPT architecture or training hyperparameters
- Enable/disable W&B logging

---

## Citation

```bibtex
@article{regneri-scheller-2025-ambiguity,
  title   = {Ambiguity and Abstraction: Analyzing Factors of Linguistic Complexity for Language Model Optimization},
  author  = {Regneri, Michaela and Scheller, Nina},
  year    = {2025}
}
```
