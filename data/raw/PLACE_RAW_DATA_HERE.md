# data/raw — Place raw corpora here

Before running the pipeline, place the following files in this directory:

## RecipeNLG

**File:** `RecipeNLG_dataset.csv`
**Source:** https://recipenlg.cs.put.poznan.pl/
**Required columns:** `title`, `directions`

## TinyStories (V2, GPT-4 version)

**Files:**
- `TinyStoriesV2-GPT4-train.txt`
- `TinyStoriesV2-GPT4-valid.txt`

**Source:** https://huggingface.co/datasets/roneneldan/TinyStories

Download:
```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Once these files are in place, run:
```bash
python 01_create_base_corpora.py
```
