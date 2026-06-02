# trained_models/

This directory is populated by the pipeline. **Model files are not tracked in git** due to size.

## Expected contents after running the pipeline

### Word2Vec models (step 2 + step 5)

One directory per model:
```
word2vec_{corpus}_{N}_{KIND}/
    word2vec.model        ← Gensim format
    word2vec.bin          ← Binary Word2Vec format  
    training_log.txt      ← Training summary
```

Example names:
- `word2vec_recipe_base/`
- `word2vec_recipe_10_HOM/`
- `word2vec_tiny_50_HYP_HALF/`

### GPT models (step 7)

One directory per experimental condition:
```
gpt_{corpus}_{N}_{KIND}/
    ckpt1.pt      ← Run 1 (seed 1337)
    ckpt2.pt      ← Run 2 (seed 1338)
    ckpt3.pt      ← Run 3 (seed 1339)
```

Example names:
- `gpt_recipe_base/`
- `gpt_recipe_10_HOM/`
- `gpt_tiny_50_HYP_HALF/`

## Size estimate

Each GPT checkpoint is approximately **5–20 MB** depending on architecture.  
Total trained model storage: ~50 GB for all conditions × 3 seeds.

## Restoring from checkpoint

To resume training from a checkpoint, set `init_from` in `config.py` to the checkpoint path.
