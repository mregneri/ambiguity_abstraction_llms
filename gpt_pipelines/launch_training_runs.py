"""
launch_training_runs.py
========================

This script automates the creation and execution of three GPT training runs for a given model name.
It generates three config files with unique checkpoint names and seeds,
but identical wandb run names (to group runs together in Weights & Biases).

Usage:
    python launch_training_runs.py <model_name>

Example:
    python launch_training_runs.py gpt_tiny_1_HOM

Expected dataset folder:
    /home/nina/ambiguity/gpt_pipelines/allkinds/data/<model_name>/
        ├── train.bin
        ├── val.bin
        └── test.bin
"""

import os
import sys
import subprocess

TEMPLATE = """# Auto-generated training config for {model_name} (Run {run_id})

config = {{
    # Output and checkpointing
    'out_dir': '{out_dir}',
    'checkpoint_name': 'ckpt{run_id}.pt',
    'always_save_checkpoint': False,

    # Logging and evaluation
    'eval_interval': 250,
    'eval_iters': 200,
    'log_interval': 10,
    'wandb_log': True,
    'wandb_project': 'ambiguity_allkinds',
    'wandb_run_name': '{model_name}',
    'wandb_tags': {wandb_tags},

    # Dataset and batching
    'dataset': '{data_path}',
    'batch_size': 64,
    'block_size': 256,
    'gradient_accumulation_steps': 1,

    # Model architecture
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.2,
    'bias': False,

    # Training schedule
    'learning_rate': 1e-3,
    'max_iters': 5000,
    'lr_decay_iters': 5000,
    'min_lr': 1e-4,
    'beta1': 0.9,
    'beta2': 0.99,
    'weight_decay': 1e-1,
    'warmup_iters': 100,
    'decay_lr': True,

    # Runtime and optimization
    'device': 'cuda',
    'dtype': 'float16',
    'compile': False,
    'eval_only': False,
    'grad_clip': 1.0,

    # Model initialization
    'init_from': 'scratch',

    # Random seed for variation
    'seed': {seed}
}}
"""

def main():
    if len(sys.argv) != 2:
        print("Usage: python launch_training_runs.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    base_dir = "/home/nina/ambiguity"

    # --- Script paths ---
    train_script = os.path.join(base_dir, "gpt_pipelines", "train.py")
    config_dir = os.path.join(base_dir, "gpt_pipelines", "allkinds", "config")

    # dataset path relative to gpt_pipelines/
    data_path = os.path.join("allkinds", "data", model_name)
    full_data_dir = os.path.join(base_dir, "gpt_pipelines", data_path)

    # *** Output directory ***
    out_dir = os.path.join(base_dir, "trained_models", "allkinds", model_name)

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # --- Wandb tags ---
    tags = []
    if "recipe" in model_name.lower():
        tags.append("recipe")
    elif "tiny" in model_name.lower():
        tags.append("tiny")
    if "half" in model_name.lower():
        tags.append("half")

    wandb_tags_str = str(tags)

    print(f"\n🚀 Launching training runs for: {model_name}")
    print(f"   Dataset folder: {full_data_dir}")
    print(f"   Output folder:  {out_dir}")
    print(f"   W&B tags:       {wandb_tags_str}\n")

    # --- Generate configs + run training ---
    for run_id, seed in zip([1, 2, 3], [1337, 1338, 1339]):
        config_filename = f"train_{model_name}_run{run_id}.py"
        config_path = os.path.join(config_dir, config_filename)

        config_code = TEMPLATE.format(
            model_name=model_name,
            run_id=run_id,
            seed=seed,
            out_dir=out_dir,
            data_path=data_path,
            wandb_tags=wandb_tags_str
        )

        with open(config_path, "w") as f:
            f.write(config_code)

        print(f"Starting run {run_id} (seed={seed}) with config: {config_path}")
        subprocess.run(["python", train_script, config_path])

    print("\n🎉 All training runs completed successfully!")

if __name__ == "__main__":
    main()