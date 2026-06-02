# Auto-generated training config for gpt_recipe_10_HOM (Run 1)

config = {
    # Output and checkpointing
    'out_dir': '/home/nina/ambiguity/trained_models/allkinds/gpt_recipe_10_HOM',
    'checkpoint_name': 'ckpt1.pt',
    'always_save_checkpoint': False,

    # Logging and evaluation
    'eval_interval': 250,
    'eval_iters': 200,
    'log_interval': 10,
    'wandb_log': True,
    'wandb_project': 'ambiguity_allkinds',
    'wandb_run_name': 'gpt_recipe_10_HOM',
    'wandb_tags': ['recipe'],

    # Dataset and batching
    'dataset': 'allkinds/data/gpt_recipe_10_HOM',
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
    'seed': 1337
}
