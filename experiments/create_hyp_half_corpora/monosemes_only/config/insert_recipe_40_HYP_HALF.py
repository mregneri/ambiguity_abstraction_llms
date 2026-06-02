# config/insert_recipe_40_HYP_HALF.py
config = {
    'train_in': '/home/nina/ambiguity/datasets/recipe_base_train.json',
    'val_in': '/home/nina/ambiguity/datasets/recipe_base_val.json',
    'test_in': '/home/nina/ambiguity/datasets/recipe_base_test.json',

    'train_out': '/home/nina/ambiguity/datasets/recipe_40_HYP_HALF_train.json',
    'val_out': '/home/nina/ambiguity/datasets/recipe_40_HYP_HALF_val.json',
    'test_out': '/home/nina/ambiguity/datasets/recipe_40_HYP_HALF_test.json',

    'replacements': '/home/nina/ambiguity/experiments/create_hypernym_corpora/hypernym_maps/recipe_40_HYP_map.json',
    'master_inflections': '/home/nina/ambiguity/experiments/inflection_maps/recipe_master_HYP_inflections_edited.json',
    'summary_log': '/home/nina/ambiguity/experiments/create_hyp_half_corpora/logs/recipe_40_HYP_HALF_summary.txt'
}
