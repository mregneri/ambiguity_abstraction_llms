# config/insert_tiny_50_HYP.py
config = {
    'train_in': '/home/nina/ambiguity/datasets/tiny_base_train.json',
    'val_in': '/home/nina/ambiguity/datasets/tiny_base_val.json',
    'test_in': '/home/nina/ambiguity/datasets/tiny_base_test.json',
    'train_out': '/home/nina/ambiguity/datasets/tiny_50_HYP_train.json',
    'val_out': '/home/nina/ambiguity/datasets/tiny_50_HYP_val.json',
    'test_out': '/home/nina/ambiguity/datasets/tiny_50_HYP_test.json',
    'replacements': '/home/nina/ambiguity/experiments/create_hypernym_corpora/hypernym_maps/tiny_50_HYP_map.json',
    'master_inflections': '/home/nina/ambiguity/experiments/inflection_maps/tiny_master_HYP_inflections_edited.json',
    'summary_log': '/home/nina/ambiguity/experiments/create_hypernym_corpora/logs/tiny_50_HYP_summary.txt'
}
