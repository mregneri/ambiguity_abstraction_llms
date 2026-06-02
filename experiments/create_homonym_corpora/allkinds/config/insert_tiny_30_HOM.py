# config/insert_tiny_30_HOM.py
config = {
    'train_in': '/home/nina/ambiguity/datasets/base/tiny_base_train.json',
    'val_in': '/home/nina/ambiguity/datasets/base/tiny_base_val.json',
    'test_in': '/home/nina/ambiguity/datasets/base/tiny_base_test.json',
    'train_out': '/home/nina/ambiguity/datasets/allkinds/tiny_30_HOM_train.json',
    'val_out': '/home/nina/ambiguity/datasets/allkinds/tiny_30_HOM_val.json',
    'test_out': '/home/nina/ambiguity/datasets/allkinds/tiny_30_HOM_test.json',
    'replacements': '/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/homonym_maps/tiny_30_HOM_map.json',
    'master_inflections': '/home/nina/ambiguity/experiments/inflection_maps/allkinds/tiny_master_HOM_inflections_edited.json',
    'summary_log': '/home/nina/ambiguity/experiments/create_homonym_corpora/allkinds/logs/tiny_30_HOM_summary.txt'
}
