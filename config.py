import json
import os
from easydict import EasyDict as edict

def get_config_regression(model_name: str, dataset_name: str, config_file: str = ""
) -> dict:
    if config_file == '':
        config_file = 'configs/config.json'
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(dataset_args)
    config['dataset_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], dataset_name)
    config = edict(config)
    return config
