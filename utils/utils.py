import yaml
from typing import Dict
from datasets import load_dataset

def load_yaml(path: str) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_dataset(data_path: str,
                 split: str):
    dataset = load_dataset(data_path, split = split)
    return dataset