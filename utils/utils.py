import yaml
from typing import Dict

def load_yaml(path: str) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

