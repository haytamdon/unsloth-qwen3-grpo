import yaml
from typing import Dict
from datasets import load_dataset
import re

def load_yaml(path: str) -> Dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def regex_format_search(solution_end_regex, reasoning_end, solution_start):
    # Add optional EOS token matching
    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    return match_format

