import pandas as pd
import numpy as np
import os
from utils.utils import load_yaml
from main import prompt_configs, system_prompt
from datasets import Dataset

reasoning_start = prompt_configs["reasoning_start"]
reasoning_end = prompt_configs["reasoning_end"]
solution_start = prompt_configs["solution_start"]
solution_end = prompt_configs["solution_end"]


def keep_numbers_only(dataset: Dataset) -> pd.DataFrame:
    pd_dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # Try converting to number - if not, replace with NaN
    is_number = pd.to_numeric(pd.Series(pd_dataset["expected_answer"]), errors = "coerce").notnull()
    # Select only numbers
    pd_dataset_num = pd_dataset.iloc[np.where(is_number)[0]]
    return pd_dataset_num

def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]

def truncate_data(pd_dataset, tokenizer, max_seq_length):
    pd_dataset["N"] = pd_dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    pd_dataset_truncated = pd_dataset.loc[pd_dataset["N"] <= max_seq_length/2].copy()
    return pd_dataset_truncated

def data_processing(dataset, tokenizer, max_seq_length):
    pd_dataset_num = keep_numbers_only(dataset)
    pd_dataset_num["Messages"] = pd_dataset_num.apply(format_dataset, axis = 1)
    pd_dataset_truncated = truncate_data(pd_dataset_num, tokenizer, max_seq_length)
    pd_dataset_truncated["text"] = tokenizer.apply_chat_template(pd_dataset_truncated["Messages"].values.tolist(), tokenize = False)
    dataset = Dataset.from_pandas(pd_dataset_truncated)
    return dataset