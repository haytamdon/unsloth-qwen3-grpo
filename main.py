import torch
import os
import numpy as np
from utils.utils import load_yaml, regex_format_search
from utils.prompts import define_system_prompt, define_chat_template
from models.models import load_model
from data.process_data import data_processing
from training.sft_trainer import define_sft_trainer
from training.grpo_trainer import define_grpo_trainer, get_grpo_args
from dotenv import load_dotenv
from datasets import load_dataset
import gc
import re
load_dotenv()

model_config_path = os.environ["MODEL_CONFIG_PATH"]
format_config_path = os.environ["FORMAT_CONFIG_PATH"]
training_config_path = os.environ["TRAINING_CONFIG_PATH"]
prompt_configs = load_yaml(path= format_config_path)
reasoning_start = prompt_configs["reasoning_start"]
reasoning_end = prompt_configs["reasoning_end"]
solution_start = prompt_configs["solution_start"]
solution_end = prompt_configs["solution_end"]
system_prompt = define_system_prompt(**prompt_configs)

def main():
    # Load configs
    model_configs = load_yaml(path= model_config_path)

    #Load model and tokenizer
    model, tokenizer = load_model(model_configs= model_configs)
    # Define system prompt
    chat_template = define_chat_template(system_prompt= system_prompt,
                                         reasoning_start= prompt_configs["reasoning_start"])
    tokenizer.chat_template = chat_template

    # Load data and process the data
    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split= "cot")
    dataset = data_processing(dataset, tokenizer, model_configs["max_seq_length"])

    # Pre-Fine Tune for formatting
    trainer = define_sft_trainer(model, tokenizer, dataset, training_config_path)
    trainer.train()

    # Clean Cache
    del dataset
    torch.cuda.empty_cache()
    gc.collect()

    # Load new data
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": x["solution"],
    })

    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"
    global match_format
    match_format = regex_format_search(solution_end_regex, reasoning_end, solution_start)
    tokenized = dataset.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    )
    tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

    import numpy as np
    maximum_length = int(np.quantile(tokenized["L"], 0.9))

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized
    training_args = get_grpo_args(maximum_length, model_configs["max_seq_length"], tokenizer)
    trainer = define_grpo_trainer(model, tokenizer, training_args, dataset)
    trainer.train()