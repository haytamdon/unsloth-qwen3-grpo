import torch
import os
from utils.utils import load_yaml, load_dataset
from utils.prompts import define_system_prompt, define_chat_template
from models.models import load_model
from data.process_data import data_processing
from dotenv import load_dotenv
load_dotenv()

model_config_path = os.environ["MODEL_CONFIG_PATH"]
format_config_path = os.environ["FORMAT_CONFIG_PATH"]
training_config_path = os.environ["TRAINING_CONFIG_PATH"]
prompt_configs = load_yaml(path= format_config_path)
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

    # Load data
    omr_data = load_dataset(data_path= "unsloth/OpenMathReasoning-mini",
                            split= "cot")
    
    dataset = data_processing(omr_data, tokenizer, model_configs["max_seq_length"])
