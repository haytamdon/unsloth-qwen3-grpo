from unsloth import FastLanguageModel # type: ignore
from typing import Dict

def load_model(model_configs: Dict):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_configs["model_path"],
        max_seq_length = model_configs["max_seq_length"],
        load_in_4bit = False, 
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = model_configs["lora_rank"],
        gpu_memory_utilization = model_configs["memory_usage"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = model_configs["lora_rank"],
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = model_configs["lora_rank"]*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    return model, tokenizer