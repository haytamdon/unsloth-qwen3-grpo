from trl import SFTTrainer, SFTConfig
from utils.utils import load_yaml

def define_sft_trainer(model, tokenizer, dataset, training_config_path):
    sft_args = load_yaml(training_config_path)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(**sft_args),
    )
    return trainer