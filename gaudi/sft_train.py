from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_seq_length=512,
    output_dir="/tmp",
)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
# 4. Print relevant info
print("Is GPU available? ", torch.cuda.is_available())
print("Trainer device:", trainer.args.device)
print("Model device:", trainer.model.device)

trainer.train()