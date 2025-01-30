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
if hasattr(torch, 'hpu'):
    print("Is HPU available? ", torch.hpu.is_available())
    if torch.hpu.is_available():
        print("HPU device count:", torch.hpu.device_count())
else:
    print("HPU not supported in this PyTorch installation.")

trainer.train()