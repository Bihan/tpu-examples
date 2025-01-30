import os
import torch
from datasets import load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer

# Fetch Hugging Face Hub Token from the environment variable
HF_TOKEN = os.getenv("HF_TOKEN", None)

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it before running this script.")

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")

# 2. Configure training with GaudiSFTConfig
training_args = GaudiSFTConfig(
    use_habana=True,
    max_seq_length=512,
    output_dir="/tmp",
)

# 3. Initialize the GaudiSFTTrainer
trainer = GaudiSFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text"
)

# 4. Print HPU availability information
if hasattr(torch, 'hpu'):
    print("Is HPU available? ", torch.hpu.is_available())
    if torch.hpu.is_available():
        print("HPU device count:", torch.hpu.device_count())
else:
    print("HPU not supported in this PyTorch installation.")

# 5. Start training
trainer.train()
