from datasets import load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer
import torch

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = GaudiSFTConfig(
    use_habana=True,
    max_seq_length=512,
    output_dir="/tmp",
)
trainer = GaudiSFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text"
)
# 4. Print relevant info
if hasattr(torch, 'hpu'):
    print("Is HPU available? ", torch.hpu.is_available())
    if torch.hpu.is_available():
        print("HPU device count:", torch.hpu.device_count())
else:
    print("HPU not supported in this PyTorch installation.")

trainer.train()