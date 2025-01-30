import os
import torch
from datasets import load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")

# 2. Configure training with GaudiSFTConfig
training_args = GaudiSFTConfig(
    use_habana=True,
    max_seq_length=512, #max_seq_length (int or None, optional, defaults to None) — Maximum sequence length for the ConstantLengthDataset and for automatically creating the dataset. If None, it uses the smaller value between tokenizer.model_max_length and 1024.
    output_dir="/tmp",
)
gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = True
gaudi_config.use_fused_clip_norm = True

# 3. Initialize the GaudiSFTTrainer
trainer = GaudiSFTTrainer(
    model="facebook/opt-350m",
    gaudi_config=gaudi_config,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text" #dataset_text_field (str, optional, defaults to "text") — Name of the text field of the dataset. If provided, the trainer will automatically create a ConstantLengthDataset based on dataset_text_field.
)

# 5. Start training
trainer.train()
