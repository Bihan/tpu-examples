import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer

# 1. Load dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")

# 2. Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)

# 3. Configure training with GaudiSFTConfig
training_args = GaudiSFTConfig(
    use_habana=True,
    max_seq_length=512,
    output_dir="/tmp",
)

gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = True
gaudi_config.use_fused_clip_norm = True

# 4. Initialize the GaudiSFTTrainer
trainer = GaudiSFTTrainer(
    model="deepseek-ai/deepseek-llm-7b-base",
    gaudi_config=gaudi_config,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['input_ids'] for f in data])},
)

# 5. Start training
trainer.train()
