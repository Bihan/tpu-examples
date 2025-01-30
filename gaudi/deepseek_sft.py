import torch

# Hugging Face & Optimum Habana imports
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer

# -------------------------------------------------------------------------
# 1. Load the DeepSeek LLM and Tokenizer
# -------------------------------------------------------------------------
model_name = "deepseek-ai/deepseek-llm-7b-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# If your model uses a special pad token, set it. Otherwise, default to the EOS token for padding.
# For example:
# tokenizer.pad_token = tokenizer.eos_token

# Load the base model with bfloat16 precision (works well on Gaudi)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16  # or torch.float32 if BF16 is not desired
)

# Optionally configure generation settings (e.g., pad_token_id)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# -------------------------------------------------------------------------
# 2. Set Up Gaudi-Specific Training Configurations
# -------------------------------------------------------------------------
# GaudiConfig: performance optimizations (fused optimizer, etc.)
gaudi_config = GaudiConfig()
gaudi_config.use_fused_adam = True
gaudi_config.use_fused_clip_norm = True

# GaudiSFTConfig: hyperparameters and general training settings
training_args = GaudiSFTConfig(
    use_habana=True,               # Enable Gaudi training
    max_seq_length=512,            # Max tokens per sample
    output_dir="./results_deepseek_sft",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,            # Adjust based on your needs
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,                     # fp16 not supported in habana
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="epoch",
    push_to_hub=False,             # Set True if you want to push to the Hub
    hub_model_id="bihan/deepseek-llm-7b-sft-spider"  # Adjust if pushing to Hub
)
# Workaround
if not hasattr(model.generation_config, 'attn_softmax_bf16'):
    model.generation_config.attn_softmax_bf16 = False  # or True, based on your requirements
if not hasattr(model.generation_config, 'use_flash_attention'):
    model.generation_config.use_flash_attention = False  # or True, based on your requirements
if not hasattr(model.generation_config, 'flash_attention_recompute'):
    model.generation_config.flash_attention_recompute = False  # or True, based on your requirements
if not hasattr(model.generation_config, 'flash_attention_causal_mask'):
    model.generation_config.flash_attention_causal_mask = False  # or True, based on your requirements
# -------------------------------------------------------------------------
# 3. Load and Preprocess the Spider Dataset
# -------------------------------------------------------------------------
spider_dataset = load_dataset("spider")

# Example: We'll build a simple prompt: "Question: <q> SQL: " and use the query as the label.
def preprocess_function(examples):
    inputs = [f"Question: {q} SQL: " for q in examples["question"]]
    targets = examples["query"]
    
    # Encode the input prompt
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    # Encode the labels (the SQL query)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
    
    # Set the labels in the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_spider = spider_dataset.map(preprocess_function, batched=True)

# -------------------------------------------------------------------------
# 4. Initialize the GaudiSFTTrainer
# -------------------------------------------------------------------------
trainer = GaudiSFTTrainer(
    model=model,
    gaudi_config=gaudi_config,
    train_dataset=tokenized_spider["train"],
    eval_dataset=tokenized_spider["validation"],
    args=training_args,
    # Since we've already created "input_ids" and "labels" in preprocess,
    # we don't need to rely on "dataset_text_field" or auto-chunking.
    dataset_text_field=None, 
    tokenizer=tokenizer
)

# -------------------------------------------------------------------------
# 5. Fine-Tune the Model
# -------------------------------------------------------------------------
trainer.train()

# Optional: If you want to save the final weights locally
trainer.save_model("./final_deepseek_sft_model")

# -------------------------------------------------------------------------
# 6. (Optional) Push to Hugging Face Hub
# -------------------------------------------------------------------------
if training_args.push_to_hub:
    trainer.push_to_hub()
