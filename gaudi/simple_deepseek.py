import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiSFTConfig, GaudiSFTTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load the Dataset
    logger.info("Loading the IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    logger.info(f"Dataset loaded with {len(dataset)} samples.")

    # 2. Initialize the Tokenizer
    logger.info("Initializing the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

    # 3. Tokenize the Dataset
    logger.info("Tokenizing the dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    logger.info("Tokenization complete.")

    # 4. Filter Out Samples with Empty Input IDs (Optional but Recommended)
    logger.info("Filtering out samples with empty 'input_ids'...")
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
    logger.info(f"Filtered dataset now has {len(tokenized_dataset)} samples.")

    # 5. Set Dataset Format to PyTorch Tensors
    logger.info("Setting dataset format to PyTorch tensors...")
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    logger.info("Dataset format set.")

    # 6. Initialize the Data Collator for Language Modeling
    logger.info("Initializing the data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Disable Masked Language Modeling for Causal LM
    )
    logger.info("Data collator initialized.")

    # 7. Configure Training Arguments with GaudiSFTConfig
    logger.info("Configuring training arguments...")
    training_args = GaudiSFTConfig(
        use_habana=True,
        max_seq_length=512,
        output_dir="/tmp/deepseek-sft-output",
        per_device_train_batch_size=8,  # Adjust based on your hardware capabilities
        num_train_epochs=3,             # Adjust the number of epochs as needed
        logging_steps=100,              # Adjust logging frequency
        save_steps=500,                 # Adjust checkpoint saving frequency
        evaluation_strategy="steps",    # Change to "epoch" if preferred
        eval_steps=500,                 # Adjust evaluation frequency
        save_total_limit=2,             # Limit the number of saved checkpoints
    )
    logger.info("Training arguments configured.")

    # 8. Initialize Gaudi Configuration
    logger.info("Initializing Gaudi configuration...")
    gaudi_config = GaudiConfig()
    gaudi_config.use_fused_adam = True
    gaudi_config.use_fused_clip_norm = True
    logger.info("Gaudi configuration initialized.")

    # 9. Initialize the GaudiSFTTrainer
    logger.info("Initializing the GaudiSFTTrainer...")
    trainer = GaudiSFTTrainer(
        model="deepseek-ai/deepseek-llm-7b-base",
        gaudi_config=gaudi_config,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,  # Use the standard data collator
    )
    logger.info("GaudiSFTTrainer initialized.")

    # 10. Verify Model Outputs (Optional but Helpful for Debugging)
    logger.info("Verifying model outputs with a sample batch...")
    sample_batch = trainer.get_train_dataloader().dataset[:2]  # Get first 2 samples
    outputs = trainer.model(**sample_batch)
    if 'loss' in outputs:
        logger.info("Model is correctly returning loss.")
    else:
        logger.error("Model is NOT returning loss. Check data collator and labels.")
        return

    # 11. Start Training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    # 12. Save the Trained Model (Optional)
    logger.info("Saving the trained model...")
    trainer.save_model("/tmp/deepseek-sft-output/final-model")
    logger.info("Model saved.")

if __name__ == "__main__":
    main()
