# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM

dataset = load_dataset("trl-lib/tldr", split="test")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Deepseek-V2-GRPO", logging_steps=10, per_device_train_batch_size=1, bf16=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset
)
trainer.train()