import os
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

os.environ["WANDB_PROJECT"] = "fine_tune_with_sft"


@dataclass
class ScriptArguments:
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per GPU for training."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per GPU for evaluation."}
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the SFTTrainer."},
    )
    max_steps: int = field(
        default=-1, metadata={"help": "How many optimizer update steps to take"}
    )
    output_dir: str = field(
        default="./results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    optim: Optional[str] = field(
        default="adafactor",
        metadata={"help": "The optimizer to use."},
    )


full_dataset = load_dataset("lucasmccabe-lmi/openai_humaneval_alpaca_style", split="train")

dataset_dict = full_dataset.train_test_split(test_size=0.1)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj"],
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # args=SFTConfig(output_dir="/tmp", max_steps=5, per_device_train_batch_size=1,
    #                per_device_eval_batch_size=1, report_to="wandb"),
    args=SFTConfig(output_dir="/tmp", max_steps=5, per_device_train_batch_size=1,
                   per_device_eval_batch_size=1, report_to=None),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
)

trainer.train()


# model_save_path = "/tmp/fine_tuned_model"
# trainer.model.save_pretrained(model_save_path)
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")


# Merge the model
def merge_and_push():
    base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    model: PeftModel = PeftModel.from_pretrained(base_model, model_id=model_save_path)
    model.merge_and_unload()
    model.push_to_hub("opt-350-finetuned-test")
    tokenizer.push_to_hub("opt-350-finetuned-test")

# merge_and_push()

# if __name__ == "__main__":
#     parser = HfArgumentParser(ScriptArguments)
#     args = parser.parse_args_into_dataclasses()[0]
#     print(args.per_device_train_batch_size)
