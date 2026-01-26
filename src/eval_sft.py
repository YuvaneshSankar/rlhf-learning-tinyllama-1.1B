import math
import os

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import Trainer, TrainingArguments


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}"
        )
        if input_text:
            text += f"\n### Input:\n{input_text}"
        text += f"\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}


def load_model_and_tokenizer(max_seq_length: int, adapter_dir: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model = PeftModel.from_pretrained(model, adapter_dir)

    # DEBUG: Check if adapter is loaded
    print(f"=== Adapter loaded from: {adapter_dir} ===")
    print(f"Active adapters: {model.active_adapters}")
    print(f"Is PEFT model: {hasattr(model, 'active_adapters')}")

    return model, tokenizer


def build_eval_dataset(eval_samples: int, tokenizer):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if eval_samples and eval_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(min(eval_samples, len(dataset))))

    # Tokenize the dataset with proper label masking
    def tokenize_function(examples):
        texts = examples["text"]

        # Tokenize full sequences
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors=None
        )

        labels = []
        for text, input_ids in zip(texts, tokenized["input_ids"]):
            # Find where the response starts
            response_marker = "### Response:\n"
            response_idx = text.find(response_marker)

            if response_idx == -1:
                # Fallback: if no response marker, use full sequence
                labels.append(input_ids)
                continue

            # Tokenize everything up to and including the response marker
            prompt_text = text[:response_idx + len(response_marker)]
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=2048,
                add_special_tokens=False
            )["input_ids"]

            # Create label sequence: -100 for prompt, actual tokens for response
            label = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]

            # Ensure label length matches input_ids length
            label = label[:len(input_ids)]
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"]
    )
    return dataset


def evaluate(adapter_dir: str, max_seq_length: int, per_device_eval_batch_size: int, eval_samples: int):
    model, tokenizer = load_model_and_tokenizer(max_seq_length, adapter_dir)

    # Keep the model in eval mode
    model.eval()

    eval_ds = build_eval_dataset(eval_samples, tokenizer)

    eval_args = TrainingArguments(
        output_dir="./models/eval_tmp",
        per_device_eval_batch_size=per_device_eval_batch_size,
        remove_unused_columns=False,
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_ds,
        args=eval_args,
    )

    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss", float("nan")))
    ppl = math.exp(eval_loss) if math.isfinite(eval_loss) else float("nan")

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"perplexity: {ppl}")


if __name__ == "__main__":
    # Default parameters
    adapter_dir = "./models/sft/checkpoint-6000"
    max_seq_length = 2048
    per_device_eval_batch_size = 2
    eval_samples = 512

    evaluate(
        adapter_dir=adapter_dir,
        max_seq_length=max_seq_length,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_samples=eval_samples,
    )
