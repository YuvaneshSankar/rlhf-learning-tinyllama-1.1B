import argparse
import math
import os

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import SFTTrainer, SFTConfig


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

    # Attach LoRA adapter weights
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tokenizer


def build_eval_dataset(eval_samples: int):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if eval_samples and eval_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(min(eval_samples, len(dataset))))
    return dataset


def evaluate(adapter_dir: str, max_seq_length: int, per_device_eval_batch_size: int, eval_samples: int):
    model, tokenizer = load_model_and_tokenizer(max_seq_length, adapter_dir)

    # Keep the model in eval mode
    model.eval()

    eval_ds = build_eval_dataset(eval_samples)

    sft_cfg = SFTConfig(
        output_dir="./models/eval_tmp",
        per_device_eval_batch_size=per_device_eval_batch_size,
        remove_unused_columns=False,
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=eval_ds,  # not training; using for tokenizer pipeline
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=sft_cfg,
    )

    metrics = trainer.evaluate(eval_dataset=eval_ds)
    eval_loss = float(metrics.get("eval_loss", float("nan")))
    ppl = math.exp(eval_loss) if math.isfinite(eval_loss) else float("nan")

    print("=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"perplexity: {ppl}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-adapted TinyLlama (Unsloth) on Alpaca formatting")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./models/sft_unsloth",
        help="Path to the saved LoRA adapter directory (contains adapter_model.safetensors)",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=512,
        help="Number of samples from Alpaca train to use for evaluation",
    )

    args = parser.parse_args()

    evaluate(
        adapter_dir=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        per_device_eval_batch_size=args.batch_size,
        eval_samples=args.eval_samples,
    )


if __name__ == "__main__":
    main()
