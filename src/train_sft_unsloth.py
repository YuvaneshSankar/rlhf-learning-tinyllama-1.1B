import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
#we can use unsloth import UnslothSFTTrainer for better integration, but here we use trl's SFTTrainer as an example
from trl import SFTTrainer, SFTConfig
# from trl import SFTConfig
# from unsloth import UnslothSFTTrainer

max_seq=2048

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="unsloth/tinyllama-bnb-4bit",
  max_seq_length=max_seq,
  dtype=None,  # Auto-detects best dtype
  load_in_4bit=True,  # QLoRA
)

target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

model=FastLanguageModel.get_peft_model(
  model,
  r=16,
  target_modules=target_modules,
  lora_alpha=16,
  lora_dropout=0,  # Unsloth recommends 0
  bias="none",
  # task_type="CAUSAL_LM" ->FastLanguageModel.get_peft_model() internally passes task_type="CAUSAL_LM" by default
)

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}"
        if input_text:
            text += f"\n### Input:\n{input_text}"
        text += f"\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

sft_config = SFTConfig(
    output_dir="./models/sft",
    num_train_epochs=1, #so basically lora converges faster so if we have higher epochs then it may overfit
    per_device_train_batch_size=2,  # Number of samples per GPU step (because of memory constraints)
    per_device_eval_batch_size=2,    #same idea for eval though
    gradient_accumulation_steps=4,  #update wights once every 4 forward+backward passes....
    #so effective batch size=>2*4 = 8
    learning_rate=2e-4,
    optim="paged_adamw_8bit",  #This is a memory-optimized optimizer from bitsandbytes
    logging_steps=10,  #just logs training metrics every 10 steps
    save_steps=500,  #saves model checkpoint every 500 steps
    eval_steps=500,  #run evaluation every 500 steps
    max_seq_length=2048,  # TinyLlama max
    packing=False,  # This is a TRL-specific optimization....minmizes padding tokens in batches
    fp16=False,  # Use bfloat16 above
    bf16=True,
    # report_to="wandb",  # Optional: wandb login
    remove_unused_columns=False,
)

trainer = SFTTrainer(  # Gets auto-patched by Unsloth
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq,
    args=sft_config,
)



trainer.train()
model.save_pretrained("./models/sft_unsloth")