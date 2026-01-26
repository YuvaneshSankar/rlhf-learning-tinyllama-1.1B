import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./models/sft/checkpoint-6000")

# Check adapter status
print(f"Active adapters: {model.active_adapters}")
print(f"Adapter config: {model.peft_config}")

# Get a sample of adapter weights to verify they're not zeros/random
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {param.data.mean().item():.6f}")
        print(f"  Std: {param.data.std().item():.6f}")
        print(f"  Min: {param.data.min().item():.6f}")
        print(f"  Max: {param.data.max().item():.6f}")
        break  # Just check first LoRA layer

# Test generation
test_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is the capital of France?\n### Response:\n"

input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model.device)

print("\n=== Testing Generation ===")
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
