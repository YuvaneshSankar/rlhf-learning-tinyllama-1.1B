import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer , BitsAndBytesConfig , TrainingArguments
from trl import SFTTrainer,SFTConfig
model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_name = "tatsu-lab/alpaca"  # 52k instructions

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  #does quantization on the qauntization constants to reduce memory usage
    bnb_4bit_quant_type="nf4",  #normal float 4 ->datatype like fp4
    bnb_4bit_compute_dtype=torch.float16  #what datatype to use while computation
)

model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",        #Automatically assigns model layers to available devices ->like only gpu or cpu+gpu or multiple gpus...
    torch_dtype=torch.float16
)


tokenizer=AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
  tokenizer.pad_token=tokenizer.eos_token
#So some tokenizers dont have pad token defined and it is important for batching or else we get error during training....so we set it to eos token


peft_config = LoraConfig(
    r=16, #The rank of the low-rank decomposition.
    lora_alpha=32, #Scaling factor....now alpha/r=>32/16=2..so this scales the lora updates before adding them to the original weights
    lora_dropout=0.05,
    bias="none", #no need to update bias terms during training
    task_type="CAUSAL_LM" ,#causal language modeling task(basically auto regressive language modeling)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #These are the layers within the model that will be adapted using LoRA.
    #query , key , value , ouptput projection layers and feed forward network layers
    #gate->controls activation flow , up_proj->expands hidden dimensions, down_proj->compresses them back
)

dataset=load_dataset(dataset_name,split="train")
dataset=dataset.train_test_split(test_size=0.1)


#Define training arguments using SFTConfig
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
    packing=True,  # This is a TRL-specific optimization....minmizes padding tokens in batches
    fp16=True,  # Use bfloat16 above
    bf16=False,
    # report_to="wandb",  # Optional: wandb login
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    # tokenizer=tokenizer,
    dataset_text_field="text",  # TRL automatically formats Alpaca-style datasets into a text field
)


trainer.train()
trainer.save_model() #this saves -> LoRA adapter weights only,Adapter configuration,Training arguments
tokenizer.save_pretrained("./models/sft") #this saves tokenizer.json,tokenizer_config.json,special_tokens_map.json

#what is not saved
#full base model
#deos not merge LoRA into the base model
#tokenizer