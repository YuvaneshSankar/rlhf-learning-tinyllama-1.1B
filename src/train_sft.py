from transformers import AutoModelForCausalLM, AutoTokenizer , BitsandBytesConfig , TrainingArguments

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  #does quantization on the qauntization constants to reduce memory usage
    bnb_4bit_quant_type="nf4",  #normal float 4 ->datatype like fp4
    bnb_4bit_compute_dtype=torch.bfloat16  #what datatype to use while computation
)
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",        #Automatically assigns model layers to available devices ->like only gpu or cpu+gpu or multiple gpus...
    torch_dtype=torch.bfloat16
)

