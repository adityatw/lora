from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load tokenizer from your saved path
tokenizer = AutoTokenizer.from_pretrained("./flan-t5-qa-lora")

# Load base model separately
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, "./flan-t5-qa-lora")

def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(query_model("What is the capital of France?"))

