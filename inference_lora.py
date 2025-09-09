from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-qa-lora")
tokenizer = AutoTokenizer.from_pretrained("./flan-t5-qa-lora")

def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(query_model("What is the capital of France?"))

