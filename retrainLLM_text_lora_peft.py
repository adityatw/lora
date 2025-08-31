"""
    Author: Aditya Wresniyandaka, Fall 2025
    Description: Fine-tunes flan-t5-small on SQuAD v1.1 using LoRA adapters for efficient training on consumer-grade hardware.

    Dependencies:
      - transformers
      - peft
      - datasets
      - evaluate

    Usage:
     Run with Python 3.12+ and CUDA-enabled GPU (tested on NVIDIA GeForce GTX 1650)

"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_dataset
import random
import evaluate

# -----------------------------
# 1. W&B Init
# -----------------------------
wandb.init(
    project="plain-text-lora",
    name="flan-t5-small-lora-5000",
    config={
        "model": "google/flan-t5-small",
        "batch_size": 1,
        "grad_accum": 2,
        "learning_rate": 5e-5,
        "epochs": 5,
        "max_seq_length": 64,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 50,
        "num_samples": 5000,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05
    }
)

# -----------------------------
# 2. Load Dataset and Sample
# -----------------------------
dataset = load_dataset("squad", split="train")
dataset = dataset.shuffle(seed=42).select(range(wandb.config.num_samples))

# -----------------------------
# 3. Load Tokenizer + Base Model (8-bit)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(wandb.config.model)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    wandb.config.model,
    device_map="auto",
    torch_dtype=torch.float16  # xxplicit dtype for mixed precision
)

base_model.gradient_checkpointing_enable()
base_model.enable_input_require_grads()  # this is key for LoRA + 8-bit

# -----------------------------
# 4. Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=wandb.config.lora_r,
    lora_alpha=wandb.config.lora_alpha,
    target_modules=["q", "v"],
    lora_dropout=wandb.config.lora_dropout,
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(base_model, lora_config)

# -----------------------------
# 5. Preprocess Dataset
# -----------------------------
def preprocess_data(examples):
    # Ensure questions is a list of strings
    questions = [q if isinstance(q, str) else "" for q in examples["question"]]

    # Extract first answer safely
    first_answers = []
    for ans_dict in examples["answers"]:
        text_list = ans_dict.get("text", [])
        if isinstance(text_list, list) and len(text_list) > 0 and isinstance(text_list[0], str):
            first_answers.append(text_list[0])
        else:
            first_answers.append("")

    # Tokenize inputs
    inputs = tokenizer(
        questions,
        truncation=True,
        padding="max_length",
        max_length=wandb.config.max_seq_length
    )

    # Tokenize labels
    labels = tokenizer(
        first_answers,
        truncation=True,
        padding="max_length",
        max_length=wandb.config.max_seq_length
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Train / Val split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]
val_data = val_data.select(range(100))

# -----------------------------
# 6. Compute Metrics
# -----------------------------
squad_metric = evaluate.load("squad")

def pad_to_max(arrays, pad_value=0):
    """Pad a list of ndarrays to the same shape."""
    max_shape = [max(s) for s in zip(*[a.shape for a in arrays])]
    result = np.full((len(arrays), *max_shape), pad_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        slices = tuple(slice(0, s) for s in a.shape)
        result[i][slices] = a
    return result

def compute_metrics(eval_preds):
    print("\n🔍 eval_preds type:", type(eval_preds))
    if isinstance(eval_preds, tuple):
        print("🔍 tuple length:", len(eval_preds))
        for i, item in enumerate(eval_preds):
            print(f"  └─ Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")

    predictions = eval_preds.predictions
    labels = eval_preds.label_ids

    # Convert tensors to numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    normed_predictions = []
    for p in predictions:
        if p.ndim == 3:   
            p = np.argmax(p, axis=-1)
        normed_predictions.append(p)
    predictions = normed_predictions

    normed_labels = []
    for l in labels:
        normed_labels.append(l)
    labels = normed_labels


    predictions = pad_to_max(predictions, pad_value=0)
    labels = pad_to_max(labels, pad_value=-100)

    # Compute accuracy
    mask = labels != -100
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {"token_accuracy": round(float(accuracy), 4)}

# -----------------------------
# 7. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=wandb.config.eval_steps,
    save_steps=wandb.config.save_steps,
    logging_steps=wandb.config.logging_steps,
    per_device_train_batch_size=wandb.config.batch_size,
    per_device_eval_batch_size=wandb.config.batch_size,
    gradient_accumulation_steps=wandb.config.grad_accum,
    learning_rate=wandb.config.learning_rate,
    num_train_epochs=wandb.config.epochs,
    logging_dir="./logs",
    report_to="wandb",
    load_best_model_at_end=True,
    fp16=False, 
    max_grad_norm=1.0
)

# -----------------------------
# 8. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --------------------------------------------------
# 9. Train - this is an ateempt to address OOM error
# --------------------------------------------------
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("⚠️ CUDA OOM detected. Clearing cache and exiting...")
        import torch
        torch.cuda.empty_cache()
        exit(1)
    else:
        raise e

torch.cuda.empty_cache()

# -----------------------------
# 10. Save Model + Inference
# -----------------------------
model.save_pretrained("./flan-t5-qa-lora")
tokenizer.save_pretrained("./flan-t5-qa-lora")
wandb.finish()

# Example inference
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "Instruction: What is the capital of France?\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, 
                         max_new_tokens=20, 
                         num_beams=1, 
                         early_stopping=True,
                         return_dict_in_generate=False,
                         output_scores=False,
                         do_sample=False)
print("Generated Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))