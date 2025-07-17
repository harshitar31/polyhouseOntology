# Install required libraries (if not installed)
# !pip install -q transformers datasets torch accelerate bitsandbytes peft trl huggingface_hub

import transformers
print(f"Transformers version: {transformers._version_}")

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model

HF_TOKEN = ""
model_id = 'Stratos-Kakalis/complete-SPARQL-tuned-mistral'
dataset_path = '/content/drive/MyDrive/newdataset.jsonl'
output_dir = '/content/fine_tuned_mistral_sparql'

# --- LOAD DATASET ---
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

data = []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")
print(f"Loaded {len(data)} examples.")

for i in range(len(data)):
    data[i] = {
        "text": f"{data[i]['prompt']}\n{data[i]['completion']}"
    }

dataset = Dataset.from_list(data).train_test_split(test_size=0.1)

# --- TOKENIZATION ---
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- LOAD MODEL WITH QUANTIZATION ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True  # ✅ to support lower-memory GPUs
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

# --- APPLY LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --- TRAINING CONFIG ---
os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    eval_strategy="steps",  # ✅ Corrected
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    warmup_ratio=0.1,
    max_grad_norm=0.5,
    report_to="none"
)

# --- TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# --- SAVE ---
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# --- INFERENCE ---
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

sample_prompt = "Prompt: Write a SPARQL query to list all sensors observing temperature.\nCompletion:"
output = generator(sample_prompt, max_length=200, num_return_sequences=1)
print("Generated Output:", output[0]['generated_text'])
