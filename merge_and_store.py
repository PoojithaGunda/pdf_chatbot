import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Disable wandb if you don't use it
os.environ["WANDB_DISABLED"] = "true"

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"D:/Machine Learning and LLMs/LLMs/Mistral-7B-Instruct-v0.2"
DATASET_PATH = r"data/qa_pairs_instruction.jsonl"
OUTPUT_DIR = r"D:/Interns/pdf_extraction_using_chromadb/models/mistral_qlora_adapters"

# ---------------- QUANTIZATION CONFIG (4-bit) ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------- LOAD MODEL ----------------
print("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id

# ---------------- PREPARE MODEL FOR TRAINING ----------------
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ---------------- LORA CONFIG ----------------
lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # LoRA alpha (scaling factor)
    target_modules=[               # Which layers to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------- LOAD DATASET ----------------
print("Loading dataset...")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

# Split into train/eval (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ---------------- TRAINING ARGUMENTS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,              # Adjust based on dataset size
    per_device_train_batch_size=1,   # Reduce if OOM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # Effective batch size = 4
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",        # Memory-efficient optimizer
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,                       # Use mixed precision
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_total_limit=2,              # Keep only 2 checkpoints
    report_to="none"                 # Disable wandb
)

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",       # Column name in JSONL
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# ---------------- TRAIN ----------------
print("\n🚀 Starting QLoRA fine-tuning...\n")
trainer.train()

# ---------------- SAVE ADAPTERS ----------------
print("\n💾 Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Fine-tuning complete!")
print(f"📁 Adapters saved to: {OUTPUT_DIR}")