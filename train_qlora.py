import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

os.environ["WANDB_DISABLED"] = "true"

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\models\qwen"  # Update path
DATASET_PATH = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\data\qa_pairs_qwen.jsonl"
OUTPUT_DIR = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\models\qwen_qlora_adapters"

# ---------------- QUANTIZATION CONFIG ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)


# ---------------- LOAD MODEL ----------------
print("Loading Qwen 2.5 model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True  # Important for Qwen
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True  # Important for Qwen
)

# Qwen tokenizer settings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.pad_token_id = tokenizer.pad_token_id

# ---------------- PREPARE MODEL ----------------
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# ---------------- LORA CONFIG (Qwen-specific) ----------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
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
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# ---------------- FORMATTING FUNCTION FOR QWEN ----------------
def format_chat_template(example):
    """
    Apply Qwen's chat template to format messages
    """
    if "messages" in example:
        # Use Qwen's apply_chat_template
        formatted = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted}
    return example

# Apply formatting
train_dataset = train_dataset.map(format_chat_template)
eval_dataset = eval_dataset.map(format_chat_template)

# ---------------- TRAINING ARGUMENTS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,      # Qwen 1.5B is smaller, can use batch_size=2
    per_device_train_batch_size = 1
gradient_accumulation_steps = 8

    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none"
)

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# ---------------- TRAIN ----------------
print("\n🚀 Starting Qwen 2.5 QLoRA fine-tuning...\n")
trainer.train()

# ---------------- SAVE ----------------
print("\n💾 Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Fine-tuning complete!")
print(f"📁 Adapters saved to: {OUTPUT_DIR}")