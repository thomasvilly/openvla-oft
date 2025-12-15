import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Optional

# Import your custom dataset loader
# (Assumes dobot_dataset.py is in the root or accessible path)
import sys
sys.path.append(os.path.abspath("../../../")) # Add repo root to path
from dobot_dataset import DobotDataset

# --- CONFIGURATION ---
MODEL_ID = "openvla/openvla-7b"
OUTPUT_DIR = "checkpoints/dobot_finetune"
DATA_ROOT = """"C:\Code\dobot\dataset_hdf5"""

# Training Hyperparameters
BATCH_SIZE = 8          # Reduce to 4 if GPU runs out of memory
GRAD_ACCUMULATION = 2   # Effective batch size = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 50         # For "Overfit Test" (Use 50-100 to force memorization)
LORA_RANK = 32          # Power of the fine-tuning

def main():
    print(f"--- Starting Dobot Fine-Tuning ---")
    
    # 1. Load Processor (Handles image resizing & text tokenization)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 2. Load the Dataset
    print(f"Loading data from: {DATA_ROOT}")
    train_dataset = DobotDataset(DATA_ROOT, chunk_size=15)
    print(f"Found {len(train_dataset)} training samples.")

    # 3. Define Data Collator 
    # (Puts a list of samples into a batch tensor)
    def collate_fn(examples):
        # Stack images
        pixel_values = torch.stack([ex["image"] for ex in examples])
        
        # Tokenize Instructions
        instructions = [ex["instruction"] for ex in examples]
        inputs = processor(text=instructions, return_tensors="pt", padding=True)
        
        # Stack Actions
        actions = torch.stack([ex["actions"] for ex in examples])
        
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": actions # In OpenVLA, 'labels' are the actions we want to predict
        }

    # 4. Load Model (in 4-bit to save memory)
    print("Loading Model... (This takes time)")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    )

    # 5. Apply LoRA (Low-Rank Adaptation)
    # This makes fine-tuning possible on consumer GPUs
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules="all-linear", # Fine-tune all linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # VLA is treated as a Language Model
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,                  # Use Mixed Precision
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,         # Only keep last 2 checkpoints
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none"            # Disable WandB for now
    )

    # 7. Start Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    print("Starting Training Loop...")
    trainer.train()
    
    # 8. Save Final Model
    print("Saving Adapter...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    print("Done!")

if __name__ == "__main__":
    main()