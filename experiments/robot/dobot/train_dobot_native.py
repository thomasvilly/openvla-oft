import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import tqdm

# --- ROBUST PATH SETUP ---
# Calculate path to the repo root (3 folders up)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))

# Add repo root to Python path so we can import 'prismatic' and 'dobot_dataset'
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- REPO IMPORTS ---
from dobot_dataset import DobotDataset
from prismatic.models.action_heads import L1RegressionActionHead
# --- CONFIG ---
MODEL_ID = "openvla/openvla-7b"
OUTPUT_DIR = "checkpoints/dobot_native"
DATA_ROOT = "/mnt/d/DOBOT/dataset_hdf5"

BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
LORA_RANK = 32
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 25 # Must match Dataset

def main():
    print("--- Initializing OpenVLA Native Training (L1 Regression) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 2. Load Dataset
    dataset = DobotDataset(DATA_ROOT, chunk_size=NUM_ACTIONS_CHUNK)
    print(f"Dataset Loaded: {len(dataset)} samples")

    # 3. Custom Collator
    # We need to manually stack because we are bypassing their complex RLDS collator
    def native_collate_fn(batch):
        prompts = [x['prompt'] for x in batch]
        images = [x['image'] for x in batch]
        actions = torch.stack([x['actions'] for x in batch]) # [B, 25, 7]
        
        # Process Inputs (Image + Text)
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        
        return inputs, actions

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=native_collate_fn)

    # 4. Load Model (Base VLA)
    print("Loading Base VLA...")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # 5. Add LoRA to VLA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=min(LORA_RANK, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    
    # 6. Initialize Action Head (The "OFT" part)
    # This is the separate regression network that predicts numbers
    print("Initializing L1 Regression Head...")
    action_head = L1RegressionActionHead(
        input_dim=vla.module.llm_dim if hasattr(vla, 'module') else vla.llm_dim,
        hidden_dim=vla.module.llm_dim if hasattr(vla, 'module') else vla.llm_dim,
        action_dim=ACTION_DIM
    ).to(device).to(torch.bfloat16) # Match model dtype

    # 7. Optimizer
    # We train BOTH the LoRA weights in the VLA AND the new Action Head
    optimizer = AdamW(
        list(vla.parameters()) + list(action_head.parameters()),
        lr=LEARNING_RATE
    )

    # 8. Training Loop
    vla.train()
    action_head.train()
    
    print("Starting Training Loop...")
    step = 0
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for inputs, gt_actions in pbar:
            # Move to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gt_actions = gt_actions.to(device).to(torch.bfloat16)

            # Forward Pass: Get Hidden States from VLA
            # We don't want the VLA to compute loss (it thinks it's doing text generation)
            # We just want the 'output_hidden_states'
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = vla(
                    **inputs, 
                    output_hidden_states=True
                )
                
                # Extract the hidden states relevant to actions
                # Logic copied from finetune.py:
                # We take the last layer's hidden states
                last_hidden_state = outputs.hidden_states[-1] # [B, Seq_Len, Dim]
                
                # We need to slice out the part of the sequence that corresponds to the output
                # The repo uses a mask, but for our simple prompt "In: ... Out:", 
                # the action embedding is typically at the very end.
                # However, the safest way matching their logic is to grab the LAST N tokens
                # corresponding to the chunk size * action dim. 
                # BUT, their L1 Head implementation is clever: it usually projects the *entire* # prompt sequence or specific tokens.
                
                # SIMPLIFICATION FOR SINGLE-GPU TRAINING:
                # We grab the last token's embedding and assume it summarizes the context, 
                # OR we map the last N tokens.
                # Looking at finetune.py line 692: they use `actions_hidden_states`.
                # For this setup, we will use the global average pooling of the last few tokens 
                # or the last token to seed the generation.
                
                # Let's trust the Action Head's expected input: [B, Chunk*Dim, Hidden]
                # To match dimensions, we project the last token:
                # (This is a simplified assumption for the "Native" script to get it running. 
                # The full repo uses a complex mask.)
                
                # Hack: Use the last token as the "summary"
                feature_embedding = last_hidden_state[:, -1, :].unsqueeze(1) # [B, 1, Dim]
                # Expand to chunk size needed by head? 
                # Actually, L1RegressionActionHead expects [B, Time, Dim].
                # We will feed the last embeddings.
                
                # Correct Logic: We need [B, 25, Dim] to predict 25 actions.
                # We take the last 25 tokens of the sequence.
                action_embeddings = last_hidden_state[:, -NUM_ACTIONS_CHUNK:, :]
                
                # Predict Actions
                pred_actions = action_head(action_embeddings) # [B, 25, 7]
                
                # Compute Loss (L1)
                loss = nn.L1Loss()(pred_actions, gt_actions)
                
            # Backward
            loss.backward()
            
            if (step + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": loss.item()})
            
            step += 1
            
    # Save
    print("Saving...")
    vla.save_pretrained(OUTPUT_DIR)
    torch.save(action_head.state_dict(), os.path.join(OUTPUT_DIR, "action_head.pt"))
    print("Done!")

if __name__ == "__main__":
    main()