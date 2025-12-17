"""
run_dobot_offline_eval.py

Evaluates a fine-tuned OpenVLA policy on the Dobot RLDS dataset.
Replicates the logic of run_libero_eval.py but for offline data.
"""

import os
import sys
import numpy as np
import torch
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

# Append paths so we can import your existing utils
sys.path.append("../..") 
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
    get_image_resize_size,
)
from experiments.robot.openvla_utils import (
    get_processor,
)

@dataclass
class EvalConfig:
    # Model Configuration
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--600_chkpt"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Input Configuration
    center_crop: bool = True
    num_images_in_input: int = 2   # Overhead + Wrist
    use_proprio: bool = False      # Set to True if you trained with proprio
    use_film: bool = False
    use_l1_regression: bool = False
    use_diffusion: bool = False
    
    # Dataset Configuration
    dataset_name: str = "dobot_dataset"
    data_dir: str = "/mnt/d/DOBOT/rlds_dataset_folder/dobot_dataset/1.0.0" # Path to your RLDS data
    split: str = "train[:10%]"  # Evaluate on first 10% or change to 'val' if available
    
    # Un-normalization
    # IMPORTANT: This key must match the dataset name used during training/stat generation
    unnorm_key: str = "dobot_dataset" 

    # Visualization
    visualize: bool = True
    save_plots: bool = False
    output_dir: str = "./eval_outputs"

@draccus.wrap()
def main(cfg: EvalConfig):
    set_seed_everywhere(7)
    
    # 1. Initialize Model & Processor
    # We use the utils exactly as Libero does to ensure consistent loading
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)
    
    # 2. Load RLDS Dataset
    print(f"Loading dataset {cfg.dataset_name}...")
    builder = tfds.builder_from_directory(builder_dir=cfg.data_dir)
    ds = builder.as_dataset(split=cfg.split)
    
    # Create output directory
    if cfg.save_plots:
        os.makedirs(cfg.output_dir, exist_ok=True)

    total_mse = 0.0
    step_count = 0

    # 3. Evaluation Loop
    # Iterate over episodes in the dataset
    for episode_idx, episode in enumerate(ds):
        print(f"\n--- Evaluating Episode {episode_idx} ---")
        
        # Iterate over steps in the episode
        for step_idx, step in enumerate(episode['steps']):
            
            # --- Prepare Observation Dictionary ---
            # We must map your RLDS keys to the keys 'get_action' expects.
            # get_vla_action expects: 'full_image' and keys containing 'wrist'
            
            # Extract raw images (numpy)
            img_overhead = step['observation']['image'].numpy()
            img_wrist = step['observation']['image_wrist'].numpy()
            
            # Extract instruction
            # If instruction is bytes, decode it; otherwise use default
            task_label = "do the task"
            if 'language_instruction' in step:
                inst = step['language_instruction'].numpy()
                task_label = inst.decode('utf-8') if isinstance(inst, bytes) else str(inst)

            # Construct the observation dict expected by get_action
            obs = {
                "full_image": img_overhead,
                "wrist_image": img_wrist,
                # Include state if using proprio
                "state": step['observation']['state'].numpy() if cfg.use_proprio else None
            }

            # --- Run Inference ---
            # This calls the exact same logic as Libero:
            # 1. Prepares images (resize/crop)
            # 2. Tokenizes text
            # 3. Concatenates tensors correctly
            # 4. Predicts and un-normalizes action
            try:
                action_pred = get_action(
                    cfg=cfg,
                    model=model,
                    obs=obs,
                    task_label=task_label,
                    processor=processor,
                    use_film=cfg.use_film
                )
            except Exception as e:
                print(f"Inference failed at step {step_idx}: {e}")
                continue

            # --- Compare with Ground Truth ---
            action_gt = step['action'].numpy()

            # Note: get_action returns a list of actions (chunk), usually length 1 for OpenVLA.
            # We take the first one.
            if isinstance(action_pred, list):
                action_pred = action_pred[0]
            
            # Calculate MSE (Position only, first 3 dims)
            # Adjust slicing if your action space is different
            mse = np.mean((action_pred[:3] - action_gt[:3])**2)
            total_mse += mse
            step_count += 1
            
            print(f"Step {step_idx} | Task: {task_label}")
            print(f"  Pred (XYZ): {np.round(action_pred[:3], 3)}")
            print(f"  GT   (XYZ): {np.round(action_gt[:3], 3)}")
            print(f"  MSE: {mse:.5f}")

            # --- Visualization ---
            if cfg.visualize or cfg.save_plots:
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Show Overhead
                ax[0].imshow(img_overhead)
                ax[0].set_title("Overhead View")
                ax[0].axis('off')
                
                # Show Wrist
                ax[1].imshow(img_wrist)
                ax[1].set_title("Wrist View")
                ax[1].axis('off')
                
                # Overlay text
                info_text = (
                    f"Task: {task_label}\n"
                    f"Step: {step_idx}\n"
                    f"Pred Delta: {np.round(action_pred[:3], 3)}\n"
                    f"True Delta: {np.round(action_gt[:3], 3)}\n"
                    f"MSE: {mse:.4f}"
                )
                
                fig.text(0.5, 0.05, info_text, ha='center', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
                
                if cfg.save_plots:
                    save_path = os.path.join(cfg.output_dir, f"ep{episode_idx}_step{step_idx}.png")
                    plt.savefig(save_path)
                    
                if cfg.visualize:
                    plt.show(block=False)
                    plt.pause(0.5) 
                
                plt.close(fig)

    print(f"Final Average MSE: {total_mse / step_count:.5f}")

if __name__ == "__main__":
    main()