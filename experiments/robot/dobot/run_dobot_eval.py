"""
run_dobot_offline_eval.py

Offline evaluation for OpenVLA trained on Dobot data.
Features:
- Generates CSV report (Pred vs Actual)
- Optional Visualization and Plot Saving
"""

import os
import sys
import csv
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

# Append paths to find your existing utils
sys.path.append(os.getcwd()) 

# Import standard utils
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_processor,
)

@dataclass
class EvalConfig:
    # --- Paths (Updated to match your snippet) ---
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--600_chkpt"
    
    # Dataset Config
    dataset_name: str = "dobot_dataset"
    # Pointing explicitly to the version folder as requested
    data_dir: str = "/mnt/d/DOBOT/rlds_dataset_folder/dobot_dataset/1.0.0" 
    split: str = "train[:10%]" 

    # --- Model Params ---
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # --- Input Params ---
    center_crop: bool = True  
    num_images_in_input: int = 2
    
    # --- Action Params ---
    unnorm_key: str = "dobot_dataset" 
    use_proprio: bool = False
    use_film: bool = False
    
    # --- Visualization & Output Configuration ---
    # Set these to True only when you need to debug visually
    visualize: bool = False        # Pop-up window? (Default: False for speed)
    save_plots: bool = False       # Save every step as PNG? (Default: False)
    save_csv: bool = True          # Save the data report? (Default: True)
    save_first_image: bool = True  # Save just the first step's image for sanity check?
    
    output_dir: str = "./eval_outputs"

@draccus.wrap()
def main(cfg: EvalConfig):
    set_seed_everywhere(7)
    
    # 1. Load Model & Processor
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)
    
    # 2. Load RLDS Dataset
    print(f"Loading dataset {cfg.dataset_name} from {cfg.data_dir}...")
    # Note: If this fails, try removing '/dobot_dataset/1.0.0' from the path
    builder = tfds.builder_from_directory(builder_dir=cfg.data_dir)
    ds = builder.as_dataset(split=cfg.split)
    
    # Handle Output Directory
    if cfg.save_plots or cfg.save_csv or cfg.save_first_image:
        os.makedirs(cfg.output_dir, exist_ok=True)
        abs_path = os.path.abspath(cfg.output_dir)
        print(f"\n[INFO] Saving outputs to: {abs_path}")
    
    # Initialize CSV Writing
    csv_file = None
    csv_writer = None
    if cfg.save_csv:
        csv_path = os.path.join(cfg.output_dir, "results.csv")
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write Header
        csv_writer.writerow([
            "Episode", "Step", "Task", "MSE", 
            "Pred_X", "Pred_Y", "Pred_Z", 
            "GT_X", "GT_Y", "GT_Z"
        ])
        print(f"[INFO] CSV report will be saved to: {csv_path}")

    total_mse = 0.0
    step_count = 0
    quit_eval = False

    # 3. Main Evaluation Loop
    try:
        for episode_idx, episode in enumerate(ds):
            if quit_eval: break
            print(f"\n--- Evaluating Episode {episode_idx} ---")
            
            for step_idx, step in enumerate(episode['steps']):
                if quit_eval: break

                # --- A. Prepare Observation ---
                img_overhead = step['observation']['image'].numpy()
                img_wrist = step['observation']['image_wrist'].numpy()
                
                task_label = "do the task"
                if 'language_instruction' in step:
                    inst = step['language_instruction'].numpy()
                    task_label = inst.decode('utf-8') if isinstance(inst, bytes) else str(inst)

                obs = {
                    "full_image": img_overhead,
                    "wrist_image": img_wrist,
                    "state": step['observation']['state'].numpy() if cfg.use_proprio else None
                }

                # --- B. Inference ---
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
                    print(f"Inference error at step {step_idx}: {e}")
                    continue

                # --- C. Compare with Ground Truth ---
                if isinstance(action_pred, list):
                    action_pred = action_pred[0]
                    
                action_gt = step['action'].numpy()

                # Calculate MSE on XYZ
                mse = np.mean((action_pred[:3] - action_gt[:3])**2)
                total_mse += mse
                step_count += 1
                
                # Print status
                print(f"Step {step_idx} | MSE: {mse:.5f} | Pred: {np.round(action_pred[:3], 3)}")

                # --- D. Save to CSV ---
                if cfg.save_csv and csv_writer:
                    csv_writer.writerow([
                        episode_idx, step_idx, task_label, mse,
                        action_pred[0], action_pred[1], action_pred[2],
                        action_gt[0], action_gt[1], action_gt[2]
                    ])

                # --- E. Visualization & Interaction ---
                
                # Check if we should render this frame
                should_render = cfg.visualize or cfg.save_plots
                
                # Special Case: Save ONLY the very first image for sanity check
                if cfg.save_first_image and episode_idx == 0 and step_idx == 0:
                    should_render = True

                if should_render:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Show Overhead
                    ax[0].imshow(img_overhead)
                    ax[0].set_title("Overhead")
                    ax[0].axis('off')
                    
                    # Show Wrist
                    ax[1].imshow(img_wrist)
                    ax[1].set_title("Wrist")
                    ax[1].axis('off')
                    
                    # Overlay Text
                    info = (
                        f"Task: {task_label}\n"
                        f"Pred Delta: {np.round(action_pred[:3], 3)}\n"
                        f"GT Delta:   {np.round(action_gt[:3], 3)}\n"
                        f"MSE:  {mse:.4f}"
                    )
                    fig.text(0.5, 0.05, info, ha='center', fontsize=12, 
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
                    
                    # 1. Save First Image Logic
                    if cfg.save_first_image and episode_idx == 0 and step_idx == 0:
                        save_path = os.path.join(cfg.output_dir, "sanity_check_first_frame.png")
                        plt.savefig(save_path)
                        print(f"[INFO] Saved sanity check image to {save_path}")

                    # 2. Save All Plots Logic
                    if cfg.save_plots:
                        save_path = os.path.join(cfg.output_dir, f"ep{episode_idx}_step{step_idx}.png")
                        plt.savefig(save_path)
                    
                    # 3. Visualization Logic
                    if cfg.visualize:
                        plt.show(block=False)
                        print("  [Interaction] Press SPACE for next, 'q' to quit.")
                        
                        while True:
                            if plt.waitforbuttonpress(timeout):
                                break
                    
                    plt.close(fig)

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"[INFO] CSV file closed and saved.")

    print(f"\nFinal Average MSE over {step_count} steps: {total_mse / step_count:.5f}")

if __name__ == "__main__":
    main()