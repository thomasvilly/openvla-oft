"""
run_dobot_offline_eval.py

Offline evaluation for OpenVLA trained on Dobot data.
Features:
- Visualizes Side-by-Side Comparison
- Interactive Mode (Spacebar to advance)
- Saves CSV report (Pred vs Actual)
- Configurable Plot Saving and Visualization
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
    # --- Paths (Updated per your request) ---
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--600_chkpt"
    
    # Dataset Config
    dataset_name: str = "dobot_dataset"
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
    visualize: bool = False         # Set True to see pop-up window and step through with Spacebar
    save_plots: bool = False       # Set True to save images to disk (False to save disk space)
    save_csv: bool = True          # Set True to save the CSV report
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
    builder = tfds.builder_from_directory(builder_dir=cfg.data_dir)
    ds = builder.as_dataset(split=cfg.split)
    
    # Handle Output Directory
    if cfg.save_plots or cfg.save_csv:
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
                # Only enter this block if we are visualizing OR saving plots
                if cfg.visualize or cfg.save_plots:
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
                    
                    # Save Logic (Only if save_plots is True)
                    if cfg.save_plots:
                        save_path = os.path.join(cfg.output_dir, f"ep{episode_idx}_step{step_idx}.png")
                        plt.savefig(save_path)
                    
                    # Display Logic (Only if visualize is True)
                    if cfg.visualize:
                        plt.show(block=False)
                        print("  [Interaction] Press SPACE for next, 'q' to quit.")
                        
                        # Interactive loop
                        while True:
                            # waitforbuttonpress returns True if key pressed, False if mouse click
                            if plt.waitforbuttonpress():
                                # If you want to check for specific key 'q', current matplotlib backends vary
                                # but usually just closing the window manually or Ctrl+C is easier.
                                break
                    
                    plt.close(fig)

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        # Ensure CSV is closed properly
        if csv_file:
            csv_file.close()
            print(f"[INFO] CSV file closed and saved.")

    print(f"\nFinal Average MSE over {step_count} steps: {total_mse / step_count:.5f}")

if __name__ == "__main__":
    main()