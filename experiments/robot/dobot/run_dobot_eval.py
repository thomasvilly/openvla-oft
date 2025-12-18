"""
run_dobot_offline_eval.py

Offline evaluation for OpenVLA trained on Dobot data.
Features:
- Handles Action Chunking (10 steps) -> Receding Horizon Control
- Handles 5-Dim Action Space (Pads to 7 for reporting)
- Generates detailed CSV report & Summary Plots
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
    # --- Paths ---
    # Update this to your actual Run 3 checkpoint path!
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--1500_chkpt"
    
    # Dataset Config
    dataset_name: str = "dobot_dataset"
    # UPDATED: Point to 1.1.0 (The strided/pruned version)
    data_dir: str = "/mnt/d/DOBOT/rlds_dataset_folder/dobot_dataset/1.1.0" 
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
    # UPDATED: We used proprioception in Run 3!
    use_proprio: bool = True
    use_film: bool = False
    
    # --- Visualization & Output Configuration ---
    visualize: bool = False        # Set True to see it live
    save_plots: bool = False       # Set True to save step images
    save_csv: bool = True          # Save the math
    save_first_image: bool = True  # Always good for debugging
    
    output_dir: str = "./eval_outputs"
    action_scale = 25.0

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
        
        # Write Expanded Header
        csv_writer.writerow([
            "Episode", "Step", "Task", "MSE_XYZ", 
            "Pred_X", "GT_X", 
            "Pred_Y", "GT_Y", 
            "Pred_Z", "GT_Z",
            "Pred_Roll", "GT_Roll",
            "Pred_Pitch", "GT_Pitch",
            "Pred_Yaw", "GT_Yaw",
            "Pred_Grip", "GT_Grip"
        ])
        print(f"[INFO] CSV report will be saved to: {csv_path}")

    # --- Metrics Storage for Plotting ---
    all_preds = []
    all_gts = []
    
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
                
                # FIX: Color Swap (BGR -> RGB) if needed
                # If your visualizer needed this, your eval needs it too for the model to see 'Red' correctly.
                # Uncomment the next line if you confirmed your data is BGR
                # img_overhead = img_overhead[..., ::-1]
                # img_wrist = img_wrist[..., ::-1]

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

                # --- C. Handle Chunking & Ground Truth ---
                
                # Receding Horizon Control: Take the FIRST action of the chunk
                # Check for List OR Numpy Array (Output is usually (10, 5))
                if isinstance(action_pred, list):
                    action_pred = action_pred[0]
                elif isinstance(action_pred, np.ndarray) and action_pred.ndim > 1:
                    action_pred = action_pred[0]
                
                if len(action_pred) >= 4:
                    action_pred[:4] = action_pred[:4] * cfg.action_scale
                    
                action_gt = step['action'].numpy()

                # Pad vectors to ensure 7 dims (XYZ, RPY, Grip)
                # Model outputs 5 dims. We pad with 0s to fill the 7-slot report.
                pred_7d = np.pad(action_pred, (0, max(0, 7-len(action_pred))), mode='constant')
                gt_7d = np.pad(action_gt, (0, max(0, 7-len(action_gt))), mode='constant')

                # Store for final plotting
                all_preds.append(pred_7d)
                all_gts.append(gt_7d)

                # Calculate MSE on XYZ
                mse = np.mean((pred_7d[:3] - gt_7d[:3])**2)
                total_mse += mse
                step_count += 1
                
                # Print status
                def fmt_vec(v): return np.array2string(v, precision=2, separator=',', suppress_small=True)
                print(f"Step {step_idx} | MSE: {mse:.4f}")
                print(f"  Pred: {fmt_vec(pred_7d)}")
                print(f"  GT:   {fmt_vec(gt_7d)}")

                # --- D. Save to CSV ---
                if cfg.save_csv and csv_writer:
                    csv_writer.writerow([
                        episode_idx, step_idx, task_label, mse,
                        pred_7d[0], gt_7d[0],  # X
                        pred_7d[1], gt_7d[1],  # Y
                        pred_7d[2], gt_7d[2],  # Z
                        pred_7d[3], gt_7d[3],  # Roll
                        pred_7d[4], gt_7d[4],  # Gripper
                        pred_7d[5], gt_7d[5],  # Pitch
                        pred_7d[6], gt_7d[6]   # Yaw
                    ])

                # --- E. Visualization (Images) ---
                should_render = cfg.visualize or cfg.save_plots
                if cfg.save_first_image and episode_idx == 0 and step_idx == 0:
                    should_render = True

                if should_render:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # For display, we might want to swap BGR->RGB if not done earlier
                    # If you uncommented the swap above, you don't need it here.
                    show_overhead = img_overhead[..., ::-1] # Swap for plotting
                    show_wrist = img_wrist[..., ::-1]       # Swap for plotting
                    
                    ax[0].imshow(show_overhead)
                    ax[0].set_title("Overhead")
                    ax[0].axis('off')
                    
                    ax[1].imshow(show_wrist)
                    ax[1].set_title("Wrist")
                    ax[1].axis('off')
                    
                    info = (
                        f"Step: {step_idx} | MSE: {mse:.4f}\n"
                        f"Pred XYZ: {np.round(pred_7d[:3], 2)}\n"
                        f"GT XYZ:   {np.round(gt_7d[:3], 2)}"
                    )
                    fig.text(0.5, 0.05, info, ha='center', fontsize=11, 
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
                    
                    if cfg.save_first_image and episode_idx == 0 and step_idx == 0:
                        plt.savefig(os.path.join(cfg.output_dir, "sanity_check_first_frame.png"))
                        print("[INFO] Saved sanity check image.")

                    if cfg.save_plots:
                        plt.savefig(os.path.join(cfg.output_dir, f"ep{episode_idx}_step{step_idx}.png"))
                    
                    if cfg.visualize:
                        plt.show(block=False)
                        plt.pause(0.1) 
                    
                    plt.close(fig)

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user.")
    finally:
        if csv_file:
            csv_file.close()
            print(f"[INFO] CSV file closed.")

    # --- F. GENERATE SUMMARY PLOTS ---
    print("\n[INFO] Generating Summary Graphs...")
    if len(all_preds) > 0:
        all_preds = np.array(all_preds) 
        all_gts = np.array(all_gts)     
        
        dim_names = ["X", "Y", "Z", "Roll", "Gripper", "Pitch", "Yaw"]
        
        fig_summary, axes = plt.subplots(7, 1, figsize=(12, 24), sharex=True)
        
        for i, ax in enumerate(axes):
            ax.plot(all_gts[:, i], label='Ground Truth', color='black', alpha=0.6, linewidth=2)
            ax.plot(all_preds[:, i], label='Prediction', color='blue', linestyle='--', alpha=0.8, linewidth=1.5)
            
            diff = all_gts[:, i] - all_preds[:, i]
            ax.plot(diff, label='Diff (GT - Pred)', color='red', alpha=0.5, linewidth=1)
            ax.fill_between(range(len(diff)), diff, 0, color='red', alpha=0.1)
            
            ax.set_ylabel(f"{dim_names[i]}")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if i == 0:
                ax.legend(loc="upper right")

        axes[-1].set_xlabel("Evaluation Steps (Concatenated Episodes)")
        fig_summary.suptitle(f"Evaluation Metrics: {cfg.dataset_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        
        save_plot_path = os.path.join(cfg.output_dir, "summary_metrics.png")
        plt.savefig(save_plot_path)
        print(f"[INFO] Summary plots saved to: {save_plot_path}")
        
        if cfg.visualize:
            plt.show()
    else:
        print("[WARN] No data collected to plot.")

    print(f"Final Average MSE over {step_count} steps: {total_mse / max(1, step_count):.5f}")

if __name__ == "__main__":
    main()