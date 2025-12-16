import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import glob
import os

# --- CONFIGURATION ---
# Based on your physical Dobot limits
NORMALIZATION_STATS = {
    "x_min": -150.0, "x_max": 250.0,
    "y_min": -200.0, "y_max": 200.0,
    "z_min": -100.0, "z_max": 150.0,
    "r_min": -150.0, "r_max": 100.0,
    "grip_min": 0,   "grip_max": 1
}

class DobotDataset(Dataset):
    def __init__(self, data_root, chunk_size=25): # Default chunk size in repo is 25
        self.chunk_size = chunk_size
        self.episode_files = sorted(glob.glob(os.path.join(data_root, "**", "*.h5"), recursive=True))
        
        if not self.episode_files:
            raise ValueError(f"No .h5 files found in {data_root}")

        self.index_map = []
        print(f"Found {len(self.episode_files)} episodes. Indexing...")
        
        for ep_path in self.episode_files:
            try:
                with h5py.File(ep_path, 'r') as f:
                    total_steps = len(f['action'])
                    # We need enough future steps for the chunk
                    for i in range(total_steps):
                        self.index_map.append({
                            'path': ep_path,
                            'step_idx': i
                        })
            except Exception as e:
                print(f"Skipping corrupt file {ep_path}: {e}")

    def __len__(self):
        return len(self.index_map)

    def normalize_action(self, action_vector):
        x, y, z, r, pitch, yaw, grip = action_vector
        
        # Normalize to [-1, 1]
        x_norm = 2 * (x - NORMALIZATION_STATS['x_min']) / (NORMALIZATION_STATS['x_max'] - NORMALIZATION_STATS['x_min']) - 1
        y_norm = 2 * (y - NORMALIZATION_STATS['y_min']) / (NORMALIZATION_STATS['y_max'] - NORMALIZATION_STATS['y_min']) - 1
        z_norm = 2 * (z - NORMALIZATION_STATS['z_min']) / (NORMALIZATION_STATS['z_max'] - NORMALIZATION_STATS['z_min']) - 1
        r_norm = 2 * (r - NORMALIZATION_STATS['r_min']) / (NORMALIZATION_STATS['r_max'] - NORMALIZATION_STATS['r_min']) - 1
        grip_norm = 2 * (grip - NORMALIZATION_STATS['grip_min']) / (NORMALIZATION_STATS['grip_max'] - NORMALIZATION_STATS['grip_min']) - 1
        
        # Return 7-dim vector (Fixed pitch/yaw to 0.0 if not used)
        return np.array([x_norm, y_norm, z_norm, r_norm, 0.0, 0.0, grip_norm], dtype=np.float32)

    def __getitem__(self, idx):
        sample_info = self.index_map[idx]
        current_step = sample_info['step_idx']
        
        with h5py.File(sample_info['path'], 'r') as f:
            # 1. Instruction
            raw_instr = f.attrs.get('instruction', "do something")
            if isinstance(raw_instr, bytes):
                raw_instr = raw_instr.decode('utf-8')
            
            # FORMATTING: The repo expects this exact prompt structure
            instruction = f"In: What action should the robot take to {raw_instr.lower()}?\nOut:"

            # 2. Image (Return RAW PIL)
            img_raw = f['observations/images/top'][current_step]
            img_rgb = img_raw[..., ::-1] # BGR to RGB
            image_pil = Image.fromarray(img_rgb)

            # 3. Actions (Chunked & Normalized)
            actions_raw = f['action'][:]
            chunk = []
            for i in range(self.chunk_size):
                target_idx = current_step + i
                if target_idx < len(actions_raw):
                    raw_act = actions_raw[target_idx]
                else:
                    raw_act = actions_raw[-1] # Repeat last action if near end
                chunk.append(self.normalize_action(raw_act))
            
            action_tensor = torch.tensor(np.array(chunk))

        return {
            "image": image_pil, 
            "prompt": instruction,
            "actions": action_tensor
        }