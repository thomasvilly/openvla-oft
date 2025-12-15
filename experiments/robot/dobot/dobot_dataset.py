import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import glob
import os


NORMALIZATION_STATS = {
    "x_min": -150.0, "x_max": 250.0,
    "y_min": -200.0, "y_max": 200.0,
    "z_min": -100.0,  "z_max": 150.0,
    "r_min": -150.0, "r_max": 100.0,
    "grip_min": 0, "grip_max": 1
}

class DobotDataset(Dataset):
    def __init__(self, data_root, chunk_size=15):
        """
        data_root: Path to your 'dataset_hdf5' folder
        chunk_size: How many future actions to predict (Default 15 for 10-30Hz control)
        """
        self.chunk_size = chunk_size
        self.episode_files = sorted(glob.glob(os.path.join(data_root, "**", "*.h5"), recursive=True))
        
        if not self.episode_files:
            raise ValueError(f"No .h5 files found in {data_root}")

        # Build an Index Map
        # We need to map a global index (e.g., "Item 500") to ("Episode 3, Step 40")
        self.index_map = []
        print(f"Found {len(self.episode_files)} episodes. Indexing...")
        
        for ep_path in self.episode_files:
            try:
                with h5py.File(ep_path, 'r') as f:
                    total_steps = len(f['action'])
                    # We can use every step as a starting point
                    for i in range(total_steps):
                        self.index_map.append({
                            'path': ep_path,
                            'step_idx': i,
                            'total_steps_in_ep': total_steps
                        })
            except Exception as e:
                print(f"Skipping corrupt file {ep_path}: {e}")

        print(f"Total training samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def normalize_action(self, action_vector):
        """
        Squashes [x, y, z, r, pitch, yaw, grip] to [-1, 1]
        """
        # Unpack
        x, y, z, r, pitch, yaw, grip = action_vector
        
        # Normalize simple min-max
        x_norm = 2 * (x - NORMALIZATION_STATS['x_min']) / (NORMALIZATION_STATS['x_max'] - NORMALIZATION_STATS['x_min']) - 1
        y_norm = 2 * (y - NORMALIZATION_STATS['y_min']) / (NORMALIZATION_STATS['y_max'] - NORMALIZATION_STATS['y_min']) - 1
        z_norm = 2 * (z - NORMALIZATION_STATS['z_min']) / (NORMALIZATION_STATS['z_max'] - NORMALIZATION_STATS['z_min']) - 1
        r_norm = 2 * (r - NORMALIZATION_STATS['r_min']) / (NORMALIZATION_STATS['r_max'] - NORMALIZATION_STATS['r_min']) - 1
        grip_norm = 2 * (grip - NORMALIZATION_STATS['grip_min']) / (NORMALIZATION_STATS['grip_max'] - NORMALIZATION_STATS['grip_min']) - 1
        
        # Pitch/Yaw are dummies (0), keep them 0
        return np.array([x_norm, y_norm, z_norm, r_norm, 0.0, 0.0, grip_norm], dtype=np.float32)

    def __getitem__(self, idx):
        sample_info = self.index_map[idx]
        current_step = sample_info['step_idx']
        
        with h5py.File(sample_info['path'], 'r') as f:
            # 1. Get Instruction
            instruction = f.attrs.get('instruction', "do something")
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')

            # 2. Get Images (Top-Down is primary)
            # OpenVLA expects 224x224
            img_raw = f['observations/images/top'][current_step]
            img_resized = cv2.resize(img_raw, (224, 224))
            # Convert BGR (OpenCV) to RGB (AI Standard)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1] and move channels to first dim [3, 224, 224]
            img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0

            # 3. Get Action Chunk (The Future)
            # We want actions from [t] to [t + chunk_size]
            actions_raw = f['action'][:] # Read all (simpler logic)
            
            chunk = []
            for i in range(self.chunk_size):
                target_idx = current_step + i
                if target_idx < len(actions_raw):
                    raw_act = actions_raw[target_idx]
                else:
                    # If we run off the end of the episode, repeat the last action
                    raw_act = actions_raw[-1]
                
                chunk.append(self.normalize_action(raw_act))
            
            action_tensor = torch.tensor(np.array(chunk))

        return {
            "image": img_tensor,       # [3, 224, 224]
            "instruction": instruction, # Str
            "actions": action_tensor   # [Chunk_Size, 7]
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Update this path to where your dobot_data_collector folder sits relative to this file
    # Example: If they are side-by-side: "../dobot_data_collector/dataset_hdf5"
    DATA_PATH = "../dobot_data_collector/dataset_hdf5" 
    
    print(f"Testing loader with path: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("ERROR: Path does not exist. Please edit DATA_PATH in the script.")
    else:
        ds = DobotDataset(DATA_PATH, chunk_size=15)
        if len(ds) > 0:
            sample = ds[0]
            print("\nSUCCESS! Sample Loaded:")
            print(f"Instruction: '{sample['instruction']}'")
            print(f"Image Shape: {sample['image'].shape} (Should be [3, 224, 224])")
            print(f"Action Chunk: {sample['actions'].shape} (Should be [15, 7])")
            print(f"First Action (Norm): {sample['actions'][0]}")