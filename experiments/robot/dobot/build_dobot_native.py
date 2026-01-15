import h5py
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import shutil

_DESCRIPTION = """
Dobot Magician dataset for OpenVLA.
Features:
- Stride Action
- 4-Dim Action Space (X, Y, Z, Gripper)
- Discretized Gripper Commands (-1, 0, 1)
- FILTERS: Removes dead frames (<1mm) and teleportation artifacts (>50mm).
"""

_CITATION = """
@article{dobot_dataset,
  title={Dobot Magician Dataset},
  author={User},
  year={2025}
}
"""

VER = "1.4.0"

def gaussian_smooth_pure_numpy(data, sigma):
    """
    Applies Gaussian smoothing to action data using pure NumPy.
    data: shape (N, 3) for XYZ
    sigma: smoothing strength (Standard=1.0)
    """
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum() 
    
    smoothed = np.zeros_like(data)
    
    # Convolve each axis (X, Y, Z) independently
    for col in range(data.shape[1]):
        padded = np.pad(data[:, col], radius, mode='edge')
        smoothed[:, col] = np.convolve(padded, kernel, mode='valid')
        
    return smoothed

class DobotDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(VER)
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.1.0': 'Stride=2, Pruned Action Space (5-Dim), Discrete Gripper.',
        '1.2.0': 'Stride=2, Pruned Action Space (5-Dim), Discrete Gripper, Filter Homing Steps & Dead Zones.',
        '1.3.0': 'Move to overfit attempt',
        '1.4.0': 'Update action delta smoothness, still attempt overfit'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(4,), # Pruned State (X,Y,Z,G)
                            dtype=np.float32,
                            doc='Robot state (x, y, z, gripper)',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,), # Pruned Action (X,Y,Z,G)
                        dtype=np.float32,
                        doc='Robot action (d_x, d_y, d_z, gripper_cmd)',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. (Optional)'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }),
            supervised_keys=None,
            homepage='https://google.com',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Point this to your hdf5 folder
        root_dir = "/mnt/d/DOBOT/dataset_hdf5/simple_session"
        
        # Walk through all job folders
        file_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".h5"):
                    file_paths.append(os.path.join(root, file))
        
        print(f"Found {len(file_paths)} HDF5 files.")
        
        return {
            'train': self._generate_examples(file_paths),
        }

    def _generate_examples(self, file_paths):
        # --- CONFIGURATION ---
        STRIDE = 3            # Lookahead (0.3s @ 10Hz)
        SIGMA = 1.0           # Gaussian Smoothing Strength
        MIN_EPISODE_LENGTH = 16 
        
        # FILTERS
        MIN_MOVE_MM = 1.0   
        MAX_MOVE_MM = 100.0  
        
        global_idx = 0 

        for file_path in file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    # 1. Load Raw Data
                    imgs_top = f['observations/images/top'][1:] # Skip first frame (often dark/blur)
                    states = f['observations/state'][:]
                    
                    # Align lengths (Camera sometimes has 1 more/less frame than state)
                    min_len = min(len(imgs_top), len(states))
                    imgs_top = imgs_top[:min_len]
                    states = states[:min_len]
                    
                    instruction = f.attrs.get('instruction', "do the task")
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode('utf-8')

                    # 2. PRE-CALCULATE ACTIONS (Rolling Window)
                    # We need valid pairs: State[i] vs State[i+STRIDE]
                    # This slices arrays so 'curr' is 0..N-STRIDE, 'future' is STRIDE..N
                    curr_xyz = states[:-STRIDE, 0:3]
                    future_xyz = states[STRIDE:, 0:3]
                    
                    curr_grip = states[:-STRIDE, 6]
                    future_grip = states[STRIDE:, 6]
                    
                    # Calculate Raw Deltas
                    raw_actions_xyz = future_xyz - curr_xyz
                    
                    # 3. APPLY GAUSSIAN SMOOTHING
                    smoothed_actions_xyz = gaussian_smooth_pure_numpy(raw_actions_xyz, sigma=SIGMA)
                    
                    episode = []
                    n_steps = len(smoothed_actions_xyz) # This is effectively (Total_Frames - STRIDE)

                    # 4. BUILD EPISODE
                    for i in range(n_steps):
                        img = imgs_top[i]
                        
                        # XYZ Action (Smoothed)
                        delta_xyz = smoothed_actions_xyz[i]
                        
                        # Gripper Action (Discrete Logic - No smoothing on binary buttons!)
                        grip_diff = future_grip[i] - curr_grip[i]
                        if grip_diff > 0.5: grip_cmd = 1.0
                        elif grip_diff < -0.5: grip_cmd = -1.0
                        else: grip_cmd = 0.0
                        
                        # Force drop at very end of episode (optional safety)
                        is_last_step = (i == n_steps - 1)
                        
                        # --- FILTERS ---
                        move_mag = np.linalg.norm(delta_xyz)
                        has_grip_change = abs(grip_cmd) > 0.1
                        
                        # Filter: Huge teleportation (Tracking error)
                        if move_mag > MAX_MOVE_MM: continue 
                            
                        # Filter: Dead frames (Robot not moving, gripper not clicking)
                        if move_mag < MIN_MOVE_MM and not has_grip_change and not is_last_step:
                             continue 
                        # ---------------

                        # Pack Data
                        # State includes Gripper (Idx 6)
                        current_state_packed = np.concatenate([states[i, 0:3], [states[i, 6]]]).astype(np.float32)
                        action_packed = np.concatenate([delta_xyz, [grip_cmd]]).astype(np.float32)
                        
                        episode.append({
                            'observation': {
                                'image': img,
                                'state': current_state_packed,
                            },
                            'action': action_packed,
                            'discount': 1.0,
                            'reward': float(is_last_step),
                            'is_first': i == 0,
                            'is_last': is_last_step,
                            'is_terminal': is_last_step,
                            'language_instruction': instruction,
                            'language_embedding': np.zeros(512, dtype=np.float32),
                        })

                    # Final Check
                    if len(episode) < MIN_EPISODE_LENGTH:
                        continue

                    # Patch flags just in case filtering broke continuity
                    episode[0]['is_first'] = True
                    episode[-1]['is_last'] = True
                    episode[-1]['is_terminal'] = True
                    episode[-1]['reward'] = 1.0

                    sample_key = f"{global_idx:06d}"
                    global_idx += 1
                    
                    yield sample_key, {
                        'steps': episode,
                        'episode_metadata': {'file_path': file_path}
                    }

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    print("--- Running Direct Builder (Bypassing TFDS CLI) ---")
    
    dataset_name = "dobot_dataset"
    data_dir = "/mnt/d/DOBOT/rlds_dataset_folder"
    target_dir = os.path.join(data_dir, dataset_name, VER)

    # Manual Cleanup
    if os.path.exists(target_dir):
        print(f"[Cleanup] Removing partial/old build at: {target_dir}")
        shutil.rmtree(target_dir)
    else:
        print(f"[Cleanup] No existing {VER} folder found. Starting fresh.")

    builder = DobotDataset(data_dir=data_dir)
    builder.download_and_prepare()