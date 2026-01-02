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
- 5-Dim Action Space (X, Y, Z, Roll, Gripper)
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

class DobotDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.3.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.1.0': 'Stride=2, Pruned Action Space (5-Dim), Discrete Gripper.',
        '1.2.0': 'Stride=2, Pruned Action Space (5-Dim), Discrete Gripper, Filter Homing Steps & Dead Zones.',
        '1.3.0': 'Move to overfit attempt'
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
                            shape=(5,), # Pruned State (X,Y,Z,R,G)
                            dtype=np.float32,
                            doc='Robot state (x, y, z, r, gripper)',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(5,), # Pruned Action (X,Y,Z,R,G)
                        dtype=np.float32,
                        doc='Robot action (d_x, d_y, d_z, d_r, gripper_cmd)',
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
        root_dir = "/mnt/d/DOBOT/dataset_hdf5/interactive_session" 
        
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
        STRIDE = 1
        # Minimum steps required for OpenVLA to form a valid training chunk.
        # OpenVLA often needs a buffer for action chunking. 16 is a safe safety margin.
        MIN_EPISODE_LENGTH = 10
        
        global_idx = 0 
        
        # --- FILTERS ---
        MIN_MOVE_MM = 1.0   
        MAX_MOVE_MM = 50.0  
        # ---------------

        for file_path in file_paths:
            try:
                with h5py.File(file_path, 'r') as f:
                    imgs_top = f['observations/images/top'][1:]
                    states = f['observations/state'][:]
                    
                    n_steps = len(imgs_top)
                    instruction = f.attrs.get('instruction', "do the task")
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode('utf-8')
                        
                    episode = []
                    
                    # STRIDE LOOP
                    for i in range(0, n_steps, STRIDE):
                        
                        img = imgs_top[i]
                        
                        full_state = states[i]
                        curr_xyzr = full_state[[0, 1, 2, 3]]
                        curr_grip = full_state[6]
                        
                        future_state = states[i + STRIDE]
                        future_xyzr = future_state[[0, 1, 2, 3]]
                        future_grip = future_state[6]
                        
                        # --- VELOCITY & GRIPPER LOGIC ---
                        delta_xyzr = future_xyzr - curr_xyzr
                        
                        # Discretize Gripper
                        grip_diff = future_grip - curr_grip
                        if grip_diff > 0.5:
                            grip_cmd = 1.0
                        elif grip_diff < -0.5:
                            grip_cmd = -1.0
                        else:
                            grip_cmd = 0.0
                        
                        # --- FORCE DROP AT END ---
                        is_last_step = (i >= (n_steps - 2 * STRIDE))
                        # if is_last_step:
                        #     grip_cmd = -1.0
                        #     grip_diff = -1.0 
                        # ------------------------------

                        # --- DATA CLEANING ---
                        move_mag = np.linalg.norm(delta_xyzr[:3])
                        
                        # 1. Filter Outliers
                        if move_mag > MAX_MOVE_MM:
                            continue 
                            
                        # 2. Filter Laziness
                        has_grip_change = abs(grip_diff) > 0.1
                        if move_mag < MIN_MOVE_MM and not has_grip_change and not is_last_step:
                            continue 
                        # -------------------------------

                        action_5d = np.concatenate([delta_xyzr, [grip_cmd]]).astype(np.float32)
                        state_5d = np.concatenate([curr_xyzr, [curr_grip]]).astype(np.float32)
                        
                        episode.append({
                            'observation': {
                                'image': img,
                                'state': state_5d,
                            },
                            'action': action_5d,
                            'discount': 1.0,
                            'reward': float(is_last_step),
                            'is_first': i == 0,
                            'is_last': is_last_step,
                            'is_terminal': is_last_step,
                            'language_instruction': instruction,
                            'language_embedding': np.zeros(512, dtype=np.float32),
                        })

                    # --- CRITICAL FIX: CHECK LENGTH BEFORE YIELDING ---
                    if len(episode) < MIN_EPISODE_LENGTH:
                        # Skip this episode entirely if it's too short (prevents loader crash)
                        # print(f"Skipping short episode in {os.path.basename(file_path)} (Len: {len(episode)})")
                        continue

                    # If we survive, fix flags and yield
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
    version = "1.3.0"
    data_dir = "/mnt/d/DOBOT/rlds_dataset_folder"
    target_dir = os.path.join(data_dir, dataset_name, version)

    # Manual Cleanup
    if os.path.exists(target_dir):
        print(f"[Cleanup] Removing partial/old build at: {target_dir}")
        shutil.rmtree(target_dir)
    else:
        print(f"[Cleanup] No existing {version} folder found. Starting fresh.")

    builder = DobotDataset(data_dir=data_dir)
    builder.download_and_prepare()
    
    print("--- Success! Dataset 1.3.0 Generated ---")