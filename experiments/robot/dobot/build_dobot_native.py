import tensorflow_datasets as tfds
import tensorflow as tf
import h5py
import numpy as np
import glob
import os

# --- CONFIGURATION ---
INPUT_HDF5_DIR = "/mnt/d/DOBOT/dataset_hdf5"
OUTPUT_RLDS_DIR = "/mnt/d/DOBOT/rlds_dataset_folder"

class DobotDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dobot_dataset (RLDS Format)."""
    
    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dobot dataset using Temporal Deltas for OpenVLA.",
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32), # Now Temporal Deltas
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
                        'image_wrist': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'), # New key
                        'state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    'language_instruction': tfds.features.Text(),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                }),
                'episode_metadata': tfds.features.FeaturesDict({'file_path': tfds.features.Text()}),
            }),
            supervised_keys=None, 
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        file_paths = sorted(glob.glob(os.path.join(INPUT_HDF5_DIR, "**", "*.h5"), recursive=True))
        return {'train': self._generate_examples(file_paths)}

    def _generate_examples(self, file_paths):
        for episode_idx, file_path in enumerate(file_paths):
            try:
                with h5py.File(file_path, 'r') as f:
                    # 'state' in your HDF5 is the actual robot pose recorded
                    states = f['observations/state'][:] 
                    images_top = f['observations/images/top'][:]
                    images_side = f['observations/images/side'][:]
                    
                    instruction = f.attrs.get('instruction', "pick up block")
                    if isinstance(instruction, bytes): instruction = instruction.decode('utf-8')
                    
                    num_steps = len(states)
                    episode_steps = []

                    # Process up to num_steps - 1 because we need i+1 for the delta
                    for i in range(num_steps):
                        img_top = images_top[i]
                        img_side = images_side[i]

                        if img_top.shape[-1] == 3: img_top = img_top[... , ::-1] # BGR to RGB
                        if img_side.shape[-1] == 3: img_side = img_side[... , ::-1]
                        
                        # --- CALC DELTA ---
                        if i < num_steps - 1:
                            # Arm Deltas (X, Y, Z, R, P, Y)
                            delta_arm = states[i+1][:6] - states[i][:6]
                            # Gripper: Use Absolute state from step i+1
                            # This tells the model: "Given this image, make the gripper look like THIS next"
                            target_gripper = states[i+1][6]
                        else:
                            delta_arm = np.zeros(6)
                            target_gripper = states[i][6]

                        # Combine into the 7-DOF action vector
                        action_vec = np.append(delta_arm, target_gripper).astype(np.float32)
                        
                        episode_steps.append({
                            'action': action_vec,
                            'observation': {
                                'image': img_top,
                                'image_wrist': img_side,
                                'state': states[i].astype(np.float32), # Store absolute current state
                            },
                            'language_instruction': instruction,
                            'is_first': i == 0,
                            'is_last': i == (num_steps - 1),
                            'is_terminal': i == (num_steps - 1),
                        })
                    
                    yield f"episode_{episode_idx}", {
                        'steps': episode_steps,
                        'episode_metadata': {'file_path': file_path}
                    }
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    builder = DobotDataset(data_dir=OUTPUT_RLDS_DIR)
    builder.download_and_prepare()