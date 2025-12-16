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
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Dobot dataset for OpenVLA fine-tuning.",
            features=tfds.features.FeaturesDict({
                # RLDS Structure: An Episode contains a Dataset of Steps
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(None, None, 3), encoding_format='jpeg'),
                        'state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    'language_instruction': tfds.features.Text(),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                }),
                # Episode-level metadata (optional)
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            }),
            supervised_keys=None, 
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        file_paths = sorted(glob.glob(os.path.join(INPUT_HDF5_DIR, "**", "*.h5"), recursive=True))
        print(f"Found {len(file_paths)} episodes to process.")
        return {
            'train': self._generate_examples(file_paths),
        }

    def _generate_examples(self, file_paths):
        """Yields episodes."""
        for episode_idx, file_path in enumerate(file_paths):
            try:
                with h5py.File(file_path, 'r') as f:
                    actions = f['action'][:] 
                    images = f['observations/images/top'][:] 
                    
                    instruction = f.attrs.get('instruction', "do task")
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode('utf-8')
                    
                    num_steps = len(actions)
                    episode_steps = []

                    for i in range(num_steps):
                        img = images[i]
                        if img.shape[-1] == 3:
                            img = img[..., ::-1] # BGR to RGB
                        
                        episode_steps.append({
                            'action': actions[i].astype(np.float32),
                            'observation': {
                                'image': img,
                                'state': actions[i].astype(np.float32),
                            },
                            'language_instruction': instruction,
                            'is_first': i == 0,
                            'is_last': i == (num_steps - 1),
                            'is_terminal': i == (num_steps - 1),
                        })
                    
                    # Yield the Full Episode
                    yield f"episode_{episode_idx}", {
                        'steps': episode_steps,
                        'episode_metadata': {'file_path': file_path}
                    }

            except Exception as e:
                print(f"Skipping file {file_path}: {e}")

if __name__ == "__main__":
    print(f"Building Dataset in: {OUTPUT_RLDS_DIR}")
    builder = DobotDataset(data_dir=OUTPUT_RLDS_DIR)
    builder.download_and_prepare()
    print("Success! Dataset built with valid metadata.")