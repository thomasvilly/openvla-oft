from torch.utils.data import DataLoader
from dobot_dataset import DobotDataset
import torch

# --- CONFIG ---
# Point this to your actual data folder
DATA_PATH = """C:\Code\dobot\dataset_hdf5"""

def test_the_bridge():
    print(f"--- TESTING DATASET LOADING FROM: {DATA_PATH} ---")
    
    # 1. Instantiate the Dataset
    try:
        ds = DobotDataset(DATA_PATH, chunk_size=15)
    except Exception as e:
        print(f"\nCRITICAL FAIL: Could not find/load data. Error: {e}")
        return

    # 2. Wrap it in a PyTorch DataLoader (Simulates Training)
    # Batch Size = 4 means we grab 4 random moments from your recording at once
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    print(f"Total Samples: {len(ds)}")
    print("Fetching one batch...\n")

    # 3. Grab one batch
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"CRITICAL FAIL: DataLoader crashed. Error: {e}")
        return

    # 4. INSPECT THE TENSORS (The Health Check)
    images = batch['image']
    actions = batch['actions']
    instructions = batch['instruction']

    print(f"Images Shape:   {images.shape}")
    print(f"   -> Expected: [4, 3, 224, 224]")
    print(f"   -> Min/Max:  {images.min():.3f} / {images.max():.3f} (Should be 0.0 to 1.0)\n")

    print(f"Actions Shape:  {actions.shape}")
    print(f"   -> Expected: [4, 15, 7]")
    print(f"   -> Min/Max:  {actions.min():.3f} / {actions.max():.3f} (Should be approx -1.0 to 1.0)\n")

    print(f"Instructions:   {instructions}")
    print("   -> Should be a list of strings like 'Pick up the red block'\n")

    # 5. Check Normalization Logic
    # If the action values are huge (e.g. 150.0), normalization failed.
    if torch.abs(actions).max() > 1.5:
        print("!!! WARNING: Action values look too large! Normalization might be broken.")
    else:
        print("SUCCESS: Data looks valid and ready for the GPU.")

if __name__ == "__main__":
    test_the_bridge()