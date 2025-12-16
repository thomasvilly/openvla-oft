import os
import sys
import time

# --- SETUP PATHS (Same as training script) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- CONFIG ---
# Update this to match your real path
DATA_ROOT = "/mnt/d/DOBOT/dataset_hdf5" 

def test_1_cuda_access():
    print("\n--- TEST 1: CUDA & GPU Access ---")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available. Found {torch.cuda.device_count()} device(s).")
            print(f"   Current Device: {torch.cuda.get_device_name(0)}")
            
            # Simple Tensor Test
            x = torch.tensor([1.0]).cuda()
            print("✅ Tensor successfully moved to GPU memory.")
        else:
            print("❌ CUDA is NOT available. Training will be painfully slow or fail.")
            return False
            
    except ImportError as e:
        print(f"❌ CRITICAL: Failed to import PyTorch. Error: {e}")
        return False
    except Exception as e:
        print(f"❌ CRITICAL: Unknown CUDA error. Error: {e}")
        return False
        
    return True

def test_2_dataset_loading():
    print("\n--- TEST 2: Dataset & HDF5 Loading ---")
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Error: DATA_ROOT path does not exist: {DATA_ROOT}")
        return False
        
    try:
        from dobot_dataset import DobotDataset
        dataset = DobotDataset(DATA_ROOT, chunk_size=25)
        print(f"✅ Dataset initialized. Found {len(dataset)} samples.")
        
        # Try to load the first item (Validates HDF5 reading & Normalization)
        sample = dataset[0]
        img_shape = sample['image'].size
        act_shape = sample['actions'].shape
        
        print(f"✅ Successfully loaded sample[0].")
        print(f"   Image Size: {img_shape} (Should be PIL Image)")
        print(f"   Action Shape: {act_shape} (Should be [25, 7])")
        
    except ImportError as e:
        print(f"❌ Import Error: Could not load dobot_dataset.py. {e}")
        return False
    except Exception as e:
        print(f"❌ Dataset Error: Failed to read data. {e}")
        return False
        
    return True

def test_3_model_components():
    print("\n--- TEST 3: Model Imports & Architecture ---")
    try:
        from prismatic.models.action_heads import L1RegressionActionHead
        from transformers import AutoProcessor
        
        print("✅ Imports successful (prismatic, transformers).")
        
        # Test Action Head Instantiation
        # This confirms we can build the specific neural network layer the repo uses
        head = L1RegressionActionHead(
            input_dim=4096, # standard 7b dim
            hidden_dim=4096,
            action_dim=7
        )
        print("✅ L1RegressionActionHead instantiated successfully.")
        
    except ImportError as e:
        print(f"❌ Import Error: Failed to load Repo components. {e}")
        print("   Did you run 'pip install -e .' ?")
        return False
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Running Diagnostics...")
    
    cuda_pass = test_1_cuda_access()
    data_pass = test_2_dataset_loading()
    model_pass = test_3_model_components()
    
    print("\n" + "="*30)
    if cuda_pass and data_pass and model_pass:
        print("🟢 SYSTEM READY. You can proceed to training.")
    else:
        print("🔴 SYSTEM FAILED. Fix the errors above before training.")
    print("="*30)