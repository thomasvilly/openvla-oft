import socket
import struct
import json
import numpy as np
import cv2
import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image

# --- CONFIG ---
# "localhost" in WSL often maps to the Windows Host automatically. 
# If this fails, find your Windows IP using `ipconfig` (usually vEthernet WSL).
WINDOWS_IP = '192.168.208.1' 
PORT = 65432

# Paths from your previous step
CHECKPOINT_DIR = "checkpoints/dobot_finetune/openvla-7b+dobot_mixture+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--1000_chkpt/lora_adapter"
STATS_PATH = "checkpoints/dobot_finetune/openvla-7b+dobot_mixture+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug/dataset_statistics.json"
INSTRUCTION = "Pick up the blue block"

class RemoteRobotClient:
    def __init__(self, ip, port):
        print(f"Connecting to Windows Robot Server at {ip}:{port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        print("Connected!")

    def _send_command(self, cmd_dict):
        # Send JSON command
        data = json.dumps(cmd_dict).encode('utf-8')
        self.sock.sendall(struct.pack('>I', len(data)) + data)
        
        # Expecting Image?
        if cmd_dict['cmd'] == "GET_IMAGE":
            # Read Image Size
            raw_len = self._recv_all(4)
            img_len = struct.unpack('>I', raw_len)[0]
            # Read Image Bytes
            img_data = self._recv_all(img_len)
            # Decode
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Expecting JSON Response?
        else:
            raw_len = self._recv_all(4)
            msg_len = struct.unpack('>I', raw_len)[0]
            resp_data = self._recv_all(msg_len)
            return json.loads(resp_data.decode('utf-8'))

    def _recv_all(self, n):
        data = b''
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet: raise Exception("Socket connection broken")
            data += packet
        return data

    def get_image(self):
        frame = self._send_command({"cmd": "GET_IMAGE"})
        # Convert BGR (OpenCV/Windows) to RGB (OpenVLA/Linux)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def move_block(self, x, y, z, r):
        self._send_command({"cmd": "MOVE", "x": x, "y": y, "z": z, "r": r})

    def set_gripper(self, state):
        self._send_command({"cmd": "GRIP", "state": state})

class VLABrain:
    def __init__(self, checkpoint_path, stats_path):
        print(f"Loading Model from {checkpoint_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Base Model
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        base_model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        
        # 2. Load LoRA Adapter
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 3. BRAINWASHING (CPU/Numpy Mode)
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            self.dataset_name = list(stats.keys())[0]
            print(f"Injecting statistics for: {self.dataset_name}")
            
            # Extract raw stats
            raw_stats = stats[self.dataset_name]['action']
            numpy_stats = {}
            
            # Helper: Convert to simple Numpy Array (CPU)
            def to_numpy(key):
                if key in raw_stats:
                    return np.array(raw_stats[key], dtype=np.float32)
                return None

            # Load q01/q99 (Standard) or min/max (Fallback)
            if 'q01' in raw_stats:
                numpy_stats['q01'] = to_numpy('q01')
                numpy_stats['q99'] = to_numpy('q99')
            else:
                numpy_stats['min'] = to_numpy('min')
                numpy_stats['max'] = to_numpy('max')
                
            # Load other keys
            for k in ['mean', 'std', 'mask']:
                if k in raw_stats:
                     numpy_stats[k] = to_numpy(k)

            # --- TARGET THE REAL MODEL ---
            # We inject purely Numpy arrays. The model's internal code handles the rest.
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "model"):
                real_model = self.model.base_model.model
            else:
                real_model = self.model 
            
            real_model.norm_stats = {
                self.dataset_name: {
                    "action": numpy_stats,
                    "proprio": {} 
                }
            }
            print(f"Model registry updated on {type(real_model).__name__} (using Numpy).")

    def predict_action(self, image_numpy, instruction):
        image_pil = Image.fromarray(image_numpy)
        inputs = self.processor(text=instruction, images=image_pil, return_tensors="pt").to(self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            action_real = self.model.predict_action(
                **inputs, 
                unnorm_key=self.dataset_name, 
                do_sample=False
            ) 
            
            # If the result is a Tensor, move to CPU numpy. 
            # If it's already numpy (because the model did the math on CPU), this line is safe.
            if isinstance(action_real, torch.Tensor):
                action_real = action_real.cpu().numpy()[0]
            
        return action_real

def main():
    try:
        # 1. Connect Components
        robot = RemoteRobotClient(WINDOWS_IP, PORT)
        brain = VLABrain(CHECKPOINT_DIR, STATS_PATH)
        
        print(f"\n>>> READY. Task: {INSTRUCTION}")
        input("Press Enter to Start...")
        
        # 2. Loop
        while True:
            loop_start_time = time.time() 

            # A. Get Image from Windows
            img_fetch_start = time.time()
            img = robot.get_image()
            img_fetch_time = time.time() - img_fetch_start

            # B. Get Action from WSL
            pred_start = time.time()
            action = brain.predict_action(img, INSTRUCTION)
            pred_time = time.time() - pred_start
            
            x, y, z, r, pitch, yaw, grip = action
            
            # C. Send Move to Windows
            send_start = time.time()
            robot.move_block(x, y, z, r)
            robot.set_gripper(grip)
            send_time = time.time() - send_start

            loop_end_time = time.time()
            total_time = loop_end_time - loop_start_time

            # Print all the timings
            print(f"Act: [{x:.1f}, {y:.1f}, {z:.1f}] Grip: {grip:.2f}")
            print(f"--- Timing (s) --- Total: {total_time:.3f} | Img Fetch: {img_fetch_time:.3f} | Prediction: {pred_time:.3f} | Send: {send_time:.3f}")
            
            # We need to explicitly pause the loop to wait for the physical robot to move 
            # and for the camera to capture the new, stable position.
            # This is a temporary, safe pause for diagnosis.
            time.sleep(5.0)
            
    except KeyboardInterrupt:
        print("Stopping...")

if __name__ == "__main__":
    main()