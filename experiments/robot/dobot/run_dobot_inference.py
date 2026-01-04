"""
run_dobot_inference.py

Live Inference Client for OpenVLA on Dobot.
UPDATED FIXES:
- Uses get_vla_action (Connects Brain to Hands)
- Manually loads Action Head & Proprio Projector (Fixes "Dumb Model" issue)
- Disables Center Crop (Fixes "Zoom" issue)
- Action Scale Tuned
"""

import os
import sys
import socket
import struct
import json
import time
import glob
import numpy as np
import cv2
import torch
import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Union

sys.path.append(os.getcwd()) 

# --- IMPORTS UPDATED ---
from experiments.robot.robot_utils import (
    get_model,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action  # <--- The correct inference function
)

# 1. Action Scale (Gain)
# Based on your findings, we need a high scale because the regression is dampened.
ACTION_SCALE = 25.0

# 2. Gripper Hysteresis
GRIPPER_ON_THRESHOLD = 0.3
GRIPPER_OFF_THRESHOLD = -0.3

MAX_STEP_MM = 20.0 

@dataclass
class InferenceConfig:
    robot_ip: str = "192.168.208.1"  
    robot_port: int = 65432
    
    # Path to your working checkpoint
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r64+dropout-0.0--2500_chkpt"
    
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # --- CRITICAL FIXES ---
    center_crop: bool = False  # MUST BE FALSE to match training
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    # ----------------------

    num_images_in_input: int = 1
    unnorm_key: str = "dobot_dataset" 
    use_proprio: bool = True
    use_film: bool = False
    
    instruction: str = "pick up the red block"

class RobotClient:
    """Handles Socket Communication with Auto-Reconnect."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        """Attempts to connect to the robot server."""
        try:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            print(f"[Client] Connecting to Robot at {self.ip}:{self.port}...")
            self.sock.connect((self.ip, self.port))
            print("[Client] Connected.")
            return True
        except Exception as e:
            print(f"[Error] Connection failed: {e}")
            return False

    def _send_json(self, data):
        try:
            if self.sock is None:
                raise socket.error("No socket")
                
            msg = json.dumps(data).encode('utf-8')
            self.sock.sendall(struct.pack('>I', len(msg)) + msg)
            return True
        except (socket.error, BrokenPipeError, AttributeError):
            print("[Warn] Socket broken during SEND. Reconnecting...")
            self.connect()
            return False

    def _recv_json(self):
        try:
            if self.sock is None:
                raise socket.error("No socket")

            raw_len = self._recv_exact(4)
            if not raw_len: return None
            
            msg_len = struct.unpack('>I', raw_len)[0]
            data = self._recv_exact(msg_len)
            return json.loads(data.decode('utf-8'))
        except (socket.error, socket.timeout, AttributeError):
            print("[Warn] Socket error during RECV. Reconnecting...")
            self.connect()
            return None

    def _recv_exact(self, n):
        data = b''
        start_t = time.time()
        while len(data) < n:
            try:
                packet = self.sock.recv(n - len(data))
                if not packet: break
                data += packet
            except socket.timeout:
                break
            if time.time() - start_t > 2.0: 
                break
        return data

    def get_observation(self):
        for _ in range(3):
            if self._send_json({"cmd": "GET_IMAGE"}):
                try:
                    len1_data = self._recv_exact(4)
                    if not len1_data: raise socket.error("No header")
                    len1 = struct.unpack('>I', len1_data)[0]
                    img1_data = self._recv_exact(len1)
                    
                    # Consume second image if protocol sends it (even if unused)
                    len2_data = self._recv_exact(4)
                    if len2_data:
                        len2 = struct.unpack('>I', len2_data)[0]
                        _ = self._recv_exact(len2)
                    
                    img_overhead = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    self._send_json({"cmd": "GET_POSE"})
                    resp = self._recv_json()
                    if resp and 'pose' in resp:
                        pose = np.array(resp['pose'][:4])
                        return img_overhead, pose
                except Exception as e:
                    print(f"[Info] Observation failed ({e}). Retrying...")
            
            self.connect()
            time.sleep(0.5)
            
        print("[Error] Could not get observation after retries.")
        dummy_img = np.zeros((240,320,3), dtype=np.uint8)
        dummy_pose = np.array([200.0, 0.0, 0.0, 0.0])
        return dummy_img, dummy_pose

    def move(self, x, y, z):
        cmd = {"cmd": "MOVE", "x": x, "y": y, "z": z}
        self._send_json(cmd)
        self._recv_json()

    def grip(self, state):
        cmd = {"cmd": "GRIP", "state": float(state)}
        self._send_json(cmd)
        self._recv_json()
        
    def close(self):
        if self.sock: self.sock.close()

def check_safety_clamp(target, current):
    x, y, z = target
    z = np.clip(z, -50.0, 150.0) 

    radius = np.sqrt(x**2 + y**2)
    max_radius = 310.0
    min_radius = 140.0
    
    if radius > max_radius:
        scale = max_radius / radius
        x *= scale
        y *= scale
    elif radius < min_radius:
        scale = min_radius / radius
        x *= scale
        y *= scale

    cur_x, cur_y, cur_z = current
    dist_sq = (x - cur_x)**2 + (y - cur_y)**2 + (z - cur_z)**2
    max_step = MAX_STEP_MM**2
    
    if dist_sq > max_step:
        scale = np.sqrt(max_step / dist_sq)
        x = cur_x + (x - cur_x) * scale
        y = cur_y + (y - cur_y) * scale
        z = cur_z + (z - cur_z) * scale

    return x, y, z

@draccus.wrap()
def main(cfg: InferenceConfig):
    set_seed_everywhere(7)
    
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    
    # --- COMPONENT LOADER (From your working Eval script) ---
    def load_component_weights(component, path):
        print(f"    > Loading weights from: {path}")
        state_dict = torch.load(path, map_location=model.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        try:
            component.load_state_dict(new_state_dict)
            print("    [+] Weights loaded successfully!")
        except Exception as e:
            print(f"    [!] Error loading weights: {e}")

    # 1. Setup & Load Action Head
    print("\n[LOADER] Setting up Action Head...")
    model.action_head = get_action_head(cfg, llm_dim=4096) 
    model.action_head.to(model.device)
    model.action_head.dtype = model.dtype
    
    ah_files = glob.glob(os.path.join(cfg.pretrained_checkpoint, "action_head*.pt"))
    if ah_files:
        load_component_weights(model.action_head, ah_files[0])
    else:
        print("[!] WARNING: No Action Head weights found!")

    # 2. Setup & Load Proprio Projector
    if cfg.use_proprio:
        print("\n[LOADER] Setting up Proprio Projector...")
        model.proprio_projector = get_proprio_projector(cfg, llm_dim=4096, proprio_dim=4)
        model.proprio_projector.to(model.device)
        model.proprio_projector.dtype = model.dtype
        
        pp_files = glob.glob(os.path.join(cfg.pretrained_checkpoint, "proprio_projector*.pt"))
        if pp_files:
            load_component_weights(model.proprio_projector, pp_files[0])
    # -----------------------------------------------------

    processor = get_processor(cfg)

    try:
        robot = RobotClient(cfg.robot_ip, cfg.robot_port)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print(f"\n>>> READY FOR INFERENCE")
    print(f">>> Action Scale: {ACTION_SCALE}")
    print(">>> Press Ctrl+C to STOP.\n")

    current_gripper_state = 0 

    try:
        step = 0
        while True:
            if robot.sock is None:
                print("Waiting for robot connection...")
                robot.connect()
                time.sleep(1.0)
                continue

            t0 = time.time()
            
            # 1. Observation
            img_overhead_bgr, current_pose_4d = robot.get_observation()
            img_overhead = cv2.cvtColor(img_overhead_bgr, cv2.COLOR_BGR2RGB)

            # 2. Proprioception (Pad to 7D logic if needed, but 4D works if stats patched)
            if cfg.use_proprio:
                # We feed the 4D state: X, Y, Z, Grip
                proprio_state = np.array([
                    current_pose_4d[0],
                    current_pose_4d[1],
                    current_pose_4d[2],
                    float(current_gripper_state)
                ])
            else:
                proprio_state = None

            obs = {
                "full_image": img_overhead,
                "state": proprio_state
            }
            
            # 3. Inference (UPDATED to get_vla_action)
            action = get_vla_action(
                cfg=cfg,
                vla=model,
                obs=obs,
                task_label=cfg.instruction,
                processor=processor,
                action_head=model.action_head,
                proprio_projector=model.proprio_projector if cfg.use_proprio else None,
                use_film=cfg.use_film
            )
            
            # 4. Handle Chunking
            if isinstance(action, list):
                action = action[0]
            elif isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
            
            # 5. Process Deltas
            delta_xyz = action[:3]
            raw_gripper = action[-1]
            
            delta_xyz = delta_xyz * ACTION_SCALE
            
            # 6. Gripper Logic
            if raw_gripper > GRIPPER_ON_THRESHOLD:
                current_gripper_state = 1
            elif raw_gripper < GRIPPER_OFF_THRESHOLD:
                current_gripper_state = 0

            # 7. Safety & Execute
            target_pose = current_pose_4d + delta_xyz
            safe_x, safe_y, safe_z = check_safety_clamp(target_pose, current_pose_4d)
            
            dt = time.time() - t0
            print(f"Step {step} ({dt:.3f}s):")
            print(f"  Delta: {np.round(delta_xyz, 2)}")
            print(f"  Grip:  {raw_gripper:.2f} -> {current_gripper_state}")
            
            robot.move(safe_x, safe_y, safe_z)
            robot.grip(current_gripper_state)
            
            step += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()