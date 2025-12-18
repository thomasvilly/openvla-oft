"""
run_dobot_inference.py

Live Inference Client for OpenVLA on Dobot.
Updates:
- Added Gripper Hysteresis (prevents premature suction)
- Added Action Scaling (dampens erratic movement)
"""

import os
import sys
import socket
import struct
import json
import time
import numpy as np
import cv2
import torch
import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Union

sys.path.append(os.getcwd()) 

from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import (
    get_processor,
)

# --- SAFETY & CONTROL ---
SAFETY_LIMITS = {
    "x_min": -150.0, "x_max": 250.0,  
    "y_min": -200.0, "y_max": 200.0, 
    "z_min": -90.0,  "z_max": 150.0, 
    "r_min": -150.0, "r_max": 150.0
}

# 1. Dampening Factor: Multiplies model output.
# Set < 1.0 to move slower/safer. Set > 1.0 if robot is too sluggish.
ACTION_SCALE = 1.0

# 2. Gripper Thresholds (Hysteresis)
GRIPPER_ON_THRESHOLD = 0.5  # Must be very confident to turn ON
GRIPPER_OFF_THRESHOLD = -0.5 # Must be very confident to turn OFF

MAX_STEP_MM = 20.0 

@dataclass
class InferenceConfig:
    robot_ip: str = "192.168.208.1"  
    robot_port: int = 65432
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b+dobot_dataset+b16+lr-2e-05+lora-r32+dropout-0.0--image_aug--1500_chkpt"
    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True  
    num_images_in_input: int = 2
    unnorm_key: str = "dobot_dataset" 
    use_proprio: bool = True
    use_film: bool = False
    instruction: str = "pick up the red block"

class RobotClient:
    """Handles Socket Communication with Windows Server."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[Client] Connecting to Robot at {ip}:{port}...")
        self.sock.connect((ip, port))
        print("[Client] Connected.")

    def _send_json(self, data):
        msg = json.dumps(data).encode('utf-8')
        self.sock.sendall(struct.pack('>I', len(msg)) + msg)

    def _recv_json(self):
        raw_len = self._recv_exact(4)
        if not raw_len: return None
        msg_len = struct.unpack('>I', raw_len)[0]
        data = self._recv_exact(msg_len)
        return json.loads(data.decode('utf-8'))

    def _recv_exact(self, n):
        data = b''
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet: break
            data += packet
        return data

    def get_observation(self):
        self._send_json({"cmd": "GET_IMAGE"})
        len1 = struct.unpack('>I', self._recv_exact(4))[0]
        img1_data = self._recv_exact(len1)
        len2 = struct.unpack('>I', self._recv_exact(4))[0]
        img2_data = self._recv_exact(len2)
        
        img_overhead = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
        img_wrist = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_COLOR)
        
        self._send_json({"cmd": "GET_POSE"})
        resp = self._recv_json()
        pose = np.array(resp['pose'][:4])
        return img_overhead, img_wrist, pose

    def move(self, x, y, z, r):
        cmd = {"cmd": "MOVE", "x": x, "y": y, "z": z, "r": r}
        self._send_json(cmd)
        self._recv_json() 

    def grip(self, state):
        cmd = {"cmd": "GRIP", "state": float(state)}
        self._send_json(cmd)
        self._recv_json() 
        
    def close(self):
        self.sock.close()

def check_safety_clamp(target, current):
    tgt = np.array(target)
    cur = np.array(current)
    
    delta = tgt - cur
    step_dist = np.linalg.norm(delta[:3])
    if step_dist > MAX_STEP_MM:
        scale = MAX_STEP_MM / step_dist
        delta = delta * scale
        tgt = cur + delta

    x, y, z, r = tgt
    x = np.clip(x, SAFETY_LIMITS["x_min"], SAFETY_LIMITS["x_max"])
    y = np.clip(y, SAFETY_LIMITS["y_min"], SAFETY_LIMITS["y_max"])
    z = np.clip(z, SAFETY_LIMITS["z_min"], SAFETY_LIMITS["z_max"])
    r = np.clip(r, SAFETY_LIMITS["r_min"], SAFETY_LIMITS["r_max"])
    return x, y, z, r

@draccus.wrap()
def main(cfg: InferenceConfig):
    set_seed_everywhere(7)
    
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)
    
    try:
        robot = RobotClient(cfg.robot_ip, cfg.robot_port)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print(f"\n>>> READY FOR INFERENCE")
    print(f">>> Config: Action Scale={ACTION_SCALE}, Gripper Hysteresis [{GRIPPER_OFF_THRESHOLD}-{GRIPPER_ON_THRESHOLD}]")
    print(">>> Press Ctrl+C to STOP.\n")

    current_gripper_state = 0 # Track current state for hysteresis

    try:
        step = 0
        while True:
            t0 = time.time()
            
            # 1. Observation
            img_overhead_bgr, img_wrist_bgr, current_pose = robot.get_observation()
            img_overhead = cv2.cvtColor(img_overhead_bgr, cv2.COLOR_BGR2RGB)
            img_wrist = cv2.cvtColor(img_wrist_bgr, cv2.COLOR_BGR2RGB)

            obs = {
                "full_image": img_overhead,
                "wrist_image": img_wrist,
                "state": current_pose if cfg.use_proprio else None
            }
            
            # 2. Inference
            action = get_action(
                cfg=cfg,
                model=model,
                obs=obs,
                task_label=cfg.instruction,
                processor=processor,
                use_film=cfg.use_film
            )
            
            if isinstance(action, list): action = action[0]
            
            # 3. Process Deltas (Scaled)
            delta_xyzr = action[:4]
            raw_gripper = action[-1]
            
            # APPLY SCALING
            delta_xyzr = delta_xyzr * ACTION_SCALE
            
            # 4. Gripper Hysteresis Logic
            # Only switch if confident. Otherwise keep previous state.
            if raw_gripper > GRIPPER_ON_THRESHOLD:
                current_gripper_state = 1
            elif raw_gripper < GRIPPER_OFF_THRESHOLD:
                current_gripper_state = 0
            # else: keep current_gripper_state (ignore weak signals like 0.5)

            # 5. Calculate Target & Safety
            target_pose = current_pose + delta_xyzr
            safe_x, safe_y, safe_z, safe_r = check_safety_clamp(target_pose, current_pose)
            
            print(f"Step {step}:")
            print(f"  Raw Delta: {np.round(action[:4], 2)}")
            print(f"  Gripper:   {raw_gripper:.3f} -> State: {current_gripper_state}")
            print(f"  Target:    {np.round([safe_x, safe_y, safe_z], 1)}")
            
            robot.move(safe_x, safe_y, safe_z, safe_r)
            robot.grip(current_gripper_state)
            
            step += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()