"""
run_dobot_inference.py

Live Inference Client for OpenVLA on Dobot.
FIXES:
- Patches 'mask' in stats (Fixes 7D vs 5D crash)
- Handles 7D Proprioception Padding
- Tuned Action Scale (25.0)
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


# 1. Action Scale (Gain)
ACTION_SCALE = 1.0

# 2. Gripper Hysteresis
GRIPPER_ON_THRESHOLD = 0.5 
GRIPPER_OFF_THRESHOLD = -0.5

MAX_STEP_MM = 20.0 

@dataclass
class InferenceConfig:
    robot_ip: str = "192.168.208.1"  
    robot_port: int = 65432
    
    # Path to your 5000-step checkpoint
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
    """Handles Socket Communication with Auto-Reconnect."""
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.connect()

    def connect(self):
        """Attempts to connect to the robot server."""
        try:
            # Always close old socket before creating a new one
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0) # 2-second timeout to detect hangs
            
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
            # Pack length (4 bytes) + JSON message
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
            # Hard timeout to prevent infinite blocking
            if time.time() - start_t > 2.0: 
                break
        return data

    def get_observation(self):
        # Retry loop: Try 3 times to get an image, then fail gracefully
        for _ in range(3):
            # Only try to receive if send succeeded
            if self._send_json({"cmd": "GET_IMAGE"}):
                try:
                    # Manually handle the raw bytes for images here
                    # (We can't use _recv_json because images are raw bytes, not JSON)
                    len1_data = self._recv_exact(4)
                    if not len1_data: raise socket.error("No header")
                    len1 = struct.unpack('>I', len1_data)[0]
                    img1_data = self._recv_exact(len1)
                    
                    len2_data = self._recv_exact(4)
                    if not len2_data: raise socket.error("No header 2")
                    len2 = struct.unpack('>I', len2_data)[0]
                    img2_data = self._recv_exact(len2)
                    
                    img_overhead = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Get Pose
                    self._send_json({"cmd": "GET_POSE"})
                    resp = self._recv_json()
                    if resp and 'pose' in resp:
                        pose = np.array(resp['pose'][:4])
                        return img_overhead, pose
                except Exception as e:
                    print(f"[Info] Observation failed ({e}). Retrying...")
            
            # If we failed, reconnect and wait a bit
            self.connect()
            time.sleep(0.5)
            
        # If all retries fail, return zeros to prevent the whole script from crashing
        print("[Error] Could not get observation after retries.")
        dummy_img = np.zeros((240,320,3), dtype=np.uint8)
        dummy_pose = np.array([200.0, 0.0, 0.0, 0.0]) # Safe home pose
        return dummy_img, dummy_img, dummy_pose

    def move(self, x, y, z, r):
        cmd = {"cmd": "MOVE", "x": x, "y": y, "z": z, "r": r}
        self._send_json(cmd)
        self._recv_json() # Wait for ACK

    def grip(self, state):
        cmd = {"cmd": "GRIP", "state": float(state)}
        self._send_json(cmd)
        self._recv_json() # Wait for ACK
        
    def close(self):
        if self.sock: self.sock.close()

def check_safety_clamp(target, current):
    x, y, z, r = target
    
    # 1. Clamp Z and R (Standard Box Limits)
    # Z: -50 is usually table level. 150 is safe height.
    z = np.clip(z, -50.0, 150.0) 
    r = np.clip(r, -140.0, 140.0)

    # 2. Clamp Radius (The "Arc" Limit)
    # Dobot Magician Max Reach is ~320mm. We limit to 310mm for safety.
    # Dobot Min Reach (Inner Radius) is ~140mm (can't hit its own base).
    radius = np.sqrt(x**2 + y**2)
    max_radius = 310.0
    min_radius = 140.0
    
    if radius > max_radius:
        # Pull back towards base
        scale = max_radius / radius
        x = x * scale
        y = y * scale
    elif radius < min_radius:
        # Push out away from base
        scale = min_radius / radius
        x = x * scale
        y = y * scale

    # 3. Step Size Clamp (Prevent Teleporting)
    # Re-calculate delta from CURRENT valid pose to TARGET valid pose
    cur_x, cur_y, cur_z, _ = current
    dist_sq = (x - cur_x)**2 + (y - cur_y)**2 + (z - cur_z)**2
    max_step = MAX_STEP_MM**2
    
    if dist_sq > max_step:
        scale = np.sqrt(max_step / dist_sq)
        x = cur_x + (x - cur_x) * scale
        y = cur_y + (y - cur_y) * scale
        z = cur_z + (z - cur_z) * scale

    return x, y, z, r

@draccus.wrap()
def main(cfg: InferenceConfig):
    set_seed_everywhere(7)
    
    print(f"Loading model from {cfg.pretrained_checkpoint}...")
    model = get_model(cfg)
    processor = get_processor(cfg)
    
    # --- HOTFIX: Patch Model Statistics (7D -> 5D) ---
    # if cfg.unnorm_key in model.norm_stats:
    #     print(f"[Patch] Checking statistics dimensions for {cfg.unnorm_key}...")
    #     action_stats = model.norm_stats[cfg.unnorm_key]["action"]
        
    #     # ADDED "mask" TO THIS LIST
    #     for key in ["q01", "q99", "min", "max", "mean", "std", "mask"]:
    #         if key in action_stats:
    #             stat_arr = action_stats[key]
    #             if len(stat_arr) == 7:
    #                 print(f"  -> Slicing {key} from 7 to 5 dimensions.")
    #                 action_stats[key] = stat_arr[:5]
    # -------------------------------------------------

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
            # SAFETY CHECK: If robot disconnected, wait for it
            if robot.sock is None:
                print("Waiting for robot connection...")
                robot.connect()
                time.sleep(1.0)
                continue

            # ... rest of loop ...
            t0 = time.time()
            
            # 1. Observation
            img_overhead_bgr, current_pose_4d = robot.get_observation()
            
            img_overhead = cv2.cvtColor(img_overhead_bgr, cv2.COLOR_BGR2RGB)

            # 2. Proprioception Construction (Pad to 7D)
            if cfg.use_proprio:
                proprio_state = np.array([
                    current_pose_4d[0], # X
                    current_pose_4d[1], # Y
                    current_pose_4d[2], # Z
                    current_pose_4d[3], # Roll
                    float(current_gripper_state) # Grip
                ])
            else:
                proprio_state = None

            obs = {
                "full_image": img_overhead,
                "state": proprio_state
            }
            
            # 3. Inference
            action = get_action(
                cfg=cfg,
                model=model,
                obs=obs,
                task_label=cfg.instruction,
                processor=processor,
                use_film=cfg.use_film
            )
            
            # 4. Handle Chunking
            if isinstance(action, list):
                action = action[0]
            elif isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
            
            # 5. Process Deltas
            delta_xyzr = action[:4]
            raw_gripper = action[-1]
            
            delta_xyzr = delta_xyzr * ACTION_SCALE
            
            # 6. Gripper Logic
            if raw_gripper > GRIPPER_ON_THRESHOLD:
                current_gripper_state = 1
            elif raw_gripper < GRIPPER_OFF_THRESHOLD:
                current_gripper_state = 0

            # 7. Safety & Execute
            target_pose = current_pose_4d + delta_xyzr
            safe_x, safe_y, safe_z, safe_r = check_safety_clamp(target_pose, current_pose_4d)
            
            dt = time.time() - t0
            print(f"Step {step} ({dt:.3f}s):")
            print(f"  Delta: {np.round(delta_xyzr, 2)}")
            print(f"  Grip:  {raw_gripper:.2f} -> {current_gripper_state}")
            
            robot.move(safe_x, safe_y, safe_z, safe_r)
            robot.grip(current_gripper_state)
            
            step += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()