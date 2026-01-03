import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "/mnt/d/DOBOT/rlds_dataset_folder" 
DATASET_NAME = "dobot_dataset"

def visualize_rlds():
    print(f"Loading dataset {DATASET_NAME} from {DATA_DIR}...")
    # Note: version='1.1.0' if you updated the version string in the builder
    dataset = tfds.load(DATASET_NAME, data_dir=DATA_DIR, split='train')

    for episode in dataset:
        file_path = episode['episode_metadata']['file_path'].numpy().decode('utf-8')
        print(f"Viewing Episode: {file_path}")
        
        for step in episode['steps']:
            # Pull both images
            img_top = step['observation']['image'].numpy()

            img_top = img_top[:, :, ::-1]
            
            state = step['observation']['state'].numpy()
            action = step['action'].numpy() # Temporal Delta
            instr = step['language_instruction'].numpy().decode('utf-8')
            
            # Convert RGB to BGR and Resize
            top_view = cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR)
            top_view = cv2.resize(top_view, (500, 500))
            
            # Draw Delta Arrow on Top View
            cx, cy = 250, 250
            scale = 10.0
            ex = int(cx + (action[1] * scale)) 
            ey = int(cy + (action[0] * scale))
            
            if np.linalg.norm(action[:2]) > 0.05:
                cv2.arrowedLine(top_view, (cx, cy), (ex, ey), (0, 0, 255), 2)
            
            # Black overlay for text
            overlay = np.zeros((150, top_view.shape[1], 3), dtype=np.uint8)
            y0, dy = 30, 25
            info = [
                f"Instruction: {instr}",
                f"State (Abs): X:{state[0]:.1f} Y:{state[1]:.1f} Z:{state[2]:.1f}",
                f"Action (Delta): dX:{action[0]:+.3f} dY:{action[1]:+.3f} dZ:{action[2]:+.3f}",
                f"Grip: {action[-1]:.0f} | Is_First: {step['is_first'].numpy()} | Is_Last: {step['is_last'].numpy()}"
            ]
            for i, line in enumerate(info):
                cv2.putText(overlay, line, (20, y0 + i*dy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            final_display = np.vstack((top_view, overlay))
            cv2.imshow("RDLS Dual-Camera Verification", final_display)
            
            key = cv2.waitKey(0)
            if key == ord('q'): return
            if key == ord('n'): break 

if __name__ == "__main__":
    visualize_rlds()