# Source - https://stackoverflow.com/a
# Posted by Astariul, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-05, License - CC BY-SA 4.0

import tensorflow as tf
import json
from google.protobuf.json_format import MessageToJson
import base64
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

dataset = tf.data.TFRecordDataset(r"C:\Users\tevillen\Downloads\liber_o10-train.tfrecord-00000-of-00032")
output_filename = "robot_episode_view.gif"
figure_filename = "robot_trajectory.png"
save = 1

for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d.numpy())
    m = json.loads(MessageToJson(ex))
    print(m['features']['feature'].keys())
    # print(m['features']['feature']['steps/discount'])
    # print(m['features']['feature']['steps/action'])
    # print(m['features']['feature']['steps/reward'])
    # print(m['features']['feature']['episode_metadata/file_path'])
    instructions = m['features']['feature']['steps/language_instruction']['bytesList']['value']
    instruc1 = base64.b64decode(instructions[0]).decode('utf-8')
    print(f"Instruction: {instruc1}")
    s = 0
    for instruc in instructions:
        encoded_str = instruc
        decoded_bytes = base64.b64decode(encoded_str).decode('utf-8')
        if decoded_bytes != instruc1:
            print(decoded_bytes)
        else:
            s+=1
    print(f"{s} total steps")
    # print(len(m['features']['feature']['steps/action']['floatList']['value']))
    # print(len(m['features']['feature']['steps/action']['floatList']['value'])/s)

    if save:
        image_bytes_list = m['features']['feature']['steps/observation/image']['bytesList']['value']
        
        frames = []
        print(f"Found {len(image_bytes_list)} frames. Processing...")

        for img_bytes in image_bytes_list:
            # 1. Decode bytes to Image
            img_bytes = base64.b64decode(img_bytes)
            image = Image.open(io.BytesIO(img_bytes))
            # 2. Append to list
            frames.append(image)

        # Save as GIF
        # duration=50 means 50ms per frame (approx 20fps)
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        
        print(f"GIF saved to {output_filename}")

    data = m['features']['feature']['steps/action']['floatList']['value']
    dofs = [data[i::7] for i in range(7)]
    x, y, z, r, p, w, g = dofs[0], dofs[1], dofs[2], dofs[3], dofs[4], dofs[5], dofs[6]
    
    # Create 3 stacked subplots sharing the same Time (X) axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    fig.suptitle('Robot Action Trajectory', fontsize=16)

    # Plot 1: Translation (Deltas for X, Y, Z)
    ax1.plot(x, label='Delta X', color='#FF5733', linewidth=1.5)
    ax1.plot(y, label='Delta Y', color='#33FF57', linewidth=1.5)
    ax1.plot(z, label='Delta Z', color='#3357FF', linewidth=1.5)
    ax1.set_ylabel('Translation', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('End-Effector Translation', fontsize=10)

    # Plot 2: Rotation (Deltas for Roll, Pitch, Yaw)
    ax2.plot(r, label='Delta Roll', color='cyan', linewidth=1.5)
    ax2.plot(p, label='Delta Pitch', color='magenta', linewidth=1.5)
    ax2.plot(w, label='Delta Yaw', color='orange', linewidth=1.5)
    ax2.set_ylabel('Rotation (Radians)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('End-Effector Rotation', fontsize=10)

    # Plot 3: Gripper
    ax3.plot(g, label='Gripper', color='black', linewidth=2)
    ax3.set_ylabel('State', fontsize=12)
    ax3.set_xlabel('Timestep', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Gripper Action', fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig(figure_filename, dpi=300)
        print(f"Figure saved to {figure_filename}")
    else:
        plt.show()
    break