import os
import argparse
import sys
import numpy as np
import torch
import imageio
import base64
import json
import socket
from PIL import Image
from pathlib import Path

# Add MotionCLIP to path
REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONCLIP_ROOT = REPO_ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

# Import visualization tool
from src.visualize.anim import plot_3d_motion

def generate_skeleton_gif(motion_data, save_path, fps=30):
    """
    Generate GIF from motion data (T, J, C).
    Input:
        motion_data: numpy array (T, J, 3) or (T, J, 2)
        save_path: output path
    """
    T, J, C = motion_data.shape
    
    # If 2D, add z=0
    if C == 2:
        zeros = np.zeros((T, J, 1))
        motion_data = np.concatenate([motion_data, zeros], axis=2)
    
    # MotionCLIP visualizer expects (J, 3, T)
    motion_for_vis = motion_data.transpose(1, 2, 0) # (J, 3, T)
    
    # Using dummy params required by plot_3d_motion
    params = {
        "pose_rep": "xyz",
        "appearance_mode": "motionclip"
    }
    
    # Generate frames
    print(f"Generating GIF to {save_path}...")
    plot_3d_motion(motion_for_vis, T, save_path, params, title="Motion", interval=1000/fps)
    return save_path

def query_vlm(image_path, prompt, host="localhost", port=54321):
    """
    Send GIF (as list of images) to VLM server
    """
    print(f"Reading {image_path}...")
    # Read GIF
    frames = imageio.mimread(image_path, memtest=False)
    
    # Sample 8 frames for VLM
    indices = np.linspace(0, len(frames)-1, 8, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    encoded_images = []
    for frame in sampled_frames:
        pil_img = Image.fromarray(frame)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())
    
    request_data = {
        'images': encoded_images,
        'query': prompt
    }
    
    print(f"Sending request to {host}:{port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data_bytes = json.dumps(request_data).encode()
        s.sendall(len(data_bytes).to_bytes(8, 'big'))
        s.sendall(data_bytes)
        
        # Receive response
        size_data = s.recv(8)
        size = int.from_bytes(size_data, 'big')
        
        response_data = b''
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet
            
    return json.loads(response_data.decode())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, help="Path to .npy or .pkl motion file")
    parser.add_argument("--vlm_port", type=int, default=54321)
    parser.add_argument("--output_gif", type=str, default="test_motion.gif")
    args = parser.parse_args()
    
    # 1. Load Motion
    if args.motion_file:
        data = np.load(args.motion_file, allow_pickle=True)
        # Assume data format based on file extension/content
        # TODO: Adapt to specific file format
        motion = data # Placeholder
    else:
        # Generate dummy motion (walking-ish)
        print("Generating dummy motion...")
        T = 60
        J = 14 # Hoyo joints
        t = np.linspace(0, 4*np.pi, T)
        motion = np.zeros((T, J, 3))
        # Simple swaying
        motion[:, :, 0] = np.sin(t)[:, None] * 0.2
        motion[:, :, 1] = np.cos(t)[:, None] * 0.1
        
    # 2. Visualize
    generate_skeleton_gif(motion, args.output_gif)
    
    # 3. Query VLM
    import io
    prompt = "Describe the motion style of this person using Japanese onomatopoeia (e.g., 'teku-teku', 'fura-fura'). List top 3 candidates."
    
    try:
        response = query_vlm(args.output_gif, prompt, port=args.vlm_port)
        print("\nVLM Response:", response)
    except Exception as e:
        print(f"\nVLM Query Failed: {e}")
        print("Make sure vlm_server.py is running!")

if __name__ == "__main__":
    main()

