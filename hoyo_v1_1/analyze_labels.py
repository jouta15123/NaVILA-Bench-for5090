
import json
import pickle
import os
import glob
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

data_dir = "data"
json_files = glob.glob(os.path.join(data_dir, "*.json"))

label_stats = defaultdict(lambda: {'count': 0, 'lengths': [], 'velocities': []})

print(f"Found {len(json_files)} data points.")

for json_path in json_files:
    file_id = os.path.basename(json_path).replace(".json", "")
    pickle_path = os.path.join(data_dir, f"{file_id}.pickle")
    
    if not os.path.exists(pickle_path):
        continue
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
        try:
            label = meta['annotation']['instruction']
        except KeyError:
            label = "unknown"
        
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            # data shape (T, J, D)
            # Calculate velocity (magnitude of difference between frames), averaged over joints and time
            # shape: (T, 14, 2) -> diff (T-1, 14, 2) -> norm (T-1, 14) -> mean
            if data.shape[0] > 1:
                velocity = np.linalg.norm(np.diff(data, axis=0), axis=2).mean()
                label_stats[label]['velocities'].append(velocity)
            
            label_stats[label]['lengths'].append(data.shape[0])
            label_stats[label]['count'] += 1

print(f"\n{'Label':<15} | {'Count':<5} | {'Mean Length':<11} | {'Mean Velocity':<13}")
print("-" * 55)

for label, stats in label_stats.items():
    if stats['count'] > 0:
        mean_len = np.mean(stats['lengths'])
        mean_vel = np.mean(stats['velocities']) if stats['velocities'] else 0.0
        print(f"{label:<15} | {stats['count']:<5} | {mean_len:<11.1f} | {mean_vel:<13.4f}")

