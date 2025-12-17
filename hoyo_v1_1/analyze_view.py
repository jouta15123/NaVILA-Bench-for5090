
import json
import pickle
import os
import glob
import numpy as np

data_dir = "data"
json_files = glob.glob(os.path.join(data_dir, "*.json"))

print(f"Checking {len(json_files)} files for coordinate variance (View Analysis)...")

x_ranges = []
y_ranges = []
views = []
initials = []

for json_path in json_files:
    file_id = os.path.basename(json_path).replace(".json", "")
    pickle_path = os.path.join(data_dir, f"{file_id}.pickle")
    
    if not os.path.exists(pickle_path):
        continue
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
        views.append(meta.get('view', 'unknown'))
        initials.append(meta.get('phase', {}).get('initial', 'unknown'))
        
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            # data: (T, J, 2)
            # Check range of motion of the Root/Center to determining global movement direction
            # Assuming mean of all joints approximates center roughly
            center_traj = data.mean(axis=1) # (T, 2)
            
            # Range = Max - Min
            x_range = center_traj[:, 1].max() - center_traj[:, 1].min() # index 1 might be X based on common.py swap?
            y_range = center_traj[:, 0].max() - center_traj[:, 0].min() # index 0 might be Y?
            
            # Actually common.py loads as [y, x] originally, then swaps to [x, y].
            # Here we are loading raw pickle. raw pickle is [y, x] (image coords).
            # So index 1 is X (width), index 0 is Y (height).
            
            x_ranges.append(x_range)
            y_ranges.append(y_range)

x_mean = np.mean(x_ranges)
y_mean = np.mean(y_ranges)

print(f"Mean X Range (Horizontal): {x_mean:.2f}")
print(f"Mean Y Range (Vertical):   {y_mean:.2f}")

from collections import Counter
print("Views:", Counter(views))
print("Initials:", Counter(initials))

# Verify if X range is significantly larger than Y range (implies walking across screen)
if x_mean > y_mean * 2:
    print("CONCLUSION: Motion is primarily HORIZONTAL (Profile/Side view). Stride should be visible.")
else:
    print("CONCLUSION: Motion is not strictly horizontal. Stride might be hard to see.")
