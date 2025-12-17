
import json
import pickle
import os
import glob
import numpy as np
from collections import defaultdict

data_dir = "data"
json_files = glob.glob(os.path.join(data_dir, "*.json"))

stats = defaultdict(list)

for json_path in json_files:
    file_id = os.path.basename(json_path).replace(".json", "")
    pickle_path = os.path.join(data_dir, f"{file_id}.pickle")
    
    if not os.path.exists(pickle_path):
        continue
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
        label = meta.get('annotation', {}).get('instruction', 'unknown')
        
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f) # (T, 14, 2)
        if isinstance(data, np.ndarray):
            stats[label].append(data.shape[0])

print(f"\n{'Label':<15} | {'Min':<5} | {'Max':<5} | {'Mean':<5}")
print("-" * 40)
min_of_all = 9999
for label, lengths in stats.items():
    if lengths:
        mn = np.min(lengths)
        mx = np.max(lengths)
        avg = np.mean(lengths)
        print(f"{label:<15} | {mn:<5} | {mx:<5} | {avg:<5.1f}")
        if mn < min_of_all:
            min_of_all = mn

print(f"\nAbsolute Minimum Length in Dataset: {min_of_all}")
