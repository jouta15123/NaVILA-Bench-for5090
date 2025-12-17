
import json
import pickle
import os
import numpy as np

data_dir = "data"
files_to_inspect = [0, 1, 10]

for idx in files_to_inspect:
    json_path = os.path.join(data_dir, f"{idx}.json")
    pickle_path = os.path.join(data_dir, f"{idx}.pickle")
    
    print(f"--- Process {idx} ---")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta = json.load(f)
            # print(f"JSON content: {json.dumps(meta, indent=2)}") 
            # concise output
            print(f"JSON Instruction: {meta.get('annotation', {}).get('instruction')}")
    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, np.ndarray):
                print(f"Pickle data is numpy array. Shape: {data.shape}, Dtype: {data.dtype}")
                print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            else:
                print(f"Pickle data type: {type(data)}")
                if hasattr(data, 'keys'):
                     print(f"Keys: {data.keys()}")

