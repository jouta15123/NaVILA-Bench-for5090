
import json
import os
from collections import Counter

def main():
    root_dir = "/home/jouta/NaVILA-Bench/hoyo_v1_1/data"
    instruction_counts = Counter()
    
    files = [f for f in os.listdir(root_dir) if f.endswith(".json")]
    print(f"Total files: {len(files)}")
    
    for fname in files:
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        inst = data.get("annotation", {}).get("instruction", "UNKNOWN")
        instruction_counts[inst] += 1
            
    print("\nInstruction Counts:")
    for k, v in instruction_counts.most_common():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
