
import json
import os
from collections import Counter

def load_and_count(root_dir):
    freestyle_counts = Counter()
    
    files = [f for f in os.listdir(root_dir) if f.endswith(".json")]
    print(f"Found {len(files)} json files.")
    
    for fname in files:
        path = os.path.join(root_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ann = data.get("annotation", {})
        freestyle = ann.get("freestyle", []) or []
        for w in freestyle:
            freestyle_counts[w] += 1
            
    return freestyle_counts

def main():
    root_dir = "/home/jouta/NaVILA-Bench/hoyo_v1_1/data"
    counts = load_and_count(root_dir)
    
    # 10 instruction words to exclude (optional, but paper implies "general words" might include these or not, 
    # but for "Evaluation" we usually exclude Training words. 
    # However, the user asked for the list "from the paper" which had 27 words.
    # The paper says: "Among the 27 words... the most frequent was 'tekuteku'". 
    # 'tekuteku' IS a training word. So the 27 words INCLUDE the training words.
    # I will list all words with count >= 3.
    
    print("\nWords with count >= 3:")
    filtered = {w: c for w, c in counts.items() if c >= 3}
    sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    for w, c in sorted_words:
        print(f"| {w} | {c} |")
        
    print(f"\nTotal words with count >= 3: {len(sorted_words)}")

if __name__ == "__main__":
    main()
