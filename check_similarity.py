
import sys
import os
import torch
import torch.nn.functional as F
from pathlib import Path

# Add root to sys.path
root_dir = Path("/home/jouta/NaVILA-Bench")
sys.path.insert(0, str(root_dir))

# Try importing
try:
    from hoyo_v1_1.models.common import encode_semantics_sarashina
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    device = "cpu" # sufficient for inference check
    
    # Pairs to check (Learned vs Target/Novel)
    pairs = [
        # Pattern A (General/Phonological variants)
        ("てくてく", "とことこ"),
        ("てくてく", "とんとん"),
        ("のろのろ", "とろとろ"),
        ("のしのし", "のそのそ"),
        ("よろよろ", "ふらふら"),
        ("すたすた", "すらすら"),
        
        # Cross-category checks (should be lower?)
        ("てくてく", "のろのろ"),
        ("すたすた", "よろよろ"),
        
        # Pattern B (Novel/Nonsense)
        ("てくてく", "ぎそぎそ"), 
        ("よろよろ", "もよもよ"),
        
        # Random/Other
        ("通常", "てくてく"),
        ("通常", "とことこ"),
    ]
    
    unique_words = list(set([w for p in pairs for w in p]))
    unique_words.sort()
    
    print("Loading Sarashina BERT and encoding words...")
    try:
        embeddings = encode_semantics_sarashina(unique_words, device)
    except Exception as e:
        print(f"Error executing encoding: {e}")
        return

    word_to_emb = {w: embeddings[i] for i, w in enumerate(unique_words)}
    
    print("-" * 50)
    print(f"{'Word A':<12} | {'Word B':<12} | {'Similarity':<10}")
    print("-" * 50)
    
    for w1, w2 in pairs:
        v1 = word_to_emb[w1]
        v2 = word_to_emb[w2]
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        print(f"{w1:<12} | {w2:<12} | {sim:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
