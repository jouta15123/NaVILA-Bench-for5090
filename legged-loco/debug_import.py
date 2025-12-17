
import sys
import os
from pathlib import Path

# Simulate path setup from style_module.py
current_path = Path(__file__).resolve()
NAVILA_ROOT = current_path.parents[2] # /home/jouta/NaVILA-Bench
print(f"NAVILA_ROOT determined as: {NAVILA_ROOT}")

if str(NAVILA_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVILA_ROOT))

print(f"sys.path: {sys.path}")

try:
    print("Attempting to import load_motionclip_full_model...")
    from hoyo_v1_1.models.train_motionclip_joint import load_motionclip_full_model
    print("Success!")
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Caught ImportError: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Caught Exception: {e}")
