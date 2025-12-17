
import unittest
import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add NaVILA-Bench root
NAVILA_ROOT = Path("/home/jouta/NaVILA-Bench")
sys.path.insert(0, str(NAVILA_ROOT))
# Add legged-loco modules
LEGGED_LOCO_ROOT = NAVILA_ROOT / "legged-loco" / "isaaclab_exts" / "omni.isaac.leggedloco"
if str(LEGGED_LOCO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGGED_LOCO_ROOT))

# Adjust path to find mdp.style_module
# The module is in omni.isaac.leggedloco.leggedloco.mdp.style_module
# Need to make sure 'omni' package is importable.
# IsaacLab usually puts extensions in python path.
# We might need to mock or set up path carefully.
# legged-loco/isaaclab_exts/omni.isaac.leggedloco contains 'omni' folder?
# No, it contains 'omni/isaac/...'
# So adding legged-loco/isaaclab_exts/omni.isaac.leggedloco to path might be wrong if 'omni' is top level.
# Correct path to add: legged-loco/isaaclab_exts/omni.isaac.leggedloco (so we can import omni) is WRONG.
# Usually: legged-loco/isaaclab_exts/omni.isaac.leggedloco IS where setup.py lives?
# Wait, list_dir of isaaclab_exts shows 'omni.isaac.leggedloco'. Inside that is 'omni'.
# So adding 'isaaclab_exts/omni.isaac.leggedloco' allows 'import omni'.
# Mock omni.isaac.lab to avoid import errors during standalone testing
from unittest.mock import MagicMock
sys.modules["omni.isaac.lab"] = MagicMock()
sys.modules["omni.isaac.lab.envs"] = MagicMock()
sys.modules["omni.isaac.lab.envs.mdp"] = MagicMock()
sys.modules["omni.isaac.lab.managers"] = MagicMock()
sys.modules["omni.isaac.lab_tasks"] = MagicMock()
sys.modules["omni.isaac.lab_tasks.manager_based"] = MagicMock()
sys.modules["omni.isaac.lab_tasks.manager_based.locomotion"] = MagicMock()
sys.modules["omni.isaac.lab_tasks.manager_based.locomotion.velocity"] = MagicMock()
sys.modules["omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp"] = MagicMock()
sys.modules["omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.rewards"] = MagicMock()


# Also mock mdp.__init__ to avoid triggering the real import
# We will import StyleModule by loading the file directly or adjusting path
# But since we are inside the package structure, it's tricky.
# Simpler approach: Add the directory containing style_module.py to sys.path
# and import it as a standalone module for testing purposes.
EXT_ROOT = NAVILA_ROOT / "legged-loco" / "isaaclab_exts" / "omni.isaac.leggedloco"
STYLE_MODULE_DIR = EXT_ROOT / "omni" / "isaac" / "leggedloco" / "leggedloco" / "mdp"
sys.path.insert(0, str(STYLE_MODULE_DIR))

# Now we can import StyleModule directly
try:
    from style_module import StyleModule
except ImportError:
    # Fallback to package import if direct import fails (e.g. relative imports in style_module)
    from omni.isaac.leggedloco.leggedloco.mdp.style_module import StyleModule

class TestStyleModule(unittest.TestCase):
    def setUp(self):
        # Patch the imports inside style_module.py if needed, or better, mock them at module level?
        # Since we imported StyleModule class, the module is already loaded.
        # But if it failed to import dependencies, it set them to None.
        # We need to monkeytype patch the module's globals.
        
        import style_module as sm_module
        
        # Create mocks
        self.mock_model = MagicMock()
        self.mock_model.latent_dim = 512
        self.mock_model.eval.return_value = None
        self.mock_model.return_value = {"mu": torch.zeros(1, 512)} # Mock forward pass
        
        self.mock_load_model = MagicMock(return_value=(self.mock_model, {}))
        
        self.mock_encode_semantics = MagicMock(return_value=torch.zeros(1, 1792))
        
        # Patch the module's globals
        sm_module.load_motionclip_full_model = self.mock_load_model
        sm_module.encode_semantics_sarashina = self.mock_encode_semantics
        sm_module.INSTRUCTION_ONOMATOPEIA = ["test_style"]
        
        # Also mock numpy load for centroids
        self.original_load = np.load
        np.load = MagicMock(return_value={
            "z_m": np.zeros((10, 512)), 
            "labels_idx": np.zeros(10), 
            "label_list": ["test_style"]
        })
    
    def tearDown(self):
        np.load = self.original_load

    def test_init(self):
        print("Testing StyleModule Initialization...")
        # Use CPU for test to be safe/fast
        module = StyleModule(device="cpu", run_name="sarashina_full_fixed")
        self.assertIsNotNone(module.motion_model)
        self.assertIsNotNone(module.sem_proj)
        print("StyleModule initialized successfully.")
        
    def test_encode(self):
        module = StyleModule(device="cpu", run_name="sarashina_full_fixed")
        text = "すたすた"
        z_onm, centroid = module.encode_instruction(text)
        self.assertEqual(z_onm.shape, (1, 512))
        # Centroid might be same as z_onm if missing, but shape holds
        self.assertEqual(centroid.shape, (1, 512))
        print("Encoding test passed.")

if __name__ == '__main__':
    unittest.main()
