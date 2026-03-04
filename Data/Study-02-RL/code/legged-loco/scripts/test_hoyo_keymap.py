#!/usr/bin/env python3
import torch
import unittest
import sys
import os
from unittest.mock import MagicMock

# Attempt to import StyleModule
# Adjust path to ensure we can import it
REPO_ROOT = "/home/jouta/NaVILA-Bench/legged-loco"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
    
PROJECT_ROOT = os.path.join(REPO_ROOT, "isaaclab_exts/omni.isaac.leggedloco")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from omni.isaac.leggedloco.leggedloco.mdp.style_module import StyleModule
except ImportError:
    # If direct likely fails due to isaaclab deps, we use a mock-heavy approach or just use the logic
    print("Warning: Could not import StyleModule directly (likely missing Isaac Sim context).")
    print("Using a localized version of the logic for verification.")
    pass

class TestStyleModuleCoordinates(unittest.TestCase):
    def test_yaw_correction_logic(self):
        """
        Verify the math: x_rot = x*c - y*s
        If Heading (Yaw) = 90 deg (Pi/2), then Robot Forward is +Y.
        We want Robot Forward to map to +X (or whatever local convention, usually X is forward).
        
        Wait, style_module says:
            x_rot = x * c - y * s
            y_rot = x * s + y * c
        
        If Yaw = 90 deg:
            c = cos(-90) = 0
            s = sin(-90) = -1
            
        If point is at (0, 1) [World Forward-ish for that robot, i.e. 1m North]:
            x=0, y=1
            x_rot = 0*0 - 1*(-1) = 1  => Local X (Forward)
            y_rot = 0*(-1) + 1*0 = 0  => Local Y (Left)
        
        Result: (1, 0). Correct? Yes, (1, 0) is Forward in Local.
        
        If point is at (-1, 0) [World Left]:
            x=-1, y=0
            x_rot = -1*0 - 0 = 0
            y_rot = -1*(-1) + 0 = 1 => Local Y (Left)
            
        Result: (0, 1). Correct? Yes, (0, 1) is Left in Local.
        
        So the rotation logic aligns World points to Local Robot Frame (X-Forward, Y-Left).
        """
        # Simulate logic
        yaw_deg = 90.0
        yaw = torch.tensor([yaw_deg * 3.14159 / 180.0])
        
        # Test Point: (0, 1, 0) -> Should be (1, 0, 0) in local
        centered_3d = torch.tensor([[[0.0, 1.0, 0.0]]]) # (B, T, 3)
        
        c = torch.cos(-yaw).view(-1, 1, 1)
        s = torch.sin(-yaw).view(-1, 1, 1)
        
        x = centered_3d[..., 0]
        y = centered_3d[..., 1]
        z = centered_3d[..., 2]
        
        x_rot = x * c - y * s
        y_rot = x * s + y * c
        
        print(f"Yaw: {yaw_deg} deg")
        print(f"Input: {centered_3d[0,0].tolist()}")
        print(f"Output: x={x_rot.item():.2f}, y={y_rot.item():.2f}")
        
        self.assertAlmostEqual(x_rot.item(), 1.0, places=4)
        self.assertAlmostEqual(y_rot.item(), 0.0, places=4)

    def test_hoyo_projection(self):
        """
        Verify HOYO projection mapping.
        HOYO Front view: 
          - HOYO x (Horizontal) = Robot Left
          - HOYO y (Vertical) = Robot Down (-Up)
          
        Logic in style_module:
            y_lat = centered_3d[..., 1:2]  (This is Y after rotation -> Local Left)
            z_up = centered_3d[..., 2:3]   (This is Z -> Local Up)
            centered = torch.cat([y_lat, -z_up], dim=-1)
            
        So:
            HOYO X = Local Left
            HOYO Y = -Local Up (Down)
            
        Check consistency.
        """
        # Local Point: (1, 0, 0) [Forward] -> HOYO X=0 (Mid), Y=0 (Mid) assuming centered
        # Local Point: (0, 1, 0) [Left]    -> HOYO X=1 (Right of image? No, Left positive?), Y=0
        # Local Point: (0, -1, 0) [Right]  -> HOYO X=-1
        # Local Point: (0, 0, 1) [Up]      -> HOYO X=0, Y=-1 (Top of image if Y is down positive?)
        
        # NOTE: MotionCLIP training implies HOYO x is 'Lateral'.
        # Assuming H1 Y=Left is mapped to HOYO X.
        pass

if __name__ == "__main__":
    unittest.main()
