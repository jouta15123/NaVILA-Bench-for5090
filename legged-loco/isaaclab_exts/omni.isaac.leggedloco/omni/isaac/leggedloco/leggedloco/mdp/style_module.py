
import os
import sys
import json # Added json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import isaaclab.utils.math as math_utils
from pathlib import Path
from collections import deque

def _get_navila_root():
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "hoyo_v1_1").exists() and (parent / "MotionCLIP").exists():
            return parent
    return Path("/home/jouta/NaVILA-Bench")

NAVILA_ROOT = _get_navila_root()
if str(NAVILA_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVILA_ROOT))

try:
    from hoyo_v1_1.models.train_motionclip_joint import load_motionclip_full_model
    from hoyo_v1_1.models.common import encode_semantics_sarashina, INSTRUCTION_ONOMATOPEIA, apply_normalization_from_stats
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Warning: Failed to import NaVILA modules: {e}")
    load_motionclip_full_model = None
    encode_semantics_sarashina = None
    INSTRUCTION_ONOMATOPEIA = []

# HOYO skeleton order (14 joints)
# 0: Head, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Hand
# 5: L-Shoulder, 6: L-Elbow, 7: L-Hand
# 8: R-Hip, 9: R-Knee, 10: R-Ankle
# 11: L-Hip, 12: L-Knee, 13: L-Ankle

class StyleModule:
    """
    Module to handle Onomatopoeia style encoding and reward computation for NaVILA.
    Maintains a buffer of motion history for reward computation.
    """
    def __init__(self, device: str = "cuda", run_name: str = "sarashina_full_fixed", num_envs: int = 1):
        self.device = torch.device(device)
        self.run_name = run_name
        self.root_dir = NAVILA_ROOT / "hoyo_v1_1" / "joint_training_results" / run_name
        self.num_envs = num_envs
        self.buffer_len = 60
        
        # Load Checkpoints
        self._load_models()
        self._load_centroids()
        self._load_norm_stats()
        
        # Motion Buffer: (num_envs, buffer_len, 14, 3)
        # We initialize with zeros or some default pose
        self.motion_buffer = torch.zeros(self.num_envs, self.buffer_len, 14, 3, device=self.device)
        self.heading_buffer = torch.zeros(self.num_envs, self.buffer_len, device=self.device)
        self.ptr = 0 # Ring buffer pointer? Or shift? Shifting is easier for now (less efficient but fine for 60 frames)
        
        # Cache for body indices to avoid string lookup every step
        self.body_indices = None
        
        # Load Mapping
        self.mapping_path = NAVILA_ROOT / "configs" / "h1_to_hoyo_mapping.json"
        if not self.mapping_path.exists():
            print(f"Warning: Mapping file not found at {self.mapping_path}. Using defaults.")
            self.mapping_dict = None
        else:
            with open(self.mapping_path, "r") as f:
                self.mapping_dict = json.load(f)
        
        # Track warmup state per environment
        self.warmup_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.warmup_frames = 60  # Need at least 60 frames before valid reward

    def reset_buffer(self, env_ids=None):
        """
        Reset motion buffer for specified environments.
        Should be called on episode reset to avoid stale data.
        
        Args:
            env_ids: Tensor of environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self.motion_buffer.zero_()
            self.heading_buffer.zero_()
            self.warmup_counter.zero_()
        else:
            self.motion_buffer[env_ids] = 0
            self.heading_buffer[env_ids] = 0
            self.warmup_counter[env_ids] = 0

    def _load_models(self):
        # ... (Same as before)
        self.motion_model, self.model_params = load_motionclip_full_model(self.device, target_len=60)
        self.motion_model.eval()
        for p in self.motion_model.parameters():
            p.requires_grad = False

        self.motion_dim = self.motion_model.latent_dim

        # Try to infer sem_dim from checkpoint first, then from encode_semantics_sarashina
        sem_proj_path = self.root_dir / "checkpoints" / "sem_proj_joint_best.pth"
        checkpoint = None
        if sem_proj_path.exists():
            checkpoint = torch.load(sem_proj_path, map_location=self.device)
            # weight shape is [out_features, in_features] = [motion_dim, sem_dim]
            self.sem_dim = checkpoint["weight"].shape[1]
        else:
            try:
                dummy_text = ["test"]
                dummy_emb = encode_semantics_sarashina(dummy_text, self.device)
                self.sem_dim = dummy_emb.shape[1]
            except Exception as e:
                print(f"Falling back to default sem_dim due to {e}")
                self.sem_dim = 1792  # Default for sarashina

        self.sem_proj = nn.Linear(self.sem_dim, self.motion_dim, bias=False).to(self.device)

        # If checkpoint exists but dimensions still mismatch (e.g., due to stubbed encoder),
        # rebuild sem_proj to match checkpoint before loading.
        if checkpoint is not None:
            ckpt_weight_shape = checkpoint["weight"].shape
            if tuple(self.sem_proj.weight.shape) != tuple(ckpt_weight_shape):
                in_features = ckpt_weight_shape[1]
                out_features = ckpt_weight_shape[0]
                self.sem_proj = nn.Linear(in_features, out_features, bias=False).to(self.device)
            self.sem_proj.load_state_dict(checkpoint, strict=False)
        self.sem_proj.eval()
        for p in self.sem_proj.parameters():
            p.requires_grad = False

    def _load_centroids(self):
        # ... (Same as before)
        snapshot_path = self.root_dir / "latent_snapshot_final.npz"
        if snapshot_path.exists():
            data = np.load(snapshot_path)
            z_all = torch.from_numpy(data["z_m"]).to(self.device)
            labels = data["labels_idx"]
            self.class_centroids = {}
            for lab_idx in np.unique(labels):
                mask = (labels == lab_idx)
                centroid = z_all[mask].mean(dim=0)
                self.class_centroids[int(lab_idx)] = F.normalize(centroid, dim=-1)
            self.label_list = list(data["label_list"]) if "label_list" in data else INSTRUCTION_ONOMATOPEIA
            self.label_to_id = {lab: i for i, lab in enumerate(self.label_list)}
        else:
            self.class_centroids = {}
            self.label_to_id = {}

    def _load_norm_stats(self):
        self.norm_path = self.root_dir.parent.parent / "data" / "normalization_stats.json"
        if not self.norm_path.exists():
             # Try joint_training_results/normalization_stats.json
             self.norm_path = NAVILA_ROOT / "hoyo_v1_1" / "joint_training_results" / "normalization_stats.json"
        
        self.mean = None
        self.std = None
        if self.norm_path.exists():
            import json
            with open(self.norm_path, "r") as f:
                stats = json.load(f)
                self.mean = torch.tensor(stats["mean"], device=self.device, dtype=torch.float32)
                self.std = torch.tensor(stats["std"], device=self.device, dtype=torch.float32)
        else:
            print(f"Warning: Normalization stats not found at {self.norm_path}. Using defaults.")
            # Default values estimated from HOYO data (approx)
            self.mean = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float32)
            self.std = torch.tensor([0.2, 0.2], device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode_instruction(self, text: str):
        # ... (Same as before)
        emb = encode_semantics_sarashina([text], self.device)
        # Align embedding dimensionality with sem_proj input (handles mocked encoders)
        in_feat = self.sem_proj.in_features
        if emb.shape[1] != in_feat:
            if emb.shape[1] > in_feat:
                emb = emb[:, :in_feat]
            else:
                pad = torch.zeros(emb.shape[0], in_feat - emb.shape[1], device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=1)
        z_onm = self.sem_proj(emb)
        z_onm = F.normalize(z_onm, dim=-1)
        
        if text in self.label_to_id:
            lab_idx = self.label_to_id[text]
            if lab_idx in self.class_centroids:
                centroid = self.class_centroids[lab_idx].unsqueeze(0)
                return z_onm, centroid
        return z_onm, z_onm

    def _get_hoyo_joints_from_h1(self, body_pos_w, body_names, body_quat_w=None):
        """
        Maps H1 (19-dof / many bodies) -> HOYO (14 joints)
        body_pos_w: (num_envs, num_bodies, 3)
        body_quat_w: (num_envs, num_bodies, 4) - Optional, used for offsets
        Returns: (num_envs, 14, 3)
        """
        if self.body_indices is None:
            # Create mapping from H1 to HOYO skeleton (14 joints)
            # HOYO order: Head, Neck, R-Shoulder, R-Elbow, R-Hand, L-Shoulder, L-Elbow, L-Hand, R-Hip, R-Knee, R-Ankle, L-Hip, L-Knee, L-Ankle
            # Note: Head and Hand links don't exist in H1, so we use parent links with offsets
            
            # Print available body names for debugging
            print(f"[StyleModule] Available H1 body names: {list(body_names)}")
            
            # Default Mapping (matches experiment_summary.md documentation)
            map_dict = {
                "head": "torso_link",  # Head approximated with torso_link + offset [0.0, 0.0, 0.25]
                "neck": "torso_link",  # Neck approximated with torso_link
                "r_shoulder": "right_shoulder_pitch_link",
                "r_elbow": "right_elbow_link",
                "r_hand": "right_elbow_link",  # Hand approximated with elbow_link + offset [0.30, 0.0, 0.0]
                "l_shoulder": "left_shoulder_pitch_link",
                "l_elbow": "left_elbow_link",
                "l_hand": "left_elbow_link",  # Hand approximated with elbow_link + offset [0.30, 0.0, 0.0]
                "r_hip": "right_hip_yaw_link",
                "r_knee": "right_knee_link",
                "r_ankle": "right_ankle_link",
                "l_hip": "left_hip_yaw_link",
                "l_knee": "left_knee_link",
                "l_ankle": "left_ankle_link",
            }
            
            # Override with JSON config if available (configs/h1_to_hoyo_mapping.json)
            if self.mapping_dict is not None:
                map_dict.update(self.mapping_dict)
                
            # HOYO order with indices
            # 0: Head, 1: Neck, 2: R-Shoulder, 3: R-Elbow, 4: R-Hand
            # 5: L-Shoulder, 6: L-Elbow, 7: L-Hand
            # 8: R-Hip, 9: R-Knee, 10: R-Ankle
            # 11: L-Hip, 12: L-Knee, 13: L-Ankle
            hoyo_order = [
                "head", "neck", 
                "r_shoulder", "r_elbow", "r_hand",
                "l_shoulder", "l_elbow", "l_hand",
                "r_hip", "r_knee", "r_ankle",
                "l_hip", "l_knee", "l_ankle"
            ]
            
            self.body_indices = [-1] * 14
            missing_links = []
            self._missing_joints = set()  # Track which joints are missing for offset computation
            
            for i, key in enumerate(hoyo_order):
                target_name = map_dict.get(key, "torso_link")
                found = False
                for idx, bname in enumerate(body_names):
                    if target_name in bname: # substring match
                        self.body_indices[i] = idx
                        found = True
                        break
                
                if not found:
                    missing_links.append(f"{key}->{target_name}")
                    self._missing_joints.add(key)
                    # Fallback to pelvis (0) or torso
                    # Try finding torso
                    for idx, bname in enumerate(body_names):
                        if "torso_link" in bname:
                            self.body_indices[i] = idx
                            break
                    if self.body_indices[i] == -1:
                        self.body_indices[i] = 0 # Absolute fallback
            
            if missing_links:
                print(f"[StyleModule] Warning: Could not find some H1 links: {missing_links}. Mapped to torso/pelvis.")
                print(f"[StyleModule] Missing joints will use offset-based estimation.")
                
            self.body_indices = torch.tensor(self.body_indices, device=self.device, dtype=torch.long)
            
            # Define Offsets (For missing endpoints like Head/Hand if mapped to parent)
            # Based on H1 robot dimensions (approximate):
            # - Height: ~1.8m, Shoulder width: ~0.4m, Arm length: ~0.6m
            # - Head above torso: ~0.25m, Neck: ~0.15m
            self.body_offsets = torch.zeros(14, 3, device=self.device)
            
            # Head offset (above torso)
            if "head" in self._missing_joints or ("torso" in map_dict["head"] and "head" not in map_dict["head"]):
                self.body_offsets[0] = torch.tensor([0.0, 0.0, 0.30], device=self.device)  # +Z for Head (30cm above torso)
            
            # Neck offset (slightly above torso)
            if "neck" in self._missing_joints or ("torso" in map_dict["neck"]):
                self.body_offsets[1] = torch.tensor([0.0, 0.0, 0.15], device=self.device)  # +Z for Neck (15cm above torso)
            
            # Right arm chain (if shoulders are missing, offset from torso)
            if "r_shoulder" in self._missing_joints:
                self.body_offsets[2] = torch.tensor([0.0, -0.20, 0.10], device=self.device)  # Right side, slight up
            if "r_elbow" in self._missing_joints:
                self.body_offsets[3] = torch.tensor([0.0, -0.35, 0.0], device=self.device)  # Further right
            if "r_hand" in self._missing_joints or ("elbow" in map_dict["r_hand"] and "hand" not in map_dict["r_hand"]):
                self.body_offsets[4] = torch.tensor([0.0, -0.25, -0.25], device=self.device)  # Down from elbow (forearm)
            
            # Left arm chain (if shoulders are missing, offset from torso)
            if "l_shoulder" in self._missing_joints:
                self.body_offsets[5] = torch.tensor([0.0, 0.20, 0.10], device=self.device)  # Left side, slight up
            if "l_elbow" in self._missing_joints:
                self.body_offsets[6] = torch.tensor([0.0, 0.35, 0.0], device=self.device)  # Further left
            if "l_hand" in self._missing_joints or ("elbow" in map_dict["l_hand"] and "hand" not in map_dict["l_hand"]):
                self.body_offsets[7] = torch.tensor([0.0, 0.25, -0.25], device=self.device)  # Down from elbow (forearm)
            
            print(f"[StyleModule] Final body indices: {self.body_indices.tolist()}")
            print(f"[StyleModule] Offsets applied for: {[hoyo_order[i] for i in range(14) if self.body_offsets[i].abs().sum() > 0]}")

        # Gather joints
        # (B, NumBodies, 3) -> (B, 14, 3)
        hoyo_3d = body_pos_w[:, self.body_indices, :]
        
        # Apply Offsets if orientation is provided
        if body_quat_w is not None:
             # Expand offsets: (1, 14, 3)
             offsets = self.body_offsets.unsqueeze(0).expand(hoyo_3d.shape[0], -1, -1)
             
             # Get orientations for the mapped bodies
             hoyo_quat = body_quat_w[:, self.body_indices, :]
             
             # Rotate offsets
             # (B, 14, 4) x (B, 14, 3)
             B = hoyo_3d.shape[0]
             q_flat = hoyo_quat.reshape(-1, 4)
             v_flat = offsets.reshape(-1, 3)
             
             offsets_rot_flat = math_utils.quat_rotate(q_flat, v_flat)
             offsets_rot = offsets_rot_flat.view(B, 14, 3)
             
             hoyo_3d = hoyo_3d + offsets_rot
            
        return hoyo_3d # (B, 14, 3)

    def update_buffer(self, body_pos_w, root_quat_w, body_names, body_quat_w=None):
        """
        Updates the internal motion buffer with the current H1 pose.
        body_pos_w: (num_envs, num_bodies, 3)
        root_quat_w: (num_envs, 4)
        """
        if self.motion_buffer.shape[0] != body_pos_w.shape[0]:
            # Resize buffer if env count changes (e.g. play vs train)
            self.num_envs = body_pos_w.shape[0]
            self.motion_buffer = torch.zeros(self.num_envs, self.buffer_len, 14, 3, device=self.device)
            self.heading_buffer = torch.zeros(self.num_envs, self.buffer_len, device=self.device)
            self.warmup_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.body_indices = None # Reset mapping
            
        current_pose_3d = self._get_hoyo_joints_from_h1(body_pos_w, body_names, body_quat_w) # (B, 14, 3)
        
        # Calculate Heading (Yaw)
        # Using isaaclab math utility or manually
        # q = [w, x, y, z] in Isaac Lab (Scipy/Warp convention?) or [x, y, z, w]?
        # Isaac Sim uses [w, x, y, z].
        # Yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        w, x, y, z = root_quat_w[:, 0], root_quat_w[:, 1], root_quat_w[:, 2], root_quat_w[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Shift buffer
        # (B, T, 14, 3) -> (B, T-1, ...)
        self.motion_buffer = torch.cat([self.motion_buffer[:, 1:], current_pose_3d.unsqueeze(1)], dim=1)
        self.heading_buffer = torch.cat([self.heading_buffer[:, 1:], yaw.unsqueeze(1)], dim=1)
        
        # Increment warmup counter (capped at warmup_frames)
        self.warmup_counter = torch.clamp(self.warmup_counter + 1, max=self.warmup_frames)

    def encode_buffer(self):
        """
        Encodes the current buffer into style latent.
        Applies Centering and Normalization as in HOYO dataset.
        Aligns motion to Frame 0 heading.
        """
        # 1. Align Translation (Center to Frame 0 CoM)
        # We use Frame 0 of the window as the reference origin.
        # But CoM of Frame 0 is better.
        com_0 = self.motion_buffer[:, 0].mean(dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1, 3)
        centered_3d = self.motion_buffer - com_0
        
        # 2. Align Rotation (Rotate by -Yaw of Frame 0)
        yaw_0 = self.heading_buffer[:, 0] # (B,)
        
        # Rotate around Z axis (Assuming Z is up in World)
        # x' = x cos(-yaw) - y sin(-yaw)
        # y' = x sin(-yaw) + y cos(-yaw)
        # z' = z
        # Since we rotate by -yaw: cos(-a)=cos(a), sin(-a)=-sin(a)
        c = torch.cos(yaw_0).view(-1, 1, 1, 1)
        s = torch.sin(yaw_0).view(-1, 1, 1, 1)
        
        x = centered_3d[..., 0:1]
        y = centered_3d[..., 1:2]
        z = centered_3d[..., 2:3]
        
        # Inverse rotation (Rotate World -> Local)
        # To make "North" (+Y) become "East" (+X)?
        # If Yaw=90 (North), we want to rotate by -90.
        # x_local = x * cos(yaw) + y * sin(yaw)
        # y_local = -x * sin(yaw) + y * cos(yaw)
        # Let's verify: P=(0,1) (North), Yaw=90.
        # x' = 0 + 1*1 = 1. y' = 0 + 0 = 0. -> (1,0) (East). Correct.
        
        x_rot = x * c + y * s
        # y_rot = -x * s + y * c # We don't strictly need Y for 2D projection if projecting to XZ
        z_rot = z
        
        # 3. Project to 2D (Sagittal Plane X-Z)
        # Logic: After rotation, X is forward, Z is up.
        # "normalization_stats.json" has std=[0.1, 0.29].
        # Since forward motion (X) has much larger variance than vertical (Z),
        # and std[1] > std[0], we infer expected order is [LowVar, HighVar] -> [Z, X].
        # Current H1 X (Forward) -> 1.0m+ variance. H1 Z (Up) -> 0.1m variance.
        # So we MUST map X -> Index 1, Z -> Index 0.
        centered_2d = torch.cat([z_rot, x_rot], dim=-1) # (B, T, 14, 2)
        
        centered = centered_2d

        # Debug Stats (Transient)
        if torch.rand(1).item() < 0.005:
            with torch.no_grad():
                # Print raw ranges before normalization to check assumptions
                x_range = x_rot.max() - x_rot.min()
                z_range = z_rot.max() - z_rot.min()
                print(f"[DEBUG EncBuffer] RawRanges X(Fwd)={x_range:.2f}, Z(Up)={z_range:.2f}")

        # 4. Height Scaling (NEW)
        # HOYO scale: mean(norm(Head - FeetMid))
        # We calculate this from the Motion Buffer (B, T, 14, 3) *before* rotation/projection to be safe, 
        # but 2D projection is X-Z (Sagittal) and H1 is standing up (Z), so 2D distance is also good approx if side-view.
        # Use 3D distance for accuracy.
        # Head: 0, R-Ankle: 10, L-Ankle: 13
        
        head_pos = self.motion_buffer[:, :, 0, :] # (B, T, 3)
        r_ankle = self.motion_buffer[:, :, 10, :]
        l_ankle = self.motion_buffer[:, :, 13, :]
        feet_mid = 0.5 * (r_ankle + l_ankle)
        
        heights = torch.norm(head_pos - feet_mid, dim=-1) # (B, T)
        scale = heights.mean(dim=1, keepdim=True) # (B, 1) to broadcast over T
        
        # Avoid division by zero
        scale = torch.maximum(scale, torch.tensor(1.0, device=self.device))
        
        # Broadcast scale to (B, 1, 1, 1) for (B, T, 14, 2)
        scale_bc = scale.view(-1, 1, 1, 1)
        centered = centered / scale_bc

        # 5. Normalize
        if self.mean is not None:
            # Handle Global (2,) or Local (14, 2) stats
            if self.mean.numel() == 2:
                # Global stats: [mean_x, mean_y]
                # centering is (B, T, 14, 2). 
                # mean needs to broadcast to (1, 1, 1, 2)
                mean = self.mean.view(1, 1, 1, 2)
                std = self.std.view(1, 1, 1, 2)
            else:
                # Local stats: (14, 2)
                mean = self.mean.view(1, 1, 14, 2)
                std = self.std.view(1, 1, 14, 2)
                
            centered = (centered - mean) / (std + 1e-6) # Add epsilon to avoid div by zero
             
        # 3. MotionCLIP
        x = centered.permute(0, 2, 3, 1) # (B, 14, 2, 60)
        
        B = x.shape[0]
        Tcur = x.shape[3]
        
        batch = {
            "x": x,
            "mask": torch.ones((B, Tcur), dtype=torch.bool, device=self.device),
            "lengths": torch.full((B,), Tcur, dtype=torch.long, device=self.device),
            "y": torch.zeros((B,), dtype=torch.long, device=self.device)
        }
        
        out = self.motion_model(batch)
        z_m = F.normalize(out["mu"], dim=-1)
        return z_m # (B, 512)

    def compute_current_reward(self, 
                             target_z_onm: torch.Tensor, 
                             target_centroid: torch.Tensor,
                             beta_text: float = 0.5,
                             beta_centroid: float = 0.5):
                                 
        z_agent = self.encode_buffer() # (B, 512)
        
        # Check for NaNs
        if torch.isnan(z_agent).any():
            # print("Warning: NaN in z_agent! Zeroing reward to prevent crash.")
            z_agent = torch.nan_to_num(z_agent, nan=0.0)
            
        r_text = (z_agent * target_z_onm).sum(dim=-1)
        r_centroid = (z_agent * target_centroid).sum(dim=-1)
        
        reward = beta_text * r_text + beta_centroid * r_centroid
        
        # Sanitize reward
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply warmup mask: zero reward until buffer is filled
        # This prevents learning from garbage data at episode start
        warmup_mask = (self.warmup_counter >= self.warmup_frames).float()
        reward = reward * warmup_mask
        r_text = r_text * warmup_mask
        r_centroid = r_centroid * warmup_mask
        
        return reward, r_text, r_centroid
