
import os
import sys
import json # Added json
import torch
import torch.nn as nn
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
    def __init__(
        self,
        device: str = "cuda",
        run_name: str = "sarashina_full_fixed",
        num_envs: int = 1,
        coord_mode: str = "legacy_xz_yaw",
        buffer_len: int = 100,  # 2秒 × 50Hz
    ):
        self.device = torch.device(device)
        self.run_name = run_name
        self.root_dir = NAVILA_ROOT / "hoyo_v1_1" / "joint_training_results" / run_name
        self.num_envs = num_envs
        self.buffer_len = buffer_len
        self.coord_mode = coord_mode
        if self.coord_mode not in ("legacy_xz_yaw", "hoyo_front"):
            raise ValueError(f"Unsupported coord_mode: {self.coord_mode}")

        # Head/neck estimation ratios from HOYO stats (hs/hf, ns/hf).
        # Allow override via env vars if needed.
        self.head_shoulder_ratio = float(os.environ.get("HOYO_HEAD_SHOULDER_RATIO", "0.2375"))
        self.neck_shoulder_ratio = float(os.environ.get("HOYO_NECK_SHOULDER_RATIO", "0.15"))

        # Text embedding device (default CPU to avoid GPU OOM when running RL)
        self.text_device = torch.device(os.environ.get("STYLE_TEXT_DEVICE", "cpu"))
        # Centroid selection mode: "centroid" (default) or "random"
        self.centroid_mode = os.environ.get("STYLE_CENTROID_MODE", "centroid").strip().lower()
        
        # Load Checkpoints
        self._load_models()
        self._load_centroids()
        self._load_norm_stats()
        self._precompute_text_latents()
        
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

        # Optional offsets override (meters in parent link local frame)
        self.offsets_override = None
        offsets_path = NAVILA_ROOT / "configs" / "h1_to_hoyo_offsets.json"
        if offsets_path.exists():
            try:
                with open(offsets_path, "r") as f:
                    self.offsets_override = json.load(f)
            except Exception as exc:
                print(f"Warning: Failed to load offsets file: {offsets_path}: {exc}")
                self.offsets_override = None
        
        # Track warmup state per environment
        self.warmup_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.warmup_frames = self.buffer_len  # Need full buffer before valid reward
        
        # Fixed scale reference per environment (nan = uninitialized)
        self.scale_ref = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.scale_ema = 0.99  # EMA coefficient for scale update
        self.scale_min = 0.8   # Minimum valid scale (clamp)
        self.scale_max = 2.2   # Maximum valid scale (clamp)
        self.scale_healthy_threshold = 1.0  # Minimum head-feet dist to consider "standing"

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
            self.scale_ref.fill_(float("nan"))
        else:
            self.motion_buffer[env_ids] = 0
            self.heading_buffer[env_ids] = 0
            self.warmup_counter[env_ids] = 0
            self.scale_ref[env_ids] = float("nan")

    def _load_models(self):
        # ... (Same as before)
        self.motion_model, self.model_params = load_motionclip_full_model(self.device, target_len=self.buffer_len)
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

    def _precompute_text_latents(self):
        """Precompute z_onm for all known styles to avoid repeated text encoder loads."""
        self._z_onm_cache = {}
        if encode_semantics_sarashina is None or len(INSTRUCTION_ONOMATOPEIA) == 0:
            return
        try:
            emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, self.text_device)
            # Align embedding dimensionality with sem_proj input (handles mocked encoders)
            in_feat = self.sem_proj.in_features
            if emb.shape[1] != in_feat:
                if emb.shape[1] > in_feat:
                    emb = emb[:, :in_feat]
                else:
                    pad = torch.zeros(emb.shape[0], in_feat - emb.shape[1], device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=1)
            emb = emb.to(self.device)
            z_onm = F.normalize(self.sem_proj(emb), dim=-1)
            for idx, label in enumerate(INSTRUCTION_ONOMATOPEIA):
                self._z_onm_cache[label] = z_onm[idx : idx + 1]
        except Exception as e:
            print(f"Warning: Failed to precompute text latents: {e}")

    def _load_centroids(self):
        # ... (Same as before)
        snapshot_path = self.root_dir / "latent_snapshot_final.npz"
        if snapshot_path.exists():
            data = np.load(snapshot_path)
            z_all = torch.from_numpy(data["z_m"]).to(self.device)
            labels = data["labels_idx"]
            self.class_centroids = {}
            self.class_latents = {}
            for lab_idx in np.unique(labels):
                mask = (labels == lab_idx)
                z_lab = z_all[mask]
                # Normalize for safety (snapshot should already be normalized)
                z_lab = F.normalize(z_lab, dim=-1)
                centroid = z_lab.mean(dim=0)
                self.class_centroids[int(lab_idx)] = F.normalize(centroid, dim=-1)
                self.class_latents[int(lab_idx)] = z_lab
            self.label_list = list(data["label_list"]) if "label_list" in data else INSTRUCTION_ONOMATOPEIA
            self.label_to_id = {lab: i for i, lab in enumerate(self.label_list)}
        else:
            self.class_centroids = {}
            self.class_latents = {}
            self.label_to_id = {}

    def _load_norm_stats(self):
        # Priority: run-specific > global joint_training_results > data directory
        candidates = [
            self.root_dir / "normalization_stats.json",
            NAVILA_ROOT / "hoyo_v1_1" / "joint_training_results" / "normalization_stats.json",
            NAVILA_ROOT / "hoyo_v1_1" / "data" / "normalization_stats.json",
        ]
        self.norm_path = None
        for path in candidates:
            if path.exists():
                self.norm_path = path
                break
        
        self.mean = None
        self.std = None
        if self.norm_path is not None and self.norm_path.exists():
            import json
            with open(self.norm_path, "r") as f:
                stats = json.load(f)
                self.mean = torch.tensor(stats["mean"], device=self.device, dtype=torch.float32)
                self.std = torch.tensor(stats["std"], device=self.device, dtype=torch.float32)
        else:
            print("Warning: Normalization stats not found. Using defaults.")
            # Default values estimated from HOYO data (approx)
            self.mean = torch.tensor([0.0, 0.0], device=self.device, dtype=torch.float32)
            self.std = torch.tensor([0.2, 0.2], device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode_instruction(self, text: str):
        # ... (Same as before)
        if text in getattr(self, "_z_onm_cache", {}):
            z_onm = self._z_onm_cache[text]
        else:
            emb = encode_semantics_sarashina([text], self.text_device)
            # Align embedding dimensionality with sem_proj input (handles mocked encoders)
            in_feat = self.sem_proj.in_features
            if emb.shape[1] != in_feat:
                if emb.shape[1] > in_feat:
                    emb = emb[:, :in_feat]
                else:
                    pad = torch.zeros(emb.shape[0], in_feat - emb.shape[1], device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=1)
            emb = emb.to(self.device)
            z_onm = self.sem_proj(emb)
            z_onm = F.normalize(z_onm, dim=-1)
        
        if text in self.label_to_id:
            lab_idx = self.label_to_id[text]
            # Random sample from teacher latents if requested and available
            if self.centroid_mode == "random" and lab_idx in self.class_latents:
                z_lab = self.class_latents[lab_idx]
                if z_lab.shape[0] > 0:
                    ridx = torch.randint(0, z_lab.shape[0], (1,), device=z_lab.device)
                    centroid = z_lab[ridx].squeeze(0).unsqueeze(0)
                    return z_onm, centroid
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
                "head": "torso_link",  # Head approximated with torso_link (offset applied)
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
            # Cache resolved mapping for later use (e.g., head/neck estimation)
            self._resolved_map_dict = map_dict
                
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
            
            # Head/neck offsets (above torso)
            # NOTE: actual head/neck positions are estimated from shoulder mid later when missing.
            if "head" in self._missing_joints or ("torso" in map_dict["head"] and "head" not in map_dict["head"]):
                self.body_offsets[0] = torch.tensor([0.0, 0.0, 0.85], device=self.device)
            
            if "neck" in self._missing_joints or ("torso" in map_dict["neck"]):
                self.body_offsets[1] = torch.tensor([0.0, 0.0, 0.52], device=self.device)
            
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

            # Apply override offsets (if provided) when joint is missing or mapped to a parent link
            if self.offsets_override:
                applied = []
                for key, vec in self.offsets_override.items():
                    if key not in map_dict:
                        continue
                    try:
                        idx = hoyo_order.index(key)
                    except ValueError:
                        continue
                    target_name = str(map_dict.get(key, ""))
                    should_apply = (
                        key in self._missing_joints
                        or "elbow" in target_name
                        or "torso" in target_name
                        or "pelvis" in target_name
                    )
                    if not should_apply:
                        continue
                    if not (isinstance(vec, (list, tuple)) and len(vec) == 3):
                        continue
                    self.body_offsets[idx] = torch.tensor(vec, device=self.device, dtype=torch.float32)
                    applied.append(key)
                if applied:
                    print(f"[StyleModule] Offset overrides applied for: {applied}")

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
            
        # Estimate head/neck from shoulder mid if actual head/neck links are missing.
        # h = r/(1-r) * (shoulder_z - feet_z)
        map_dict = getattr(self, "_resolved_map_dict", self.mapping_dict or {})
        if (
            ("head" in self._missing_joints)
            or ("neck" in self._missing_joints)
            or ("torso" in map_dict.get("head", ""))
        ):
            if not any(k in self._missing_joints for k in ("r_shoulder", "l_shoulder", "r_ankle", "l_ankle")):
                r_sh = hoyo_3d[:, 2, :]
                l_sh = hoyo_3d[:, 5, :]
                shoulder_mid = 0.5 * (r_sh + l_sh)
                r_ank = hoyo_3d[:, 10, :]
                l_ank = hoyo_3d[:, 13, :]
                feet_mid = 0.5 * (r_ank + l_ank)
                up_vec = shoulder_mid - feet_mid
                d = torch.norm(up_vec, dim=-1)
                d = torch.clamp(d, min=1e-6)
                up_dir = up_vec / d.unsqueeze(-1)
                h_off = (self.head_shoulder_ratio / (1.0 - self.head_shoulder_ratio)) * d
                n_off = (self.neck_shoulder_ratio / (1.0 - self.neck_shoulder_ratio)) * d

                # Head / Neck along body-up direction (not fixed world-Z)
                hoyo_3d[:, 0, :] = shoulder_mid + up_dir * h_off.unsqueeze(-1)
                hoyo_3d[:, 1, :] = shoulder_mid + up_dir * n_off.unsqueeze(-1)

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

    @torch.no_grad()
    @torch.no_grad()
    def _prepare_centered_2d(
        self,
        apply_yaw_correction: bool = False,
        coord_mode: str | None = None,
        centering: str | None = None,
        standardize: bool = True,
        normalize_height: bool = True,
    ) -> torch.Tensor:
        """
        Prepare motion buffer in the same 2D format used for MotionCLIP/HOYO.

        HOYO前処理に準拠:
        1. 初期フレームの重心でセンタリング
        2. 3D→2D投影: [Y(左右), Z(上下)] → [x, y] (HOYO正面視点座標系)
        3. 身長正規化 (頭〜足の距離) - normalize_height=Trueの場合のみ
        4. 標準化 (mean/std)

        Args:
            apply_yaw_correction: Yaw補正を適用するかどうか（hoyo_frontモードのみ有効）。
            coord_mode: "legacy_xz_yaw" or "hoyo_front". If None, uses self.coord_mode.
            centering: Centering mode for hoyo_front. "pelvis" (default) or "first_frame_com".
            standardize: Whether to apply mean/std normalization.
            normalize_height: Whether to apply height normalization (default True). Set False for raw meter visualization.

        Returns:
            (B, T, 14, 2) shaped tensor in [x, y] order matching HOYO format.
        """
        coord_mode = coord_mode or self.coord_mode
        if coord_mode == "legacy_xz_yaw":
            # Legacy: XZ sagittal projection with yaw correction (pre-2025-12-16 behavior)
            # 1. Frame 0のCoMでセンタリング（バッファ未充足時は最初の有効フレームを使う）
            if hasattr(self, "warmup_counter"):
                B, T = self.motion_buffer.shape[0], self.motion_buffer.shape[1]
                valid_len = torch.clamp(self.warmup_counter, min=1, max=T).long()  # (B,)
                idx0 = (T - valid_len).clamp(min=0, max=T - 1)  # (B,)
                batch_idx = torch.arange(B, device=self.motion_buffer.device)
                frame0 = self.motion_buffer[batch_idx, idx0]  # (B, 14, 3)
                com_0 = frame0.mean(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1, 3)
            else:
                com_0 = self.motion_buffer[:, 0].mean(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1, 3)
            centered_3d = self.motion_buffer - com_0

            # 2. Align rotation by -Yaw of frame 0 (legacy)
            yaw_0 = self.heading_buffer[:, 0]  # (B,)
            c = torch.cos(yaw_0).view(-1, 1, 1, 1)
            s = torch.sin(yaw_0).view(-1, 1, 1, 1)

            x = centered_3d[..., 0:1]
            y = centered_3d[..., 1:2]
            z = centered_3d[..., 2:3]

            x_rot = x * c + y * s
            z_rot = z

            # 3. Project to 2D (Z, X) order
            centered = torch.cat([z_rot, x_rot], dim=-1)

            # 4. Height scaling (3D head-feet distance)
            head_pos = self.motion_buffer[:, :, 0, :]  # (B, T, 3)
            r_ankle = self.motion_buffer[:, :, 10, :]
            l_ankle = self.motion_buffer[:, :, 13, :]
            feet_mid = 0.5 * (r_ankle + l_ankle)
            heights = torch.norm(head_pos - feet_mid, dim=-1)  # (B, T)
            # Use only valid frames to avoid scale drift from zero-initialized buffer.
            if hasattr(self, "warmup_counter"):
                T = heights.shape[1]
                valid_len = torch.clamp(self.warmup_counter, min=1, max=T).view(-1, 1)
                t_idx = torch.arange(T, device=heights.device).view(1, T)
                mask = (t_idx >= (T - valid_len)).float()
                scale = (heights * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                scale = scale.view(-1, 1)
            else:
                scale = heights.mean(dim=1, keepdim=True)  # (B, 1)
            # Match HOYO behavior: only guard against near-zero scale
            scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            scale_bc = scale.view(-1, 1, 1, 1)
            centered = centered / scale_bc

            # 5. Normalize (mean/std)
            if standardize and self.mean is not None:
                if self.mean.numel() == 2:
                    mean = self.mean.view(1, 1, 1, 2)
                    std = self.std.view(1, 1, 1, 2)
                else:
                    mean = self.mean.view(1, 1, 14, 2)
                    std = self.std.view(1, 1, 14, 2)
                centered = (centered - mean) / (std + 1e-6)

            return centered

        # Default: HOYO frontal projection
        centering = centering or "first_frame_com"
        # Per-frame COM (HOYO-style) for scale computation only
        com_pf = self.motion_buffer.mean(dim=2, keepdim=True)  # (B, T, 1, 3)
        centered_3d_scale = self.motion_buffer - com_pf
        if centering == "pelvis":
            # 1. Per-frame Pelvis Centering (translation invariant)
            # Pelvis = midpoint of R-Hip(8) and L-Hip(11)
            r_hip = self.motion_buffer[:, :, 8, :]   # (B, T, 3)
            l_hip = self.motion_buffer[:, :, 11, :]  # (B, T, 3)
            pelvis = 0.5 * (r_hip + l_hip)           # (B, T, 3)
            centered_3d = self.motion_buffer - pelvis[:, :, None, :]  # (B, T, 14, 3)
        elif centering == "first_frame_com":
            # 1'. First-frame CoM centering (matches HOYO first_frame_com)
            # バッファ未充足時は最初の有効フレームを基準にする（ゼロ初期化の影響を避ける）
            if hasattr(self, "warmup_counter"):
                B, T = self.motion_buffer.shape[0], self.motion_buffer.shape[1]
                valid_len = torch.clamp(self.warmup_counter, min=1, max=T).long()  # (B,)
                idx0 = (T - valid_len).clamp(min=0, max=T - 1)  # (B,)
                batch_idx = torch.arange(B, device=self.motion_buffer.device)
                frame0 = self.motion_buffer[batch_idx, idx0]  # (B, 14, 3)
                com_0 = frame0.mean(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1, 3)
            else:
                com_0 = self.motion_buffer[:, 0].mean(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1, 3)
            centered_3d = self.motion_buffer - com_0  # (B, T, 14, 3)
        else:
            raise ValueError(f"Unsupported centering mode: {centering}")

        # 2. (Optional) Yaw補正 - デフォルトはOFF（HOYO正規化統計に合わせる）
        if apply_yaw_correction:
            # User Feedback: Use per-frame yaw to maintain "always frontal" view even during turns.
            yaw = self.heading_buffer  # (B, T)
            # Broadcast to (B, T, 1, 1) for (B, T, 14, 1) position data
            c = torch.cos(-yaw).view(yaw.shape[0], yaw.shape[1], 1, 1)
            s = torch.sin(-yaw).view(yaw.shape[0], yaw.shape[1], 1, 1)

            x = centered_3d[..., 0:1]
            y = centered_3d[..., 1:2]
            z = centered_3d[..., 2:3]

            x_rot = x * c - y * s
            y_rot = x * s + y * c
            centered_3d = torch.cat([x_rot, y_rot, z], dim=-1)

            # Apply the same yaw correction to scale-computation frames
            x_s = centered_3d_scale[..., 0:1]
            y_s = centered_3d_scale[..., 1:2]
            z_s = centered_3d_scale[..., 2:3]
            x_s_rot = x_s * c - y_s * s
            y_s_rot = x_s * s + y_s * c
            centered_3d_scale = torch.cat([x_s_rot, y_s_rot, z_s], dim=-1)

        # 3. 3D→2D投影: H1[Y(左右), Z(上下)] → HOYO正面視点 [x, y]
        # HOYO座標系（画像座標系、frontビュー、swap後）:
        #   - HOYO x: 被写体の左が正（画像の水平方向）
        #   - HOYO y: 下向き正（画像の垂直方向）
        # H1座標系:
        #   - H1 Y: ロボットの左が正
        #   - H1 Z: 上向き正
        # マッピング:
        #   - HOYO x = +H1 Y（左向き正をそのまま使用）
        #   - HOYO y = -H1 Z（下向き正 ← 上向き正）
        y_lat = centered_3d[..., 1:2]  # H1 Y: 左向き正
        z_up = centered_3d[..., 2:3]   # H1 Z: 上向き正
        centered = torch.cat([y_lat, -z_up], dim=-1)

        y_lat_s = centered_3d_scale[..., 1:2]
        z_up_s = centered_3d_scale[..., 2:3]
        centered_scale = torch.cat([y_lat_s, -z_up_s], dim=-1)

        # 4. 身長正規化（HOYOと同じ: head-feet 平均スケール）
        if normalize_height:
            head_pos = centered_scale[:, :, 0, :]   # (B, T, 2) - head
            r_ankle = centered_scale[:, :, 10, :]   # (B, T, 2) - right ankle
            l_ankle = centered_scale[:, :, 13, :]   # (B, T, 2) - left ankle
            feet_mid = 0.5 * (r_ankle + l_ankle)

            # Current height per env (mean over valid frames)
            heights = torch.norm(head_pos - feet_mid, dim=-1)  # (B, T)
            if hasattr(self, "warmup_counter"):
                T = heights.shape[1]
                valid_len = torch.clamp(self.warmup_counter, min=1, max=T).view(-1, 1)
                t_idx = torch.arange(T, device=heights.device).view(1, T)
                mask = (t_idx >= (T - valid_len)).float()
                scale = (heights * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                scale = heights.mean(dim=1)  # (B,)
            # Match HOYO behavior: only guard against near-zero scale
            scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
            scale_bc = scale.view(-1, 1, 1, 1)
            centered = centered / scale_bc

        # 5. 標準化 (mean/std) - HOYO正規化統計を使用
        if standardize and self.mean is not None:
            if self.mean.numel() == 2:
                mean = self.mean.view(1, 1, 1, 2)
                std = self.std.view(1, 1, 1, 2)
            else:
                mean = self.mean.view(1, 1, 14, 2)
                std = self.std.view(1, 1, 14, 2)
            centered = (centered - mean) / (std + 1e-6)

        return centered

    @torch.no_grad()
    def get_processed_buffer_2d(self) -> torch.Tensor:
        """
        Expose the current motion buffer in processed 2D format (B, T, 14, 2).
        """
        return self._prepare_centered_2d(coord_mode=self.coord_mode)

    @torch.no_grad()
    def get_buffer_for_hoyo_comparison(
        self,
        apply_yaw_correction: bool = False,
        coord_mode: str | None = None,
        centering: str = "first_frame_com",
        standardize: bool = True,
        normalize_height: bool = True,
    ) -> torch.Tensor:
        """
        HOYO誤差計算用に、HOYOデータセットと同じ前処理を適用したバッファを返す。

        回転の影響を除外したい場合は apply_yaw_correction を有効にする。

        Args:
            apply_yaw_correction: Yaw補正を適用するかどうか。
            coord_mode: "legacy_xz_yaw" or "hoyo_front". Noneならself.coord_mode。
            centering: HOYOのセンタリング方式（例: "first_frame_com"）。
            standardize: mean/std 正規化を適用するか。
            normalize_height: 身長正規化を適用するか。

        Returns:
            (B, T, 14, 2) shaped tensor in [x, y] order matching HOYO format.
        """
        coord_mode = coord_mode or self.coord_mode
        return self._prepare_centered_2d(
            apply_yaw_correction=apply_yaw_correction,
            coord_mode=coord_mode,
            centering=centering,
            standardize=standardize,
            normalize_height=normalize_height,
        )

    @torch.no_grad()
    def get_hoyo_compatible_keymap(self, standardize: bool = True, normalize_height: bool = True) -> torch.Tensor:
        """
        Get the current motion buffer as a HOYO-compatible 2D keymap with rotation invariance.
        This aligns the robot's heading to the keymap's 'forward' (or lateral) axis.

        Args:
            standardize: Whether to apply mean/std normalization.
            normalize_height: Whether to apply height normalization. Set False for raw meter visualization.

        Returns:
            (B, T, 14, 2) shaped tensor in [Lateral, Up] order.
            Lateral: Robot's Left direction
            Up: Robot's Down direction (HOYO image coordinates)
        """
        # Enable Yaw correction to ensure rotation invariance
        return self._prepare_centered_2d(apply_yaw_correction=True, coord_mode=self.coord_mode, standardize=standardize, normalize_height=normalize_height)

    @torch.no_grad()
    def encode_buffer(self):
        """
        Encodes the current buffer into style latent using MotionCLIP.

        HOYO前処理に準拠（正面視点座標系）:
        - 初期フレーム中心化
        - 身長正規化
        - [Y_lat, Z_up] → [x, y]座標変換（Yaw補正あり）
        - 標準化 (mean/std)

        Returns:
            (B, 512) shaped tensor of normalized motion latents.
        """
        # Apply Yaw correction for rotation invariant style reward
        centered = self._prepare_centered_2d(apply_yaw_correction=True, coord_mode=self.coord_mode)

        # MotionCLIP expects (B, 14, 2, T)
        x = centered.permute(0, 2, 3, 1)  # (B, 14, 2, 60)
        
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

    def compute_current_reward(
        self,
        target_z_onm: torch.Tensor,
        target_teacher_motion: torch.Tensor,
        beta_text: float = 0.5,
        beta_teacher_motion: float = 0.5,
        beta_centroid: float | None = None,
    ):
                                 
        z_agent = self.encode_buffer() # (B, 512)
        
        # Check for NaNs
        if torch.isnan(z_agent).any():
            # print("Warning: NaN in z_agent! Zeroing reward to prevent crash.")
            z_agent = torch.nan_to_num(z_agent, nan=0.0)
            
        if beta_centroid is not None:
            beta_teacher_motion = beta_centroid

        r_text = (z_agent * target_z_onm).sum(dim=-1)
        r_teacher_motion = (z_agent * target_teacher_motion).sum(dim=-1)
        
        reward = beta_text * r_text + beta_teacher_motion * r_teacher_motion
        
        # Sanitize reward
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply warmup mask: zero reward until buffer is filled
        # This prevents learning from garbage data at episode start
        warmup_mask = (self.warmup_counter >= self.warmup_frames).float()
        reward = reward * warmup_mask
        r_text = r_text * warmup_mask
        r_teacher_motion = r_teacher_motion * warmup_mask
        
        return reward, r_text, r_teacher_motion
