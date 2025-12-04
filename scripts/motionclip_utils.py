import torch
import torch.nn.functional as F
import numpy as np
import clip
from typing import List, Optional

# Assuming MotionCLIP structure based on provided files
try:
    from src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
    from src.models.modeltype.motionclip import MOTIONCLIP
except ImportError:
    # If not in path, try adding it dynamically or warn
    print("[WARNING] MotionCLIP modules not found directly. Ensure MotionCLIP is in python path.")

class MotionCLIPWrapper:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.clip_model = None
        
    def load_model(self, checkpoint_path, config_path=None):
        """
        Load MotionCLIP model and CLIP model.
        checkpoint_path: Path to .pth.tar or .pth file
        """
        print(f"Loading MotionCLIP from {checkpoint_path}...")
        
        # Load CLIP first as it's needed for MotionCLIP initialization
        # Defaulting to ViT-B/32 as per MotionCLIP paper usually
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # Infer parameters if config not provided (or hardcode based on standard MotionCLIP)
        # This is a simplified parameter set, adjust based on actual config if needed
        # These params should match what was used in train_motionclip_joint.py or original
        params = {
            "njoints": 24, # Standard SMPL
            "nfeats": 6,   # rot6d
            "num_frames": 60,
            "num_classes": 1, # dummy
            "translation": True,
            "pose_rep": "rot6d",
            "glob": True,
            "glob_rot": True,
            "latent_dim": 512,
            "ff_size": 1024,
            "num_layers": 8,
            "num_heads": 4,
            "dropout": 0.1,
            "activation": "gelu",
            "ablation": None,
            "device": self.device,
            "outputxyz": False, # We only need encoder mostly for now, or decoder for latent
            "jointstype": "vertices", # or smpl
            "vertstrans": False
        }
        
        # If parameters are in checkpoint (often in 'parameters' or 'opt')
        if 'parameters' in ckpt:
            params.update(ckpt['parameters'])
        
        # Initialize model
        encoder = Encoder_TRANSFORMER(**params)
        decoder = Decoder_TRANSFORMER(**params)
        self.model = MOTIONCLIP(encoder, decoder, clip_model=self.clip_model, **params).to(self.device)
        
        # Load weights
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
            
        self.model.eval()
        print("MotionCLIP loaded successfully.")

    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """
        Encode text descriptions into MotionCLIP latent space (via CLIP text encoder).
        Note: MotionCLIP maps motion to CLIP space, so CLIP text embedding IS the target latent.
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(text_list).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens).float()
            # Normalize? MotionCLIP usually uses normalized features for cosine loss
            # But the VAE latent might not be normalized. 
            # In MotionCLIP paper: "We enforce the latent code z_motion to be similar to the CLIP representation of the text"
            # So usually raw CLIP output (or projected) is used.
            # Let's return raw CLIP features for now, as they align with Z_motion.
            return text_features

    def decode_motion(self, latent_z: torch.Tensor, duration: int = 60):
        """
        Decode latent vector to motion.
        """
        batch_size = latent_z.shape[0]
        lengths = torch.tensor([duration]*batch_size, device=self.device)
        mask = torch.ones((batch_size, duration), dtype=torch.bool, device=self.device)
        
        # Create dummy batch for decoder
        batch = {
            "z": latent_z,
            "y": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "mask": mask,
            "lengths": lengths
        }
        
        with torch.no_grad():
            output = self.model.decoder(batch)["output"] # (B, 60, J, C) or similar
        return output





