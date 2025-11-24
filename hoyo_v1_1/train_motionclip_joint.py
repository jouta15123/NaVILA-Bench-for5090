import argparse
import os
import sys
import json
import random
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

# MotionCLIP imports
ROOT = Path(__file__).resolve().parents[1]
MOTIONCLIP_ROOT = ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

from src.models.get_model import get_model as motionclip_get_model
from src.models.tools.losses import get_loss_function

# Add project root to path so we can import as package if needed, or just import local file
sys.path.append(str(ROOT))

# Re-use HoyoInstructionDataset and logic from previous script
try:
    from hoyo_v1_1.hoyo_sem_motion_contrastive_motionclip import (
        HoyoInstructionDataset,
        encode_semantics_sarashina,
        INSTRUCTION_ONOMATOPEIA
    )
except ImportError:
    # Fallback for when running directly inside the folder
    from hoyo_sem_motion_contrastive_motionclip import (
        HoyoInstructionDataset,
        encode_semantics_sarashina,
        INSTRUCTION_ONOMATOPEIA
    )

def load_motionclip_full_model(device: torch.device, target_len: int = 60):
    """
    Loads the full MotionCLIP model (encoder + decoder).
    Re-initializes input/output layers for HOYO dimensions (14 joints, 2 coords).
    """
    opt_path = MOTIONCLIP_ROOT / "exps" / "paper-model" / "opt.yaml"
    ckpt_path = MOTIONCLIP_ROOT / "exps" / "paper-model" / "checkpoint_0100.pth.tar"
    
    with open(opt_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    params = dict(cfg)
    params["device"] = device
    params["njoints"] = 14
    params["nfeats"] = 2
    params["num_frames"] = target_len
    params["num_classes"] = 1
    
    # Adjust for HOYO (2D raw coords)
    params["pose_rep"] = "xyz"  # Treat as raw coordinates
    params["outputxyz"] = False # Don't try to convert to SMPL mesh
    
    # Only keep generic losses
    params["lambdas"] = {"rc": 1.0, "vel": 1.0} 
    params["clip_lambdas"] = {} # Disable CLIP losses
    
    cwd = os.getcwd()
    os.chdir(str(MOTIONCLIP_ROOT))
    try:
        # Initialize model with random weights first
        model = motionclip_get_model(params, clip_model=None)
        
        # Load pretrained weights where possible
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Filter out incompatible layers
        # encoder.skelEmbedding: input projection
        # decoder.finallayer: output projection
        filtered = {
            k: v for k, v in state_dict.items()
            if not (k.startswith("encoder.skelEmbedding.") or k.startswith("decoder.finallayer."))
        }
        
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded MotionCLIP (Full). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
    finally:
        os.chdir(cwd)
        
    return model, params

def train_joint(
    dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_emb: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    steps: int = 4000,
    batch_size: int = 32,
    temp: float = 0.07,
    lr: float = 1e-4,
    lr_encoder: float = 1e-5,
    lr_decoder: float = 1e-5,
    log_interval: int = 100,
    lambda_contrastive: float = 1.0,
    stage: str = "freeze",
):
    os.makedirs(out_dir, exist_ok=True)
    
    # Projector for semantics -> motion latent space
    # MotionCLIP latent dim is 512 (default)
    d_motion = model.latent_dim
    d_sem = sem_emb.shape[1]
    
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))
    
    # Stage-aware freezing
    trainable_groups = []
    if stage == "freeze":
        model.eval()
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.decoder.parameters():
            p.requires_grad = False
        trainable_groups = [
            {"params": sem_proj.parameters(), "lr": lr},
            {"params": [logit_scale], "lr": lr},
        ]
    elif stage == "encoder":
        model.train()
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = True
        trainable_groups = [
            {"params": sem_proj.parameters(), "lr": lr},
            {"params": [logit_scale], "lr": lr},
            {"params": model.encoder.parameters(), "lr": lr_encoder},
        ]
    elif stage == "full":
        model.train()
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
        trainable_groups = [
            {"params": sem_proj.parameters(), "lr": lr},
            {"params": [logit_scale], "lr": lr},
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.decoder.parameters(), "lr": lr_decoder},
        ]
    else:
        raise ValueError(f"Unknown stage '{stage}'")

    optimizer = torch.optim.AdamW(trainable_groups)
    
    labels = INSTRUCTION_ONOMATOPEIA
    sem_emb = sem_emb.to(device)
    
    loss_history = []
    
    print(f"Start Joint Training for {steps} steps... (stage={stage})")
    
    for step in range(1, steps + 1):
        # 1. Prepare Batch
        xs = []
        ys = []
        
        # Simple random sampling
        # Ideally we should use a DataLoader but this is lightweight
        for _ in range(batch_size):
            lab_idx = random.randint(0, len(labels) - 1)
            lab = labels[lab_idx]
            samples = dataset.samples_by_label[lab]
            if not samples: continue
            arr = random.choice(samples) # (T, 14, 2)
            xs.append(arr)
            ys.append(lab_idx)
            
        if not xs: continue
            
        coords = np.stack(xs, axis=0) # (B, T, 14, 2)
        coords = coords.transpose(0, 2, 3, 1) # (B, 14, 2, T) -> (B, J, C, T)
        
        x_batch = torch.from_numpy(coords).to(device)
        y_batch = torch.tensor(ys, dtype=torch.long, device=device)
        
        B, _, _, Tcur = x_batch.shape
        mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
        lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)
        
        batch = {
            "x": x_batch,
            "mask": mask,
            "lengths": lengths,
            "y": torch.zeros((B,), dtype=torch.long, device=device) # Dummy class
        }
        
        # 2. Forward (VAE)
        # model.forward computes encoder -> z -> decoder
        batch = model(batch)
        
        # 3. VAE Loss
        vae_loss, losses_detail = model.compute_loss(batch)
        # losses_detail contains 'rc', 'vel', etc.
        
        # 4. Contrastive Loss (Motion Latent <-> Semantics)
        z_m = batch["mu"] # (B, Dm) Use mean of VAE
        z_m = F.normalize(z_m, dim=-1)
        
        z_s_cls = sem_proj(sem_emb) # (11, Dm)
        z_s_cls = F.normalize(z_s_cls, dim=-1)
        z_s_inst = z_s_cls[y_batch] # (B, Dm)
        
        # Supervised Contrastive
        features = torch.cat([z_m, z_s_inst], dim=0) # (2B, Dm)
        features = F.normalize(features, dim=-1)
        labels_ext = torch.cat([y_batch, y_batch], dim=0) # (2B,)
        
        N = features.shape[0]
        sim_matrix = torch.matmul(features, features.t())
        sim_matrix = torch.exp(logit_scale) * sim_matrix
        
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max
        exp_logits = torch.exp(logits)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        exp_logits = exp_logits * (~mask_self)
        
        labels_eq = labels_ext.unsqueeze(0) == labels_ext.unsqueeze(1)
        pos_mask = labels_eq & (~mask_self)
        
        denom = exp_logits.sum(dim=1, keepdim=True)
        pos_exp = exp_logits * pos_mask
        pos_sum = pos_exp.sum(dim=1)
        
        valid = pos_sum > 0
        loss_vals = torch.zeros_like(pos_sum)
        loss_vals[valid] = -torch.log(pos_sum[valid] / (denom[valid].squeeze(1) + 1e-8))
        cont_loss = loss_vals[valid].mean()
        
        # 5. Total Loss
        total_loss = vae_loss + lambda_contrastive * cont_loss
        
        if torch.isnan(total_loss):
            print(f"[Step {step}] Loss is NaN! Reverting to previous best and stopping.")
            break
            
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        # Save best model (or just periodic)
        if step % 500 == 0:
             torch.save(model.state_dict(), out_dir / f"motionclip_full_step{step}.pth")
        
        if step % log_interval == 0 or step == 1:
            # Calculate Acc
            with torch.no_grad():
                logits_m2s = z_m @ z_s_cls.t()
                preds_m2s = logits_m2s.argmax(dim=1)
                acc = (preds_m2s == y_batch).float().mean().item()
            
            print(f"[Step {step:04d}] Total: {total_loss.item():.4f} | VAE: {vae_loss.item():.4f} | Cont: {cont_loss.item():.4f} | Acc: {acc:.3f}")
            
    # Save artifacts
    print(f"Saving models to {out_dir}")
    torch.save(model.encoder.state_dict(), out_dir / "motionclip_encoder_joint.pth")
    torch.save(sem_proj.state_dict(), out_dir / "sem_proj_joint.pth")
    torch.save(model.state_dict(), out_dir / "motionclip_full_joint.pth")
    
    return model, sem_proj

def parse_args():
    parser = argparse.ArgumentParser(description="Joint training for HOYO + MotionCLIP")
    parser.add_argument("--stage", choices=["freeze", "encoder", "full"], default="freeze", help="Training stage strategy")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-decoder", type=float, default=1e-5)
    parser.add_argument("--lambda-contrastive", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    hoyo_root = ROOT / "hoyo_v1_1"
    out_dir = hoyo_root / "joint_training_results"
    target_len = 60
    
    # Load Data
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)
    
    # Semantics
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)
    
    # Normalize Data
    all_samples = []
    for lab in INSTRUCTION_ONOMATOPEIA:
        all_samples.extend(dataset.samples_by_label[lab])
    
    all_data = np.stack(all_samples, axis=0) # (N, T, 14, 2)
    data_mean = all_data.mean(axis=(0, 1, 2))
    data_std = all_data.std(axis=(0, 1, 2)) + 1e-6
    
    print(f"Data Mean: {data_mean}, Std: {data_std}")
    
    # Apply normalization
    for lab in INSTRUCTION_ONOMATOPEIA:
        new_samples = []
        for arr in dataset.samples_by_label[lab]:
            norm_arr = (arr - data_mean) / data_std
            new_samples.append(norm_arr)
        dataset.samples_by_label[lab] = new_samples
        
    # Save normalization stats
    stats = {"mean": data_mean.tolist(), "std": data_std.tolist()}
    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f)
        
    # Model
    model, params = load_motionclip_full_model(device, target_len)
    
    # Train
    train_joint(
        dataset,
        model,
        sem_emb,
        device,
        out_dir=out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        lambda_contrastive=args.lambda_contrastive,
        stage=args.stage,
    )

if __name__ == "__main__":
    main()

