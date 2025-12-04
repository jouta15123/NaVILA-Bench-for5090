import argparse
import os
import sys
import random
import yaml
from pathlib import Path
from datetime import datetime

import inspect
from collections import namedtuple

# ---------------------------------------------------------------------------
# Compatibility patch for old libraries (e.g., chumpy) on Python 3.11+.
# chumpy still uses inspect.getargspec, which was removed in 3.11.
# We recreate it using inspect.getfullargspec so that smplx/chumpy work.
# ---------------------------------------------------------------------------
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _compat_getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shims for legacy NumPy usage in chumpy/smplx with NumPy 2.x
# ---------------------------------------------------------------------------
_legacy_numpy_types = [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
    ("unicode", str),
]
for _name, _type in _legacy_numpy_types:
    if not hasattr(np, _name):
        setattr(np, _name, _type)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

# MotionCLIP imports
REPO_ROOT = Path(__file__).resolve().parents[2]
HOYO_ROOT = REPO_ROOT / "hoyo_v1_1"
MOTIONCLIP_ROOT = REPO_ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

from src.models.get_model import get_model as motionclip_get_model

# Add project root so local package imports work when executed directly
sys.path.append(str(REPO_ROOT))

    
from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    encode_semantics_sarashina,
    encode_semantics_siglip,
    INSTRUCTION_ONOMATOPEIA,
    normalize_dataset,
    apply_normalization_from_stats,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

class Tee:
    """
    Simple tee for stdout/stderr so that all prints are saved to a log file
    while still appearing in the terminal.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        """Check if any of the streams is a TTY."""
        for s in self.streams:
            if hasattr(s, "isatty") and s.isatty():
                return True
        return False


# ---------------------------------------------------------------------------
# Coarse style groups for sanity-check experiments
# ---------------------------------------------------------------------------
# 速い系: すたすた / せかせか / てくてく
# 遅い系: 通常 / とぼとぼ / のろのろ
# 重い系: どっしどっし / のしのし
# ふらふら系: ぶらぶら / よたよた / よろよろ
COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["通常", "とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_LABELS = list(COARSE_GROUPS.keys())

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
    params["glob"] = True       # Global rotation (though unused in xyz mode, good for safety)
    params["translation"] = True # We include translation-like features (raw coords)
    
    # Only keep generic losses
    params["lambdas"] = {"rc": 1.0, "vel": 1.0} 
    params["clip_lambdas"] = {} # Disable CLIP losses
    
    # Initialize model with random weights first
    # os.chdir is removed as sys.path should handle imports correctly
    # and we want to avoid side effects on CWD.
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
        
    return model, params

def split_dataset(dataset: HoyoInstructionDataset, ratio: float = 0.8):
    """
    Splits the dataset into train and test sets by splitting samples in each label.
    Returns two dataset objects (sharing the same root but different samples).
    """
    import copy
    
    # Deepcopy to ensure independent sample lists
    # Note: is_train flag will be set individually
    train_ds = copy.copy(dataset)
    test_ds = copy.copy(dataset)
    
    # IMPORTANT: Set crop mode and disable augmentation for test
    train_ds.is_train = True
    test_ds.is_train = False
    test_ds.use_aug = False  # テストでは augmentation を明示的にオフ
    
    train_ds.samples_by_label = {}
    test_ds.samples_by_label = {}
    
    print(f"Splitting dataset (Train: {ratio:.0%}, Test: {1-ratio:.0%})...")
    
    for lab, samples in dataset.samples_by_label.items():
        n_total = len(samples)
        if n_total == 0:
            train_ds.samples_by_label[lab] = []
            test_ds.samples_by_label[lab] = []
            continue
            
        # Shuffle for random split
        shuffled = list(samples)
        random.shuffle(shuffled)
        
        n_train = int(n_total * ratio)
        # Ensure at least 1 sample in train if possible, unless empty
        if n_total > 0 and n_train == 0:
            n_train = 1
            
        train_samples = shuffled[:n_train]
        test_samples = shuffled[n_train:]
        
        train_ds.samples_by_label[lab] = train_samples
        test_ds.samples_by_label[lab] = test_samples
        
        print(f"  {lab}: {len(train_samples)} train, {len(test_samples)} test")

    # Rebuild indices for DataLoader support
    for ds in [train_ds, test_ds]:
        ds._indices = []
        # ds.label_to_id is inherited/copied, assuming labels didn't change
        for lab, samples in ds.samples_by_label.items():
            for idx in range(len(samples)):
                ds._indices.append((lab, idx))
        
    return train_ds, test_ds


def build_coarse_dataset(dataset: HoyoInstructionDataset) -> HoyoInstructionDataset:
    """
    Merge the 11 fine-grained labels into 4 coarse style groups.
    Returns a shallow copy of the dataset whose samples_by_label is regrouped.
    """
    import copy

    coarse_ds = copy.copy(dataset)
    coarse_ds.samples_by_label = {lab: [] for lab in COARSE_LABELS}

    for coarse_lab, fine_labs in COARSE_GROUPS.items():
        merged = []
        for fine_lab in fine_labs:
            merged.extend(dataset.samples_by_label.get(fine_lab, []))
        coarse_ds.samples_by_label[coarse_lab] = merged

    print("Built coarse dataset:")
    for lab in COARSE_LABELS:
        print(f"  {lab}: {len(coarse_ds.samples_by_label[lab])} samples")

    # Rebuild indices and label map for DataLoader support
    coarse_ds._indices = []
    coarse_ds.label_to_id = {lab: i for i, lab in enumerate(COARSE_LABELS)}
    for lab in COARSE_LABELS:
        samples = coarse_ds.samples_by_label[lab]
        for idx in range(len(samples)):
            coarse_ds._indices.append((lab, idx))

    return coarse_ds

@torch.no_grad()
def evaluate_dataset(
    dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_proj: nn.Module,
    sem_emb: torch.Tensor,
    device: torch.device,
    labels=None,
    batch_size: int = 64,
):
    """
    Evaluate the model on a dataset.
    
    IMPORTANT: Computes accuracy by counting correct predictions over all samples,
    NOT by averaging per-batch accuracies (which would be biased when batch sizes vary).
    """
    model.eval()
    sem_proj.eval()

    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)
    sem_emb = sem_emb.to(device)
    
    z_s_cls = sem_proj(sem_emb) 
    z_s_cls = F.normalize(z_s_cls, dim=-1)
    
    # FIX: Count correct predictions instead of averaging batch accuracies
    total_correct_1 = 0
    total_correct_3 = 0
    total_mpjpe = 0.0
    total_samples = 0
    
    # Use DataLoader for memory efficiency
    # dataset[i] returns (coords, label_id)
    # Shuffle=False for deterministic evaluation order
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for x_batch, y_batch in dataloader:
        # x_batch: (B, T, 14, 2)
        # y_batch: (B,)
        
        # Transpose to (B, J, C, T) for MotionCLIP
        # (B, T, 14, 2) -> (B, 14, 2, T)
        x_batch = x_batch.permute(0, 2, 3, 1).to(device)
        y_batch = y_batch.to(device)
        
        B, _, _, Tcur = x_batch.shape
        mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
        lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)
        
        batch_input = {
            "x": x_batch,
            "mask": mask,
            "lengths": lengths,
            "y": y_batch
        }
        
        out = model(batch_input)
        z_m = out["mu"]
        z_m = F.normalize(z_m, dim=-1)
        
        logits = z_m @ z_s_cls.t()
        
        preds = logits.argmax(dim=1)
        # FIX: Count correct predictions instead of computing batch accuracy
        total_correct_1 += (preds == y_batch).sum().item()
        
        if logits.shape[1] >= 3:
            _, top3 = logits.topk(3, dim=1)
            correct3 = top3.eq(y_batch.unsqueeze(1)).any(dim=1)
            total_correct_3 += correct3.sum().item()
        else:
            total_correct_3 += (preds == y_batch).sum().item()
            
        rec = out.get("output", out.get("rec", None))
        if rec is not None:
            diff = rec - x_batch
            # MPJPE is averaged over frames/joints, then summed over samples
            mpjpe = torch.norm(diff, dim=2).mean(dim=(1, 2)).sum().item()
        else:
            mpjpe = 0.0
            
        total_mpjpe += mpjpe
        total_samples += B
        
    if total_samples == 0:
        return {"acc@1": 0.0, "acc@3": 0.0, "mpjpe": 0.0}
        
    return {
        "acc@1": total_correct_1 / total_samples,
        "acc@3": total_correct_3 / total_samples,
        "mpjpe": total_mpjpe / total_samples,
    }


@torch.no_grad()
def dump_latent_snapshot(
    train_dataset: HoyoInstructionDataset,
    test_dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_proj: nn.Module,
    sem_emb: torch.Tensor,
    device: torch.device,
    out_path: Path,
    labels=None,
    split_modes=("test",),
    max_per_label: int = 50,
):
    """
    Collects a small subset of motion latents and their labels.
    """
    model.eval()

    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    sem_emb = sem_emb.to(device)

    # Class prototypes in motion latent space
    z_s_cls = sem_proj(sem_emb)
    z_s_cls = F.normalize(z_s_cls, dim=-1).detach().cpu().numpy()

    zs = []
    ys = []
    splits = []

    split_modes = tuple(split_modes) if split_modes else ("test",)

    def _collect_from_dataset(ds: HoyoInstructionDataset, split_name: str):
        if split_name not in split_modes:
            return

        original_is_train = ds.is_train
        original_use_aug = getattr(ds, "use_aug", False)
        ds.is_train = False
        ds.use_aug = False

        idx_to_label = {idx: lab for lab, idx in ds.label_to_id.items()}
        per_label_counts = [0 for _ in labels]

        collected = 0
        for data_idx in range(len(ds)):
            coords, label_id = ds[data_idx]
            lab_name = idx_to_label.get(int(label_id))
            if lab_name is None:
                continue
            if lab_name not in label_to_idx:
                continue

            lab_idx = label_to_idx[lab_name]
            if per_label_counts[lab_idx] >= max_per_label:
                continue

            coords = coords[np.newaxis, ...].transpose(0, 2, 3, 1)
            x = torch.from_numpy(coords).to(device)

            B, _, _, Tcur = x.shape
            mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
            lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)

            batch = {
                "x": x,
                "mask": mask,
                "lengths": lengths,
                "y": torch.zeros((B,), dtype=torch.long, device=device),
            }

            out = model(batch)
            z_m = out["mu"]
            z_m = F.normalize(z_m, dim=-1).squeeze(0).detach().cpu().numpy()

            zs.append(z_m)
            ys.append(lab_idx)
            splits.append(split_name)

            per_label_counts[lab_idx] += 1
            collected += 1

            # Early exit if we already have the quota for every label
            if all(count >= min(max_per_label, len(ds.samples_by_label.get(lbl, []))) for count, lbl in zip(per_label_counts, labels)):
                break

        ds.is_train = original_is_train
        ds.use_aug = original_use_aug

        print(f"  Collected {collected} samples from split='{split_name}'")

    _collect_from_dataset(train_dataset, "train")
    _collect_from_dataset(test_dataset, "test")

    if not zs:
        print("No samples available to dump latent snapshot.")
        return

    z_m_all = np.stack(zs, axis=0)
    labels_idx = np.asarray(ys, dtype=np.int64)
    splits_arr = np.asarray(splits)
    label_list = np.asarray(labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        z_m=z_m_all,
        labels_idx=labels_idx,
        splits=splits_arr,
        label_list=label_list,
        z_s_cls=z_s_cls,
    )
    print(f"Saved latent snapshot with {len(z_m_all)} points to {out_path}")

def train_joint(
    train_dataset: HoyoInstructionDataset,
    test_dataset: HoyoInstructionDataset,
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
    eval_interval: int = 200,
    lambda_contrastive: float = 1.0,
    lambda_vae: float = 1.0,
    stage: str = "freeze",
    contrastive_mode: str = "supcon",
    labels=None,
    vis_max_per_label: int = 50,
    snapshot_splits=("test",),
    wandb_run=None,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Projector for semantics -> motion latent space
    # MotionCLIP latent dim is 512 (default)
    d_motion = model.latent_dim
    d_sem = sem_emb.shape[1]
    
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))

    # Optionally initialize from previous stage checkpoints (e.g., freeze -> encoder -> full)
    sem_proj_ckpt = ckpt_dir / "sem_proj_joint_best.pth"
    logit_scale_ckpt = ckpt_dir / "logit_scale_joint_best.pt"
    encoder_ckpt = ckpt_dir / "motionclip_encoder_joint_best.pth"
    full_model_ckpt = ckpt_dir / "motionclip_full_joint_best.pth"
    
    if stage in ("encoder", "full"):
        # sem_proj と logit_scale をロード
        if sem_proj_ckpt.exists():
            try:
                sem_state = torch.load(sem_proj_ckpt, map_location=device)
                sem_proj.load_state_dict(sem_state)
                print(f"[Init] Loaded sem_proj from {sem_proj_ckpt}")
            except Exception as e:
                print(f"[Init] Failed to load sem_proj from {sem_proj_ckpt}: {e}")
        if logit_scale_ckpt.exists():
            try:
                ls_state = torch.load(logit_scale_ckpt, map_location=device)
                if isinstance(ls_state, dict) and "logit_scale" in ls_state:
                    logit_scale.data.copy_(ls_state["logit_scale"].to(device))
                elif torch.is_tensor(ls_state):
                    logit_scale.data.copy_(ls_state.to(device))
                print(f"[Init] Loaded logit_scale from {logit_scale_ckpt}")
            except Exception as e:
                print(f"[Init] Failed to load logit_scale from {logit_scale_ckpt}: {e}")
        
        # MotionCLIP 側も前段の best からロード（段階的 fine-tuning 用）
        if stage == "full" and full_model_ckpt.exists():
            try:
                model.load_state_dict(torch.load(full_model_ckpt, map_location=device))
                print(f"[Init] Loaded full model from {full_model_ckpt}")
            except Exception as e:
                print(f"[Init] Failed to load full model from {full_model_ckpt}: {e}")
        elif encoder_ckpt.exists():
            try:
                model.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
                print(f"[Init] Loaded encoder from {encoder_ckpt}")
            except Exception as e:
                print(f"[Init] Failed to load encoder from {encoder_ckpt}: {e}")

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
    
    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)
    sem_emb = sem_emb.to(device)
    
    loss_history = []
    best_test_acc = -1.0
    
    # Running stats for readable logging
    running_stats = {
        "total_loss": 0.0,
        "vae_loss": 0.0,
        "contrastive_loss": 0.0,
        "train_acc1": 0.0,
        "steps": 0,
    }
    
    print(f"Start Joint Training for {steps} steps... (stage={stage})")
    print(
        f"  Config: batch_size={batch_size}, lr={lr}, lr_encoder={lr_encoder}, "
        f"lr_decoder={lr_decoder}, lambda_vae={lambda_vae}, "
        f"lambda_contrastive={lambda_contrastive}, temp={temp}"
    )

    # Calculate weights for WeightedRandomSampler to handle class imbalance
    # _indices に出てくるラベルだけでカウントしてロバストに
    label_counts = {}
    for (lab, _) in train_dataset._indices:
        label_counts[lab] = label_counts.get(lab, 0) + 1
    weights = [1.0 / label_counts[lab] for (lab, _) in train_dataset._indices]
    
    # Create DataLoader
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0)
    data_iterator = iter(dataloader)
    
    for step in range(1, steps + 1):
        # stage=freeze では MotionCLIP を推論モードのまま保つ（BN/Dropout の挙動を固定）
        if stage == "freeze":
            model.eval()
        else:
            model.train()
        
        # 1. Get Batch
        try:
            x_batch, y_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            x_batch, y_batch = next(data_iterator)
            
        # Transpose: (B, T, 14, 2) -> (B, 14, 2, T)
        x_batch = x_batch.permute(0, 2, 3, 1).to(device)
        y_batch = y_batch.to(device)
        
        B, _, _, Tcur = x_batch.shape
        mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
        lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)
        
        batch = {
            "x": x_batch,
            "mask": mask,
            "lengths": lengths,
            "y": torch.zeros((B,), dtype=torch.long, device=device) # Dummy class for VAE
        }
        
        # 2. Forward (VAE)
        batch = model(batch)
        
        # 3. VAE Loss
        vae_loss, _ = model.compute_loss(batch)
        
        # 4. Contrastive Loss
        z_m = batch["mu"]
        z_m = F.normalize(z_m, dim=-1)
        B = z_m.shape[0]
        
        z_s_cls = sem_proj(sem_emb)
        z_s_cls = F.normalize(z_s_cls, dim=-1)
        z_s_inst = z_s_cls[y_batch]

        # ------------------------------------------------------------------
        # Contrastive loss
        #   - "supcon": existing supervised contrastive over motion/text
        #   - "clip_ce": CLIP-style CE (motion → semantic prototypes)
        # ------------------------------------------------------------------
        if contrastive_mode == "supcon":
            # Supervised Contrastive
            features = torch.cat([z_m, z_s_inst], dim=0)
            features = F.normalize(features, dim=-1)
            labels_ext = torch.cat([y_batch, y_batch], dim=0)
            
            N = features.shape[0]
            sim_matrix = torch.matmul(features, features.t())
            sim_matrix = torch.exp(logit_scale) * sim_matrix
            
            logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
            logits = sim_matrix - logits_max
            exp_logits = torch.exp(logits)
            mask_self = torch.eye(N, dtype=torch.bool, device=device)
            exp_logits = exp_logits * (~mask_self)
            
            labels_eq = labels_ext.unsqueeze(0) == labels_ext.unsqueeze(1)
            # Treat all pairs with the same label (excluding self) as positives,
            # regardless of modality (motion-motion, text-text, motion-text).
            pos_mask = labels_eq & (~mask_self)
            
            denom = exp_logits.sum(dim=1, keepdim=True)
            pos_exp = exp_logits * pos_mask
            pos_sum = pos_exp.sum(dim=1)
            
            valid = pos_sum > 0
            loss_vals = torch.zeros_like(pos_sum)
            loss_vals[valid] = -torch.log(pos_sum[valid] / (denom[valid].squeeze(1) + 1e-8))
            cont_loss = loss_vals[valid].mean()
            
            # Simple classification accuracy on the current batch (motion → semantics)
            with torch.no_grad():
                logits_cls = z_m @ z_s_cls.t()
                preds_train = logits_cls.argmax(dim=1)
                train_acc1 = (preds_train == y_batch).float().mean().item()
            
                # Similarity statistics for positives / negatives (for debugging)
                sim_detached = sim_matrix.detach()
                pos_sims = sim_detached[pos_mask]
                neg_sims = sim_detached[(~pos_mask) & (~mask_self)]
                pos_sim_mean = float(pos_sims.mean().item()) if pos_sims.numel() > 0 else 0.0
                neg_sim_mean = float(neg_sims.mean().item()) if neg_sims.numel() > 0 else 0.0
                num_pos = int(pos_sims.numel())
                num_neg = int(neg_sims.numel())
                # Temperature is the inverse of the logit scale
                current_temp = float(torch.exp(-logit_scale.detach()).item())

        elif contrastive_mode == "clip_ce":
            # CLIP-style cross-entropy: motion → semantic prototypes
            logits_cls = torch.exp(logit_scale) * (z_m @ z_s_cls.t())
            cont_loss = F.cross_entropy(logits_cls, y_batch)

            with torch.no_grad():
                preds_train = logits_cls.argmax(dim=1)
                train_acc1 = (preds_train == y_batch).float().mean().item()
                current_temp = float(torch.exp(-logit_scale.detach()).item())
                # We do not compute detailed pos/neg stats in this mode.
                pos_sim_mean = 0.0
                neg_sim_mean = 0.0
                num_pos = 0
                num_neg = 0
        else:
            raise ValueError(f"Unknown contrastive_mode '{contrastive_mode}'")
        
        # 5. Total Loss
        vae_weight = 0.0 if stage == "freeze" else lambda_vae
        total_loss = vae_weight * vae_loss + lambda_contrastive * cont_loss
        
        if torch.isnan(total_loss):
            print(f"[Step {step}] Loss is NaN! Reverting to previous best and stopping.")
            break
            
        optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients over all trainable parameters, including sem_proj / logit_scale.
        all_params = []
        for group in optimizer.param_groups:
            all_params.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(all_params, 0.5)
        optimizer.step()
        # Clamp logit_scale to keep temperature in a reasonable range.
        # Temp = exp(-logit_scale). Range [0.01, 100] -> logit_scale [-4.6, 4.6]
        # Default temp 0.07 is ~2.66. Previous clamp [log(0.1), log(10)] = [-2.3, 2.3] clipped it.
        with torch.no_grad():
            logit_scale.data.clamp_(np.log(1.0 / 100.0), np.log(1.0 / 0.01))
        
        loss_history.append(total_loss.item())

        # Update running stats
        running_stats["total_loss"] += float(total_loss.item())
        running_stats["vae_loss"] += float(vae_loss.item())
        running_stats["contrastive_loss"] += float(cont_loss.item())
        running_stats["train_acc1"] += float(train_acc1)
        running_stats["steps"] += 1
        
        # --- Logging ---
        if step % log_interval == 0 or step == 1:
            denom = max(running_stats["steps"], 1)
            avg_total = running_stats["total_loss"] / denom
            avg_vae = running_stats["vae_loss"] / denom
            avg_cont = running_stats["contrastive_loss"] / denom
            avg_acc1 = running_stats["train_acc1"] / denom

            print(
                f"[Step {step:05d}] "
                f"loss_total={avg_total:.4f} (vae={avg_vae:.4f}, cont={avg_cont:.4f}), "
                f"train_acc@1={avg_acc1:.3f}, B={B}, temp={current_temp:.4f}, "
                f"sim(pos={pos_sim_mean:.3f}, neg={neg_sim_mean:.3f}, "
                f"n_pos={num_pos}, n_neg={num_neg})"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/total_loss": avg_total,
                        "train/vae_loss": avg_vae,
                        "train/contrastive_loss": avg_cont,
                        "train/train_acc_top1": avg_acc1,
                        "train/step": step,
                    },
                    step=step,
                )

            # reset running stats after each log
            running_stats = {
                "total_loss": 0.0,
                "vae_loss": 0.0,
                "contrastive_loss": 0.0,
                "train_acc1": 0.0,
                "steps": 0,
            }
                
        # --- Evaluation ---
        if step % eval_interval == 0 or step == steps:
            print(f"Evaluating on Test Set...")
            metrics = evaluate_dataset(test_dataset, model, sem_proj, sem_emb, device, labels=labels)
            
            test_acc = metrics["acc@1"]
            print(f"  [TEST] Acc@1: {metrics['acc@1']:.3f} | Acc@3: {metrics['acc@3']:.3f} | MPJPE: {metrics['mpjpe']:.4f}")
            
            if wandb_run is not None:
                wandb_run.log({
                    "test/acc_top1": metrics["acc@1"],
                    "test/acc_top3": metrics["acc@3"],
                    "test/mpjpe": metrics["mpjpe"],
                }, step=step)
            
            # Save BEST model based on Test Acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"  >>> New Best Test Acc: {best_test_acc:.3f}! Saving model...")
                torch.save(model.encoder.state_dict(), ckpt_dir / "motionclip_encoder_joint_best.pth")
                torch.save(sem_proj.state_dict(), ckpt_dir / "sem_proj_joint_best.pth")
                torch.save({"logit_scale": logit_scale.detach().cpu()}, ckpt_dir / "logit_scale_joint_best.pt")
                torch.save(model.state_dict(), ckpt_dir / "motionclip_full_joint_best.pth")

        # Periodic Save (Backup)
        if step % 1000 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"motionclip_full_step{step}.pth")

    # Final Save
    print(f"Saving final models to {ckpt_dir}")
    torch.save(model.encoder.state_dict(), ckpt_dir / "motionclip_encoder_joint_final.pth")
    torch.save(sem_proj.state_dict(), ckpt_dir / "sem_proj_joint_final.pth")
    torch.save(model.state_dict(), ckpt_dir / "motionclip_full_joint_final.pth")

    # Save a compact latent snapshot for later visualization
    snapshot_path = out_dir / "latent_snapshot_final.npz"
    dump_latent_snapshot(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        sem_proj=sem_proj,
        sem_emb=sem_emb,
        device=device,
        labels=labels,
        out_path=snapshot_path,
        split_modes=snapshot_splits,
        max_per_label=vis_max_per_label,
    )
    
    return model, sem_proj

def parse_args():
    parser = argparse.ArgumentParser(description="Joint training for HOYO + MotionCLIP")
    parser.add_argument("--stage", choices=["freeze", "encoder", "full"], default="freeze", help="Training stage strategy")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["fine", "coarse"],
        default="fine",
        help="Label granularity: 'fine' (11オノマトペ) or 'coarse' (4スタイル群)",
    )
    parser.add_argument(
        "--contrastive-mode",
        choices=["supcon", "clip_ce"],
        default="supcon",
        help="Type of contrastive loss: 'supcon' (existing supervised contrastive) "
        "or 'clip_ce' (CLIP-style CE: motion → semantic prototypes).",
    )
    parser.add_argument(
        "--sem-encoder",
        choices=["sarashina", "siglip"],
        default="sarashina",
        help="Which semantic encoder to use for text embeddings.",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-decoder", type=float, default=1e-5)
    parser.add_argument("--temp", type=float, default=0.07, help="Contrastive temperature")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--eval-interval", type=int, default=200, help="Evaluation interval (steps)")
    parser.add_argument("--lambda-contrastive", type=float, default=0.1)
    parser.add_argument(
        "--lambda-vae",
        type=float,
        default=1.0,
        help="Weight for VAE loss. Set to 0.0 to train with contrastive loss only (like old script).",
    )
    parser.add_argument(
        "--vis-max-per-label",
        type=int,
        default=50,
        help="Maximum number of samples per label (per split) to include in the latent snapshot.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hoyo_motion", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for the run directory")
    parser.add_argument("--use-aug", action="store_true", help="Enable data augmentation (flip, rotation) during training")
    parser.add_argument(
        "--snapshot-splits",
        type=str,
        default="test",
        help="Comma-separated list of dataset splits to include when saving latent snapshots (e.g., 'test' or 'train,test').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    hoyo_root = HOYO_ROOT
    if args.run_name:
        run_name = args.run_name
    else:
        # デフォルトでタイムスタンプを付けて、実験結果の上書きを防ぐ
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = hoyo_root / "joint_training_results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Set up logging: tee stdout/stderr to a timestamped log file
    # ------------------------------------------------------------------
    log_path = logs_dir / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"[Logger] Writing training logs to: {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    target_len = 60
    
    # Load Data (fine-grained labels)
    dataset = HoyoInstructionDataset(
        hoyo_root, 
        INSTRUCTION_ONOMATOPEIA, 
        target_len=target_len, 
        is_train=True, 
        use_aug=args.use_aug
    )
    
    # Split Data (80% Train, 20% Test)
    train_dataset, test_dataset = split_dataset(dataset, ratio=0.8)
    
    # Semantics (fine labels first)
    if args.sem_encoder == "sarashina":
        print("Using Sarashina sentence embeddings for semantics.")
        sem_emb_fine = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)
    elif args.sem_encoder == "siglip":
        print("Using SigLIP text encoder for semantics.")
        sem_emb_fine = encode_semantics_siglip(INSTRUCTION_ONOMATOPEIA, device=device)
    else:
        raise ValueError(f"Unknown sem-encoder: {args.sem_encoder}")
    
    # Normalize TRAIN data and persist stats, then normalize TEST with the same stats
    stats_path = out_dir / "normalization_stats.json"
    normalize_dataset(train_dataset, stats_path)
    apply_normalization_from_stats(test_dataset, stats_path)

    # Optionally convert to coarse style groups
    if args.label_mode == "coarse":
        print("Using COARSE style labels:")
        for coarse, fines in COARSE_GROUPS.items():
            print(f"  {coarse}: {', '.join(fines)}")

        # Regroup datasets
        train_dataset = build_coarse_dataset(train_dataset)
        test_dataset = build_coarse_dataset(test_dataset)

        # Build coarse semantic embeddings by averaging member embeddings
        label_to_idx = {lab: i for i, lab in enumerate(INSTRUCTION_ONOMATOPEIA)}
        coarse_vecs = []
        for coarse_lab in COARSE_LABELS:
            fine_labs = COARSE_GROUPS[coarse_lab]
            idxs = [label_to_idx[fl] for fl in fine_labs]
            coarse_vecs.append(sem_emb_fine[idxs].mean(dim=0))
        sem_emb = torch.stack(coarse_vecs, dim=0)
        labels_for_training = COARSE_LABELS
    else:
        sem_emb = sem_emb_fine
        labels_for_training = INSTRUCTION_ONOMATOPEIA
    
    # Model
    model, params = load_motionclip_full_model(device, target_len)
    
    # Optional Weights & Biases logging
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed in this environment. Please install it or run without --wandb.")
        wandb_config = {
            "stage": args.stage,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "label_mode": args.label_mode,
            "sem_encoder": args.sem_encoder,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "lr_decoder": args.lr_decoder,
            "lambda_contrastive": args.lambda_contrastive,
            "lambda_vae": args.lambda_vae,
            "temp": args.temp,
            "log_interval": args.log_interval,
            "eval_interval": args.eval_interval,
            "vis_max_per_label": args.vis_max_per_label,
            "target_len": target_len,
            "device": str(device),
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=wandb_config,
        )
    
    try:
        # Train
        snapshot_splits = tuple([s.strip() for s in args.snapshot_splits.split(",") if s.strip()])
        if not snapshot_splits:
            snapshot_splits = ("test",)
        train_joint(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            sem_emb=sem_emb,
            device=device,
            out_dir=out_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            temp=args.temp,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            lr_decoder=args.lr_decoder,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            lambda_contrastive=args.lambda_contrastive,
            lambda_vae=args.lambda_vae,
            stage=args.stage,
            contrastive_mode=args.contrastive_mode,
            labels=labels_for_training,
            vis_max_per_label=args.vis_max_per_label,
            snapshot_splits=snapshot_splits,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
