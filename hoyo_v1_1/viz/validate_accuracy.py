#!/usr/bin/env python3
"""
評価精度の検証スクリプト
バグがあった評価コードを修正して、正しい精度を計算する
"""

import sys
import random
import inspect
from pathlib import Path
from collections import namedtuple

# ---------------------------------------------------------------------------
# Compatibility patch for old libraries (e.g., chumpy) on Python 3.11+.
# ---------------------------------------------------------------------------
if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _compat_getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]

import numpy as np

# NumPy 2.x compatibility
_legacy_numpy_types = [
    ("bool", bool), ("int", int), ("float", float),
    ("complex", complex), ("object", object), ("str", str), ("unicode", str),
]
for _name, _type in _legacy_numpy_types:
    if not hasattr(np, _name):
        setattr(np, _name, _type)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# MotionCLIP imports
REPO_ROOT = Path(__file__).resolve().parents[2]
HOYO_ROOT = REPO_ROOT / "hoyo_v1_1"
MOTIONCLIP_ROOT = REPO_ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))
sys.path.append(str(REPO_ROOT))

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    encode_semantics_sarashina,
    INSTRUCTION_ONOMATOPEIA,
    normalize_dataset,
    apply_normalization_from_stats,
)


def load_motionclip_and_checkpoints(device, ckpt_dir, target_len=60):
    """モデルとチェックポイントをロード"""
    import yaml
    from src.models.get_model import get_model as motionclip_get_model
    
    opt_path = MOTIONCLIP_ROOT / "exps" / "paper-model" / "opt.yaml"
    with open(opt_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    params = dict(cfg)
    params["device"] = device
    params["njoints"] = 14
    params["nfeats"] = 2
    params["num_frames"] = target_len
    params["num_classes"] = 1
    params["pose_rep"] = "xyz"
    params["outputxyz"] = False
    params["glob"] = True
    params["translation"] = True
    params["lambdas"] = {"rc": 1.0, "vel": 1.0}
    params["clip_lambdas"] = {}
    
    model = motionclip_get_model(params, clip_model=None)
    
    # Load checkpoints
    model_ckpt = ckpt_dir / "motionclip_full_joint_best.pth"
    sem_proj_ckpt = ckpt_dir / "sem_proj_joint_best.pth"
    
    if model_ckpt.exists():
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
        print(f"Loaded model from {model_ckpt}")
    
    # Create sem_proj - get dim from checkpoint
    d_motion = model.latent_dim
    sem_proj_state = torch.load(sem_proj_ckpt, map_location=device)
    d_sem = sem_proj_state['weight'].shape[1]
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    
    if sem_proj_ckpt.exists():
        sem_proj.load_state_dict(torch.load(sem_proj_ckpt, map_location=device))
        print(f"Loaded sem_proj from {sem_proj_ckpt}")
    
    return model, sem_proj


@torch.no_grad()
def evaluate_correct(
    dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_proj: nn.Module,
    sem_emb: torch.Tensor,
    device: torch.device,
    labels=None,
    batch_size: int = 64,
):
    """正しい精度計算（全サンプルベース）"""
    model.eval()

    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)
    sem_emb = sem_emb.to(device)
    
    z_s_cls = sem_proj(sem_emb)
    z_s_cls = F.normalize(z_s_cls, dim=-1)
    
    # 修正版：正解数とサンプル数をカウント
    total_correct_1 = 0
    total_correct_3 = 0
    total_samples = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for x_batch, y_batch in dataloader:
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
        correct_1 = (preds == y_batch).sum().item()
        
        _, top3 = logits.topk(3, dim=1)
        correct_3 = top3.eq(y_batch.unsqueeze(1)).any(dim=1).sum().item()
        
        total_correct_1 += correct_1
        total_correct_3 += correct_3
        total_samples += B
        
    return {
        "acc@1": total_correct_1 / total_samples,
        "acc@3": total_correct_3 / total_samples,
        "total_samples": total_samples,
        "correct_1": total_correct_1,
        "correct_3": total_correct_3,
    }


@torch.no_grad()
def evaluate_old_method(
    dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_proj: nn.Module,
    sem_emb: torch.Tensor,
    device: torch.device,
    labels=None,
    batch_size: int = 64,
):
    """元の（バグのある）評価方法"""
    model.eval()

    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)
    sem_emb = sem_emb.to(device)
    
    z_s_cls = sem_proj(sem_emb)
    z_s_cls = F.normalize(z_s_cls, dim=-1)
    
    total_acc_1 = 0.0
    total_acc_3 = 0.0
    valid_batches = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for x_batch, y_batch in dataloader:
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
        acc1 = (preds == y_batch).float().mean().item()
        
        _, top3 = logits.topk(3, dim=1)
        correct3 = top3.eq(y_batch.unsqueeze(1)).any(dim=1)
        acc3 = correct3.float().mean().item()
        
        total_acc_1 += acc1
        total_acc_3 += acc3
        valid_batches += 1
        
    return {
        "acc@1": total_acc_1 / valid_batches,
        "acc@3": total_acc_3 / valid_batches,
        "valid_batches": valid_batches,
    }


def split_dataset(dataset, ratio=0.8):
    """データセット分割（seedを固定して再現性確保）"""
    import copy
    
    train_ds = copy.copy(dataset)
    test_ds = copy.copy(dataset)
    
    train_ds.is_train = True
    test_ds.is_train = False
    
    train_ds.samples_by_label = {}
    test_ds.samples_by_label = {}
    
    for lab, samples in dataset.samples_by_label.items():
        n_total = len(samples)
        if n_total == 0:
            train_ds.samples_by_label[lab] = []
            test_ds.samples_by_label[lab] = []
            continue
            
        shuffled = list(samples)
        random.shuffle(shuffled)
        
        n_train = int(n_total * ratio)
        if n_total > 0 and n_train == 0:
            n_train = 1
            
        train_ds.samples_by_label[lab] = shuffled[:n_train]
        test_ds.samples_by_label[lab] = shuffled[n_train:]

    for ds in [train_ds, test_ds]:
        ds._indices = []
        for lab, samples in ds.samples_by_label.items():
            for idx in range(len(samples)):
                ds._indices.append((lab, idx))
        
    return train_ds, test_ds


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    dataset = HoyoInstructionDataset(
        HOYO_ROOT, 
        INSTRUCTION_ONOMATOPEIA, 
        target_len=60, 
        is_train=True
    )
    train_ds, test_ds = split_dataset(dataset, ratio=0.8)
    
    # Load checkpoints
    ckpt_dir = HOYO_ROOT / "joint_training_results" / "sarashina_fine_full" / "checkpoints"
    stats_path = HOYO_ROOT / "joint_training_results" / "sarashina_fine_full" / "normalization_stats.json"
    
    # Apply normalization
    apply_normalization_from_stats(train_ds, stats_path)
    apply_normalization_from_stats(test_ds, stats_path)
    
    # Load model
    model, sem_proj = load_motionclip_and_checkpoints(device, ckpt_dir)
    model = model.to(device)
    
    # Semantic embeddings
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION COMPARISON")
    print("="*60)
    
    print(f"\nTest samples: {len(test_ds)}")
    
    # 元の方法
    old_result = evaluate_old_method(test_ds, model, sem_proj, sem_emb, device)
    print(f"\n[OLD METHOD - Batch Average]")
    print(f"  Acc@1: {old_result['acc@1']:.4f} ({old_result['acc@1']*100:.1f}%)")
    print(f"  Acc@3: {old_result['acc@3']:.4f} ({old_result['acc@3']*100:.1f}%)")
    print(f"  Batches: {old_result['valid_batches']}")
    
    # 正しい方法
    correct_result = evaluate_correct(test_ds, model, sem_proj, sem_emb, device)
    print(f"\n[CORRECT METHOD - Sample Count]")
    print(f"  Acc@1: {correct_result['acc@1']:.4f} ({correct_result['acc@1']*100:.1f}%)")
    print(f"  Acc@3: {correct_result['acc@3']:.4f} ({correct_result['acc@3']*100:.1f}%)")
    print(f"  Correct/Total: {correct_result['correct_1']}/{correct_result['total_samples']}")
    
    # 差分
    diff = correct_result['acc@1'] - old_result['acc@1']
    print(f"\n[DIFFERENCE]")
    print(f"  Acc@1 diff: {diff:+.4f} ({diff*100:+.1f}pp)")
    
    # バッチサイズの影響を確認
    print("\n" + "="*60)
    print("BATCH SIZE IMPACT")
    print("="*60)
    
    for bs in [1, 8, 16, 32, 64, 65, 128]:
        old = evaluate_old_method(test_ds, model, sem_proj, sem_emb, device, batch_size=bs)
        correct = evaluate_correct(test_ds, model, sem_proj, sem_emb, device, batch_size=bs)
        print(f"  BS={bs:3d}: OLD={old['acc@1']*100:5.1f}%, CORRECT={correct['acc@1']*100:5.1f}%")


if __name__ == "__main__":
    main()

