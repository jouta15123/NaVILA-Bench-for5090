import argparse
import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
HOYO_ROOT = REPO_ROOT / "hoyo_v1_1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hoyo_v1_1.models.common import (  # type: ignore[import]
    HoyoInstructionDataset,
    encode_semantics_sarashina,
    INSTRUCTION_ONOMATOPEIA,
)


# -----------------------------------------------------------------------------
# Simple motion encoder (HOYO 専用, MotionCLIP なし)
# 入力: (B, T, 14, 2) -> flatten -> (B, T, 28)
# RNN で時間方向を処理して、mean pooling で 1 ベクトルに集約
# -----------------------------------------------------------------------------


class SimpleMotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 28)  # 14 joints * 2 coords
        """
        x = self.input_proj(x)  # (B, T, H)
        out, _ = self.rnn(x)  # (B, T, 2H)
        feat = out.mean(dim=1)  # (B, 2H) 時間平均
        z = self.fc(feat)  # (B, D)
        z = F.normalize(z, dim=-1)
        return z


# -----------------------------------------------------------------------------
# Supervised contrastive 学習 (motion latent <-> Sarashina semantics)
# -----------------------------------------------------------------------------


def train_sem_motion_scratch(
    dataset: HoyoInstructionDataset,
    model: SimpleMotionEncoder,
    sem_emb: torch.Tensor,
    device: torch.device,
    steps: int = 4000,
    samples_per_label: int = 4,
    temp: float = 0.07,
    lr: float = 1e-3,
    lr_encoder: float = 1e-4,
    train_encoder: bool = True,
    log_interval: int = 100,
    wandb_run=None,
) -> nn.Module:
    """
    HOYO 専用の小型 motion encoder を、Sarashina semantics と supervised contrastive で対応付ける。
    """
    labels = INSTRUCTION_ONOMATOPEIA

    sem_emb = sem_emb.to(device)  # (11, D_sem)
    d_sem = sem_emb.shape[1]
    d_motion = model.latent_dim

    # semantics -> motion latent
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))

    params = [
        {"params": sem_proj.parameters(), "lr": lr},
        {"params": [logit_scale], "lr": lr},
    ]
    if train_encoder:
        params.append({"params": model.parameters(), "lr": lr_encoder})
        model.train()
    else:
        model.eval()

    optimizer = torch.optim.AdamW(params)

    for step in range(1, steps + 1):
        xs: List[np.ndarray] = []
        ys: List[int] = []

        # 各ラベルから samples_per_label 個ずつサンプル
        for lab_idx, lab in enumerate(labels):
            samples = dataset.samples_by_label[lab]
            if not samples:
                continue
            for _ in range(samples_per_label):
                arr = random.choice(samples)  # (T,14,2)
                xs.append(arr)
                ys.append(lab_idx)

        if not xs:
            continue

        coords = np.stack(xs, axis=0)  # (B,T,14,2)
        B, T, J, C = coords.shape
        assert J == 14 and C == 2
        x_batch = coords.reshape(B, T, J * C).astype(np.float32)  # (B,T,28)
        x_batch_t = torch.from_numpy(x_batch).to(device)
        y_batch = torch.tensor(ys, dtype=torch.long, device=device)

        # motion encoder
        z_m = model(x_batch_t)  # (B, Dm)

        # semantics projector
        z_s_cls = sem_proj(sem_emb)  # (11, Dm)
        z_s_cls = F.normalize(z_s_cls, dim=-1)
        z_s_inst = z_s_cls[y_batch]  # (B, Dm)

        # supervised contrastive: motion + semantics を両方 views として扱う
        features = torch.cat([z_m, z_s_inst], dim=0)  # (2B, Dm)
        features = F.normalize(features, dim=-1)
        labels_ext = torch.cat([y_batch, y_batch], dim=0)  # (2B,)

        N = features.shape[0]
        sim_matrix = torch.matmul(features, features.t())  # (N, N)
        sim_matrix = torch.exp(logit_scale) * sim_matrix

        # 自分自身は除外
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max  # 数値安定化
        exp_logits = torch.exp(logits)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        exp_logits = exp_logits * (~mask_self)

        # ラベルが同じものを positive とするマスク
        labels_eq = labels_ext.unsqueeze(0) == labels_ext.unsqueeze(1)  # (N,N)
        pos_mask = labels_eq & (~mask_self)

        # 分母: 全ての negative + positive
        denom = exp_logits.sum(dim=1, keepdim=True)  # (N,1)
        # 分子: positive のみ
        pos_exp = exp_logits * pos_mask
        pos_sum = pos_exp.sum(dim=1)

        # positive がない anchor は除外
        valid = pos_sum > 0
        loss_vals = torch.zeros_like(pos_sum)
        loss_vals[valid] = -torch.log(pos_sum[valid] / (denom[valid].squeeze(1) + 1e-8))
        loss = loss_vals[valid].mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_interval == 0 or step == 1:
            # 簡易 Top-1 精度（motion -> semantics, プロトタイプに対する分類として）
            with torch.no_grad():
                logits_m2s = z_m @ z_s_cls.t()
                preds_m2s = logits_m2s.argmax(dim=1)
                acc_m2s = (preds_m2s == y_batch).float().mean().item()

            print(f"[Step {step:04d}] Loss: {loss.item():.4f} | Acc(m->s): {acc_m2s:.3f}")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/contrastive_loss": float(loss.item()),
                        "train/acc_m2s": float(acc_m2s),
                        "train/step": step,
                    },
                    step=step,
                )

    return model, sem_proj


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="HOYO simple motion encoder + semantic contrastive training")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--samples-per-label", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-encoder-train", action="store_true", help="Freeze encoder and only train semantic projector")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hoyo_motion_scratch", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user or team)")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    hoyo_root = HOYO_ROOT
    target_len = 60

    # Dataset
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)

    # Semantics (Sarashina)
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)

    # Model
    model = SimpleMotionEncoder(
        input_dim=14 * 2,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    # Optional wandb
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed in this environment. Please install it or run without --wandb.")
        wandb_config = {
            "steps": args.steps,
            "samples_per_label": args.samples_per_label,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "temp": args.temp,
            "train_encoder": not args.no_encoder_train,
            "device": str(device),
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=wandb_config,
        )

    try:
        train_sem_motion_scratch(
            dataset=dataset,
            model=model,
            sem_emb=sem_emb,
            device=device,
            steps=args.steps,
            samples_per_label=args.samples_per_label,
            temp=args.temp,
            lr=args.lr,
            lr_encoder=args.lr_encoder,
            train_encoder=not args.no_encoder_train,
            log_interval=args.log_interval,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()


