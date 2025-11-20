import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer


INSTRUCTION_ONOMATOPEIA = [
    "通常",
    "すたすた",
    "せかせか",
    "てくてく",
    "どっしどっし",
    "とぼとぼ",
    "のしのし",
    "のろのろ",
    "ぶらぶら",
    "よたよた",
    "よろよろ",
]


class HoyoInstructionDataset:
    """
    HOYO の JSON + pickle から instruction 11語だけを集める簡易データセット（メモリ常駐）。
    """

    def __init__(self, root: Path, target_labels: List[str], target_len: int = 128):
        self.root = root
        self.target_labels = set(target_labels)
        self.target_len = target_len

        self.samples_by_label: Dict[str, List[np.ndarray]] = {lab: [] for lab in target_labels}

        json_files = sorted(root.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            inst = data["annotation"]["instruction"]
            if inst not in self.target_labels:
                continue
            rel_path = data["path"]  # e.g. "data/100.pickle"
            pkl_path = root / rel_path
            if not pkl_path.exists():
                continue
            coords = self._load_and_resample(pkl_path)
            self.samples_by_label[inst].append(coords)

        # 各ラベルに何サンプルあるかを表示
        print("Loaded HOYO instruction samples:")
        for lab in target_labels:
            print(f"  {lab}: {len(self.samples_by_label[lab])} samples")

    def _load_and_resample(self, pkl_path: Path) -> np.ndarray:
        """
        data/xxx.pickle -> (T, 14, 2) -> 時間長を target_len にリサンプルし (T, 28) にする。
        """
        import pickle

        with open(pkl_path, "rb") as f:
            arr = pickle.load(f)  # (T, 14, 2), dtype=float

        # 中心を原点に平行移動（位置を消して形だけ）
        com = arr.mean(axis=1, keepdims=True)  # (T, 1, 2)
        arr_rel = arr - com  # (T, 14, 2)

        T, J, C = arr_rel.shape
        arr_flat = arr_rel.reshape(T, J * C)  # (T, 28)

        target_T = self.target_len
        if T == target_T:
            return arr_flat.astype(np.float32)

        # 線形補間で時間方向をリサンプル
        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, target_T)
        arr_resampled = np.empty((target_T, J * C), dtype=np.float32)
        for d in range(J * C):
            arr_resampled[:, d] = np.interp(new_t, old_t, arr_flat[:, d])

        return arr_resampled


class MotionEncoder(nn.Module):
    """
    HOYO スケルトン用のシンプルな BiGRU ベース encoder。
    入力: (B, T, 28) -> 出力: (B, D)
    """

    def __init__(self, input_dim: int = 28, hidden_dim: int = 256, num_layers: int = 2, out_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        out, h = self.gru(x)  # h: (num_layers*2, B, hidden_dim)
        # 最終層の双方向 hidden を取り出して concat
        h_last = h[-2:, :, :]  # (2, B, hidden_dim)
        h_cat = torch.cat([h_last[0], h_last[1]], dim=-1)  # (B, hidden_dim*2)
        z = self.fc(h_cat)  # (B, out_dim)
        z = F.normalize(z, dim=-1)
        return z


def encode_semantics_sarashina(labels: List[str], device: torch.device) -> torch.Tensor:
    """
    Sarashina で「〜と歩いている。」テンプレ付きの意味埋め込みを取得。
    """
    model_id = "sbintuitions/sarashina-embedding-v2-1b"
    model = SentenceTransformer(model_id, device=str(device))
    texts = [f"{w}と歩いている。" if w != "通常" else "普通に歩いている。" for w in labels]
    print("Semantic texts:")
    for t in texts:
        print(" ", t)
    emb = model.encode(texts, convert_to_tensor=True, device=device)
    emb = F.normalize(emb, dim=-1)
    return emb  # (B, D_sem)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    cov = x_centered.T @ x_centered / (x_centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    W = eigvecs[:, idx]
    x_2d = x_centered @ W
    return x_2d


def plot_motion_latent(coords: np.ndarray, labels: List[str], out_path: Path):
    os.makedirs(out_path.parent, exist_ok=True)

    plt.figure(figsize=(6, 6))

    # ざっくりカテゴリ
    cat_map: Dict[str, str] = {}
    fast = {"すたすた", "せかせか", "てくてく"}
    slow = {"のろのろ", "とぼとぼ"}
    heavy = {"どっしどっし", "のしのし"}
    unstable = {"よたよた", "よろよろ"}
    aimless = {"ぶらぶら"}

    for w in labels:
        if w in fast:
            cat_map[w] = "速い"
        elif w in slow:
            cat_map[w] = "遅い"
        elif w in heavy:
            cat_map[w] = "重い"
        elif w in unstable:
            cat_map[w] = "ふらつき"
        elif w in aimless:
            cat_map[w] = "ぶらぶら系"
        else:
            cat_map[w] = "その他"

    cat_to_color = {
        "速い": "tab:blue",
        "遅い": "tab:orange",
        "重い": "tab:red",
        "ふらつき": "tab:green",
        "ぶらぶら系": "tab:purple",
        "その他": "gray",
    }

    for (x, y), lab in zip(coords, labels):
        cat = cat_map[lab]
        color = cat_to_color.get(cat, "gray")
        plt.scatter(x, y, c=color, s=20, alpha=0.7)

    # ラベルの重心位置に文字を書く
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idxs = [i for i, L in enumerate(labels) if L == lab]
        cx = coords[idxs, 0].mean()
        cy = coords[idxs, 1].mean()
        plt.text(cx, cy, lab, fontsize=10, weight="bold")

    plt.title("HOYO 歩容 latent（instruction 11語, PCA2D）")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.3)

    handles = []
    for cat, color in cat_to_color.items():
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=cat))
    plt.legend(handles=handles, title="ざっくりカテゴリ", loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_sem_motion_contrastive(
    dataset: HoyoInstructionDataset,
    sem_emb: torch.Tensor,
    device: torch.device,
    steps: int = 4000,
    samples_per_label: int = 4,
    temp: float = 0.07,
    lr: float = 1e-3,
    log_interval: int = 100,
) -> MotionEncoder:
    """
    各ステップで 11 ラベルから複数サンプルを引いてきて、
    motion / semantics 両方を含む supervised contrastive learning を行う。
    """
    labels = INSTRUCTION_ONOMATOPEIA
    motion_enc = MotionEncoder().to(device)
    # Sarashina の次元 (例えば 1792) -> motion encoder の次元 (256) への射影を学習する
    sem_emb = sem_emb.to(device)  # (11, D_sem)
    d_sem = sem_emb.shape[1]
    d_motion = motion_enc.fc.out_features
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    # 学習可能な温度（CLIP スタイル）
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))

    optimizer = torch.optim.AdamW(
        list(motion_enc.parameters()) + list(sem_proj.parameters()) + [logit_scale],
        lr=lr,
    )

    for step in range(1, steps + 1):
        xs: List[np.ndarray] = []
        ys: List[int] = []
        for lab_idx, lab in enumerate(labels):
            samples = dataset.samples_by_label[lab]
            for _ in range(samples_per_label):
                arr = random.choice(samples)  # (T, 28)
                xs.append(arr)
                ys.append(lab_idx)

        x_batch = torch.from_numpy(np.stack(xs, axis=0)).to(device)  # (B, T, 28)
        y_batch = torch.tensor(ys, dtype=torch.long, device=device)  # (B,)

        motion_enc.train()
        z_m = motion_enc(x_batch)  # (B, Dm)
        z_m = F.normalize(z_m, dim=-1)

        # sem 側は projector を通して motion 側の次元へ（クラスプロトタイプ & 各サンプル用）
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
        mask = torch.eye(N, dtype=torch.bool, device=device)
        exp_logits = exp_logits * (~mask)

        # ラベルが同じものを positive とするマスク
        labels_eq = labels_ext.unsqueeze(0) == labels_ext.unsqueeze(1)  # (N,N)
        pos_mask = labels_eq & (~mask)

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
        optimizer.step()

        if step % log_interval == 0 or step == 1:
            # 簡易 Top-1 精度（motion -> semantics, プロトタイプに対する分類として）
            with torch.no_grad():
                logits_m2s = z_m @ z_s_cls.t()
                preds_m2s = logits_m2s.argmax(dim=1)
                acc_m2s = (preds_m2s == y_batch).float().mean().item()
            print(f"[step {step}/{steps}] loss={loss.item():.4f}, acc(m->s)={acc_m2s:.3f}")

    return motion_enc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    hoyo_root = Path("hoyo_v1_1")
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=128)

    # Semantic embeddings (frozen)
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)

    # Train contrastive motion encoder
    motion_enc = train_sem_motion_contrastive(dataset, sem_emb, device=device)

    # 全サンプルを埋め込みに通して PCA 可視化
    all_feats = []
    all_labels = []
    motion_enc.eval()
    with torch.no_grad():
        for lab, samples in dataset.samples_by_label.items():
            for arr in samples:
                x = torch.from_numpy(arr[None, ...]).to(device)  # (1, T, 28)
                z = motion_enc(x)  # (1, D)
                all_feats.append(z.cpu().numpy()[0])
                all_labels.append(lab)

    all_feats = np.stack(all_feats, axis=0)
    coords = pca_2d(all_feats)
    out_path = hoyo_root / "hoyo_motion_latent_instruction_pca.png"
    plot_motion_latent(coords, all_labels, out_path)
    print("Saved motion latent PCA plot to:", out_path)


if __name__ == "__main__":
    main()


