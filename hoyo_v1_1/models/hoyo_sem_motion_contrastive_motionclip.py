import os
import sys
import json
import random
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


# MotionCLIP を import するためにパスを追加
ROOT = Path(__file__).resolve().parents[1]
MOTIONCLIP_ROOT = ROOT / "MotionCLIP"
if str(MOTIONCLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONCLIP_ROOT))

from src.models.get_model import get_model as motionclip_get_model  # type: ignore
from src.utils.misc import load_model_wo_clip  # type: ignore


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
    MotionCLIP にそのまま渡せるように (T, 14, 2) 形式で持つ。
    """

    def __init__(self, root: Path, target_labels: List[str], target_len: int = 60):
        self.root = root
        self.target_labels = set(target_labels)
        self.target_len = target_len

        self.samples_by_label: Dict[str, List[np.ndarray]] = {lab: [] for lab in target_labels}

        # Look for JSONs in data/ subdirectory if available, else root
        data_dir = root / "data"
        if not data_dir.exists():
            data_dir = root
            
        json_files = sorted(data_dir.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            inst = data["annotation"]["instruction"]
            if inst not in self.target_labels:
                continue
            rel_path = data["path"]  # e.g. "data/100.pickle"
            # rel_path is likely relative to hoyo_v1_1 root. 
            # If we are in data/, we need to go up or check path logic.
            # Assuming data["path"] is like "data/100.pickle" and root is hoyo_v1_1/
            pkl_path = root / rel_path
            if not pkl_path.exists():
                continue
            coords = self._load_and_resample(pkl_path)
            self.samples_by_label[inst].append(coords)

        print("Loaded HOYO instruction samples (for MotionCLIP):")
        for lab in target_labels:
            print(f"  {lab}: {len(self.samples_by_label[lab])} samples")

    def _load_and_resample(self, pkl_path: Path) -> np.ndarray:
        """
        data/xxx.pickle -> (T, 14, 2) -> 時間長を target_len にリサンプルした (T, 14, 2) を返す。
        """
        import pickle

        with open(pkl_path, "rb") as f:
            arr = pickle.load(f)  # (T, 14, 2)

        T, J, C = arr.shape
        assert J == 14 and C == 2, f"Unexpected HOYO shape {arr.shape}"

        # 中心を原点に平行移動（位置を消して形だけ）
        com = arr.mean(axis=1, keepdims=True)  # (T, 1, 2)
        arr_rel = arr - com  # (T, 14, 2)

        target_T = self.target_len
        if T == target_T:
            return arr_rel.astype(np.float32)

        # 線形補間で時間方向をリサンプル
        old_t = np.linspace(0, 1, T)
        new_t = np.linspace(0, 1, target_T)
        arr_resampled = np.empty((target_T, J, C), dtype=np.float32)
        for j in range(J):
            for c in range(C):
                arr_resampled[:, j, c] = np.interp(new_t, old_t, arr_rel[:, j, c])

        return arr_resampled


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

    plt.title("HOYO MotionCLIP latent（instruction 11語, PCA2D）")
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


def load_motionclip_encoder(device: torch.device, target_len: int = 60):
    """
    exps/paper-model の checkpoint_0100.pth.tar を読み込み、
    HOYO 用の (njoints=14, nfeats=2, num_frames=target_len) 設定で MotionCLIP encoder を構築。
    入力次元は違うので skelEmbedding はランダム初期化だが、Transformer 層などは再利用される。
    """
    import yaml  # type: ignore

    opt_path = MOTIONCLIP_ROOT / "exps" / "paper-model" / "opt.yaml"
    ckpt_path = MOTIONCLIP_ROOT / "exps" / "paper-model" / "checkpoint_0100.pth.tar"
    assert opt_path.exists(), f"opt.yaml not found at {opt_path}"
    assert ckpt_path.exists(), f"checkpoint not found at {ckpt_path}"

    with open(opt_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ベース設定をコピー
    params = dict(cfg)

    # HOYO 用に上書き / 追加
    params["device"] = device
    params["njoints"] = 14
    params["nfeats"] = 2
    params["num_frames"] = target_len
    params["num_classes"] = 1
    # modeltype, pose_rep, translation, glob, glob_rot, latent_dim などは opt.yaml をそのまま利用

    # MotionCLIP の SMPL パスが ./models/smpl を前提にしているので、一時的に CWD を MotionCLIP 直下にする
    cwd = os.getcwd()
    os.chdir(str(MOTIONCLIP_ROOT))
    try:
        # clip_model は対照学習に使わないので None で OK（compute_clip_losses を呼ばない）
        model = motionclip_get_model(params, clip_model=None)
        state_dict = torch.load(ckpt_path, map_location=device)
        # 入力次元が異なる層（skelEmbedding / finallayer）は除外して部分的にロード
        filtered = {
            k: v
            for k, v in state_dict.items()
            if not (
                k.startswith("encoder.skelEmbedding.")
                or k.startswith("decoder.finallayer.")
            )
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print("Loaded MotionCLIP weights (partial). Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    finally:
        os.chdir(cwd)

    encoder = model.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, params


def train_sem_motion_contrastive_motionclip(
    dataset: HoyoInstructionDataset,
    encoder,
    sem_emb: torch.Tensor,
    device: torch.device,
    steps: int = 4000,
    samples_per_label: int = 4,
    temp: float = 0.07,
    lr: float = 1e-3,
    lr_encoder: float = 1e-4,
    train_encoder: bool = False,
    log_interval: int = 100,
) -> nn.Module:
    """
    MotionCLIP encoder の latent を固定し、
    Sarashina semantics 側の projector + 温度だけを学習する supervised contrastive 学習。
    """
    labels = INSTRUCTION_ONOMATOPEIA
    # MotionCLIP latent dim（mu のサイズ）を推定
    # とりあえず 1 サンプル通して shape を確認
    any_lab = labels[0]
    example = dataset.samples_by_label[any_lab][0]  # (T,14,2)
    T, J, C = example.shape
    x_example = torch.from_numpy(example.transpose(1, 2, 0)[None, ...]).to(device)  # (1,14,2,T)
    mask_example = torch.ones((1, T), dtype=torch.bool, device=device)
    lengths_example = torch.full((1,), T, dtype=torch.long, device=device)
    y_example = torch.zeros((1,), dtype=torch.long, device=device)
    batch_example = {"x": x_example, "mask": mask_example, "lengths": lengths_example, "y": y_example}
    with torch.no_grad():
        out = encoder(batch_example)
    d_motion = out["mu"].shape[1]

    # semantic projector: Sarashina -> MotionCLIP latent 次元
    sem_emb = sem_emb.to(device)  # (11, D_sem)
    d_sem = sem_emb.shape[1]
    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))

    if train_encoder:
        encoder.train()
        optimizer = torch.optim.AdamW(
            [
                {"params": sem_proj.parameters(), "lr": lr},
                {"params": [logit_scale], "lr": lr},
                {"params": encoder.parameters(), "lr": lr_encoder},
            ]
        )
    else:
        encoder.eval()
        optimizer = torch.optim.AdamW(
            list(sem_proj.parameters()) + [logit_scale],
            lr=lr,
        )

    for step in range(1, steps + 1):
        xs: List[np.ndarray] = []
        ys: List[int] = []
        for lab_idx, lab in enumerate(labels):
            samples = dataset.samples_by_label[lab]
            for _ in range(samples_per_label):
                arr = random.choice(samples)  # (T,14,2)
                xs.append(arr)
                ys.append(lab_idx)

        # (B,14,2,T) に変換
        coords = np.stack(xs, axis=0)  # (B,T,14,2)
        coords = coords.transpose(0, 2, 3, 1)  # (B,14,2,T)
        x_batch = torch.from_numpy(coords).to(device)
        y_batch = torch.tensor(ys, dtype=torch.long, device=device)
        B, _, _, Tcur = x_batch.shape
        mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
        lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)

        batch = {
            "x": x_batch,
            "mask": mask,
            "lengths": lengths,
            "y": torch.zeros((B,), dtype=torch.long, device=device),
        }

        if train_encoder:
            out = encoder(batch)
            z_m = out["mu"]  # (B, Dm)
            z_m = F.normalize(z_m, dim=-1)
        else:
            with torch.no_grad():
                out = encoder(batch)
                z_m = out["mu"]  # (B, Dm)
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
        optimizer.step()

        if step % log_interval == 0 or step == 1:
            # 簡易 Top-1 精度（motion -> semantics, プロトタイプに対する分類として）
            with torch.no_grad():
                logits_m2s = z_m @ z_s_cls.t()
                preds_m2s = logits_m2s.argmax(dim=1)
                acc_m2s = (preds_m2s == y_batch).float().mean().item()
            print(f"[step {step}/{steps}] loss={loss.item():.4f}, acc(m->s)={acc_m2s:.3f}")

    # 学習済み projector を返しておく（後で NaVILA でも使える）
    return sem_proj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    hoyo_root = ROOT / "hoyo_v1_1"
    target_len = 60  # MotionCLIP pretrain の num_frames に合わせておく
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)

    # Semantic embeddings (frozen Sarashina)
    sem_emb = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)

    # MotionCLIP encoder のロード
    encoder, params = load_motionclip_encoder(device=device, target_len=target_len)
    print("Loaded MotionCLIP encoder with latent_dim:", params.get("latent_dim"))

    # MotionCLIP latent + encoder の両方を HOYO に少し寄せる
    sem_proj = train_sem_motion_contrastive_motionclip(
        dataset,
        encoder,
        sem_emb,
        device=device,
        steps=4000,
        samples_per_label=4,
        temp=0.07,
        lr=1e-3,
        lr_encoder=1e-4,
        train_encoder=True,
    )

    # 全サンプルを埋め込みに通して PCA 可視化
    all_feats = []
    all_labels = []
    encoder.eval()
    with torch.no_grad():
        z_s_cls = sem_proj(sem_emb.to(device))
        z_s_cls = F.normalize(z_s_cls, dim=-1)

        for lab, samples in dataset.samples_by_label.items():
            for arr in samples:
                coords = arr.transpose(1, 2, 0)[None, ...]  # (1,14,2,T)
                x = torch.from_numpy(coords).to(device)
                B, _, _, Tcur = x.shape
                mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
                lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)
                batch = {"x": x, "mask": mask, "lengths": lengths, "y": torch.zeros((B,), dtype=torch.long, device=device)}
                out = encoder(batch)
                z = out["mu"]  # (1, D)
                z = F.normalize(z, dim=-1)
                all_feats.append(z.cpu().numpy()[0])
                all_labels.append(lab)

    all_feats = np.stack(all_feats, axis=0)
    coords = pca_2d(all_feats)
    out_path = hoyo_root / "hoyo_motionclip_motion_latent_instruction_pca.png"
    plot_motion_latent(coords, all_labels, out_path)
    print("Saved MotionCLIP-based motion latent PCA plot to:", out_path)


if __name__ == "__main__":
    main()


