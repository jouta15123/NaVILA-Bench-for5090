from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
import torch.nn.functional as F
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
    MotionCLIP にそのまま渡せるように (T, 14, 2) 形式で持つ。
    """

    def __init__(self, root: Path, target_labels: List[str], target_len: int = 60):
        self.root = root
        self.target_labels = set(target_labels)
        self.target_len = target_len

        self.samples_by_label: Dict[str, List[np.ndarray]] = {lab: [] for lab in target_labels}

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
            pkl_path = root / rel_path
            if not pkl_path.exists():
                continue
            coords = self._load_and_resample(pkl_path)
            self.samples_by_label[inst].append(coords)

        print("Loaded HOYO instruction samples (for MotionCLIP):")
        for lab in target_labels:
            print(f"  {lab}: {len(self.samples_by_label[lab])} samples")

    def _load_and_resample(self, pkl_path: Path) -> np.ndarray:
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


def _compute_stats(dataset: HoyoInstructionDataset, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    all_samples = []
    for lab in labels:
        all_samples.extend(dataset.samples_by_label[lab])

    all_data = np.stack(all_samples, axis=0)  # (N, T, 14, 2)
    data_mean = all_data.mean(axis=(0, 1, 2))
    data_std = all_data.std(axis=(0, 1, 2)) + 1e-6
    return data_mean, data_std


def _apply_stats(dataset: HoyoInstructionDataset, labels: List[str], mean: np.ndarray, std: np.ndarray) -> None:
    for lab in labels:
        new_samples = []
        for arr in dataset.samples_by_label[lab]:
            norm_arr = (arr - mean) / std
            new_samples.append(norm_arr)
        dataset.samples_by_label[lab] = new_samples


def normalize_dataset(dataset: HoyoInstructionDataset, labels: List[str], stats_path: Path) -> None:
    data_mean, data_std = _compute_stats(dataset, labels)
    print(f"Data Mean: {data_mean}, Std: {data_std}")
    _apply_stats(dataset, labels, data_mean, data_std)

    stats = {"mean": data_mean.tolist(), "std": data_std.tolist()}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f)
