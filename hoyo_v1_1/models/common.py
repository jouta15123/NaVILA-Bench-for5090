from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

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
    HOYO の JSON + pickle から instruction 11語だけを集めるデータセット。
    
    MotionCLIP 用の前処理:
      1. 身長正規化 (Scale Normalization) -> ロード時に適用
      2. Window Slicing (Crop) -> get_sample時に適用
      3. 重心除去 (Centering) -> get_sample時に適用
      4. 標準化 (Standardization) -> apply_normalization_from_stats で適用
    """

    def __init__(self, root: Path, target_labels: List[str], target_len: int = 60, is_train: bool = True, use_aug: bool = False):
        self.root = root
        self.target_labels = set(target_labels)
        self.target_len = target_len
        self.is_train = is_train
        self.use_aug = use_aug # Augmentation flag (only effective if is_train=True)

        # ラベルごとの生データリスト（長さは可変、身長正規化済み）
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
            
            # ロードして身長だけ正規化（長さはそのまま）
            coords = self._load_raw_and_scale(pkl_path)
            self.samples_by_label[inst].append(coords)

        print(f"Loaded HOYO samples (train={self.is_train}):")
        for lab in target_labels:
            print(f"  {lab}: {len(self.samples_by_label[lab])} samples")

    def _load_raw_and_scale(self, pkl_path: Path) -> np.ndarray:
        """
        Pickleを読み込み、位置合わせと身長正規化のみを行う（リサイズはしない）。
        """
        import pickle
        with open(pkl_path, "rb") as f:
            arr = pickle.load(f)  # (T, 14, 2)
        
        arr = arr.astype(np.float32)
        T, J, C = arr.shape
        assert J == 14 and C == 2, f"Unexpected HOYO shape {arr.shape}"

        # 1. 重心除去 (Global Centering for Height Calc)
        # 身長計算のためだけに、一時的に各フレームの重心を引く（体の中心基準にする）
        com = arr.mean(axis=1, keepdims=True)
        arr_centered = arr - com # 身長計算用

        # 2. 身長正規化 (Scale Normalization)
        # HOYOの関節順序: [頭, 首, 右肩, 右肘, 右手, 左肩, 左肘, 左手, 右腰, 右膝, 右足, 左腰, 左膝, 左足]
        # index:          0,  1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13
        # 座標順序: [y, x] (画像座標系)
        
        head_pos = arr_centered[:, 0, :] # 頭
        
        # 足の位置 (右足:10, 左足:13) の中点
        feet_pos = 0.5 * (arr_centered[:, 10, :] + arr_centered[:, 13, :])
        
        dists = np.linalg.norm(head_pos - feet_pos, axis=-1)
        scale = dists.mean()
        if scale < 1e-6:
            scale = 1.0
            
        # 全体をスケールで割る
        # ここでは重心除去（arr_centered）は採用せず、元のarr（移動情報あり）を使う。
        # ただし、座標値が大きくなりすぎないよう、シーケンス全体の初期位置（最初のフレームの重心）を原点にする。
        
        # 座標系変換: HOYOは [y, x] なので、 [x, y] に直すか、そのまま扱うか。
        # MotionCLIPは通常 [x, y, z] なので、[x, y] として扱いたいなら入れ替えが必要。
        # 多くの2Dポーズ推定は [x, y] なので、合わせるためにここで入れ替える。
        arr_swapped = arr[..., ::-1] # [y, x] -> [x, y]
        
        # 初期位置の重心（全関節の平均）を原点にする
        initial_com = arr_swapped[0].mean(axis=0)
        arr_rooted = arr_swapped - initial_com
        
        arr_scaled = arr_rooted / scale
        return arr_scaled

    def get_sample(self, label: str) -> np.ndarray:
        """
        指定ラベルの中からランダムに1つ選び、60フレームをクロップして返す。
        is_train=Trueならランダムクロップ＋Data Augmentation、Falseならセンタークロップ。
        """
        samples = self.samples_by_label.get(label, [])
        if not samples:
            raise ValueError(f"No samples for label {label}")
        
        # ランダムにシーケンスを選ぶ
        raw_seq = random.choice(samples) # (T, 14, 2)
        T = raw_seq.shape[0]
        tgt = self.target_len

        # 3. Window Slicing (Crop)
        if T > tgt:
            if self.is_train:
                # Random Crop
                start = random.randint(0, T - tgt)
            else:
                # Center Crop
                start = (T - tgt) // 2
            cropped = raw_seq[start : start + tgt]
        else:
            # 短すぎる場合はパディング（HOYOではありえないはずだが安全策）
            pad_len = tgt - T
            cropped = np.pad(raw_seq, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            
        # 4. Data Augmentation (Train only)
        # ここで行うのは「スタイルの本質（速度感、リズム）」を壊さないものに限る
        if self.is_train and self.use_aug:
            # A. Random Horizontal Flip (左右反転)
            # x軸を反転させる (x -> -x)
            # ただし、左右の関節（右肩<->左肩など）の入れ替えも必要
            if random.random() < 0.5:
                cropped = cropped.copy()
                # x座標反転 (arr_swappedで[x,y]になっている前提)
                cropped[..., 0] *= -1
                
                # 関節入れ替え
                # HOYO (original): [頭0, 首1, 右肩2, 右肘3, 右手4, 左肩5, 左肘6, 左手7, 右腰8, 右膝9, 右足10, 左腰11, 左膝12, 左足13]
                # Swap pairs: (2,5), (3,6), (4,7), (8,11), (9,12), (10,13)
                pairs = [(2,5), (3,6), (4,7), (8,11), (9,12), (10,13)]
                for r, l in pairs:
                    # swap
                    tmp = cropped[:, r, :].copy()
                    cropped[:, r, :] = cropped[:, l, :]
                    cropped[:, l, :] = tmp

            # B. Random Rotation (わずかな回転)
            # 進行方向が変わるだけなので、スタイルの本質は変わらないはず
            # -15度〜+15度
            if random.random() < 0.5:
                angle_deg = random.uniform(-15, 15)
                angle_rad = np.deg2rad(angle_deg)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                rot_mat = np.array([[c, -s], [s, c]], dtype=np.float32)
                
                # (T, 14, 2) @ (2, 2) -> (T, 14, 2)
                cropped = cropped @ rot_mat.T

        # 5. 重心除去 (Local Centering)
        # クロップされた区間の「最初のフレーム」の重心を引いて、
        # その区間内での移動（速度・軌跡）を保存する。
        # 元の実装（各フレームごとの重心除去）だと、移動情報が消えて足踏みになってしまう。
        com = cropped[0].mean(axis=0) # (2,)
        centered = cropped - com      # (T, J, C) - (2,) -> Broadcast
        
        return centered.astype(np.float32)

    def get_all_samples_flat(self) -> List[np.ndarray]:
        """全データをフラットなリストで返す（統計計算用など）"""
        all_data = []
        for samples in self.samples_by_label.values():
            all_data.extend(samples)
        return all_data


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


def encode_semantics_siglip(
    labels: List[str],
    device: torch.device,
    model_id: str = "google/siglip-base-patch16-256-multilingual",
) -> torch.Tensor:
    from transformers import AutoTokenizer, SiglipTextModel

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_model = SiglipTextModel.from_pretrained(model_id).to(device)

    texts = [f"{w}と歩いている。" if w != "通常" else "普通に歩いている。" for w in labels]
    print("Semantic texts (SigLIP):")
    for t in texts:
        print(" ", t)

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = text_model(**encoded)

    emb = out.last_hidden_state[:, 0, :]
    emb = F.normalize(emb, dim=-1)
    return emb


def _compute_stats(dataset: HoyoInstructionDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    データセット全体の平均・分散を計算する。
    学習時と同じ条件（クロップ＋初期フレーム重心除去）で統計を取るため、
    一時的に is_train=False にして（センタークロップ）、全サンプルを取得する。
    """
    # 現在の状態を保存
    original_is_train = dataset.is_train
    dataset.is_train = False  # センタークロップで安定的に取得

    all_processed_samples = []
    
    # 全ラベルの全サンプルについて get_sample() を実行
    # get_sample() 内で「クロップ -> 初期フレーム重心除去」が行われる
    for lab, raw_samples in dataset.samples_by_label.items():
        # get_sample はランダム選択だが、is_train=False ならセンタークロップ固定。
        # ただし、同じラベルに複数サンプルある場合、get_sample(lab) はその中からランダムに1つ選ぶ。
        # 全サンプルを確実になめるには、dataset内部の実装に頼らずここでループする方が安全だが、
        # get_sample のロジック（クロップ＆重心除去）を再利用したい。
        # -> get_sample ロジックを切り出すのがベストだが、ここでは簡易的に
        #    raw_samples を直接処理する形にする（get_sample のロジックを模倣）。
        
        tgt = dataset.target_len
        
        for raw_seq in raw_samples:
            T = raw_seq.shape[0]
            # Center Crop (is_train=False equivalent)
            if T > tgt:
                start = (T - tgt) // 2
                cropped = raw_seq[start : start + tgt]
            else:
                pad_len = tgt - T
                cropped = np.pad(raw_seq, ((0, pad_len), (0, 0), (0, 0)), mode="edge")
            
            # 初期フレーム重心除去 (get_sample と同じロジック)
            com0 = cropped[0].mean(axis=0)
            centered = cropped - com0
            
            all_processed_samples.append(centered)

    # 状態を復元
    dataset.is_train = original_is_train

    # 全フレームを結合: (TotalFrames, 14, 2)
    concatenated = np.concatenate(all_processed_samples, axis=0)
    
    data_mean = concatenated.mean(axis=(0, 1))
    
    # 異方性正規化 (X/Y独立)
    data_std = concatenated.std(axis=(0, 1)) + 1e-6
    
    return data_mean, data_std


def _apply_stats(dataset: HoyoInstructionDataset, mean: np.ndarray, std: np.ndarray) -> None:
    # メモリ上の全データを書き換える（標準化）
    for lab, samples in dataset.samples_by_label.items():
        new_samples = []
        for arr in samples:
            # arr: (T, 14, 2)
            # ブロードキャスト: (2,) -> (T, 14, 2)
            norm_arr = (arr - mean) / std
            new_samples.append(norm_arr.astype(np.float32))
        dataset.samples_by_label[lab] = new_samples


def normalize_dataset(dataset: HoyoInstructionDataset, stats_path: Path):
    """
    データセット全体の平均・分散を計算し、適用して保存する。
    """
    data_mean, data_std = _compute_stats(dataset)
    print(f"Data Mean: {data_mean}, Std: {data_std}")
    _apply_stats(dataset, data_mean, data_std)

    stats = {"mean": data_mean.tolist(), "std": data_std.tolist()}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    return data_mean, data_std


def apply_normalization_from_stats(dataset: HoyoInstructionDataset, stats_path: Path) -> None:
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    with open(stats_path, "r") as f:
        stats = json.load(f)

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    _apply_stats(dataset, mean, std)
