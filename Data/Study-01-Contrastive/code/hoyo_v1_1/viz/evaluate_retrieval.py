#!/usr/bin/env python3
"""
Retrieval/Clustering evaluation for MotionCLIP latent snapshots.

- Text -> Motion retrieval (multi-positive)
  R@K, MedR
- Motion -> Text retrieval (single-positive)
  R@K, MedR
- Silhouette score on motion latents

Usage:
  python hoyo_v1_1/viz/evaluate_retrieval.py \
    --snapshot hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
    --splits test \
    --mode auto \
    --ks 1,3,5,10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover - optional dependency
    silhouette_score = None


# Coarse style groups (same as train_motionclip_joint.py)
COARSE_GROUPS = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["通常", "とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}
COARSE_LABELS = list(COARSE_GROUPS.keys())


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_snapshot(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    label_list = [str(l) for l in data["label_list"]]
    return {
        "z_m": data["z_m"],
        "labels_idx": data["labels_idx"].astype(int),
        "label_list": label_list,
        "z_s_cls": data["z_s_cls"],
        "splits": data.get("splits", None),
    }


def filter_by_splits(
    z_m: np.ndarray,
    labels_idx: np.ndarray,
    splits: Optional[np.ndarray],
    allowed_splits: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if splits is None or not allowed_splits:
        return z_m, labels_idx, splits
    splits = np.asarray(splits).astype(str)
    mask = np.isin(splits, allowed_splits)
    return z_m[mask], labels_idx[mask], splits[mask]


def detect_label_mode(label_list: List[str]) -> str:
    if len(label_list) == len(COARSE_LABELS) and all(lab in COARSE_LABELS for lab in label_list):
        return "coarse"
    return "fine"


def build_coarse_view(
    z_m: np.ndarray,
    labels_idx: np.ndarray,
    label_list: List[str],
    z_s_cls: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # If already coarse, keep as-is
    if detect_label_mode(label_list) == "coarse":
        return z_m, labels_idx, z_s_cls, label_list

    label_to_idx = {lab: i for i, lab in enumerate(label_list)}

    # Build coarse prototypes by averaging fine prototypes
    coarse_names: List[str] = []
    coarse_protos: List[np.ndarray] = []
    for coarse_lab, fine_labs in COARSE_GROUPS.items():
        idxs = [label_to_idx[fl] for fl in fine_labs if fl in label_to_idx]
        if not idxs:
            continue
        coarse_names.append(coarse_lab)
        coarse_protos.append(z_s_cls[idxs].mean(axis=0))

    if not coarse_protos:
        raise ValueError("Failed to build coarse prototypes; label_list doesn't match COARSE_GROUPS.")

    z_s_coarse = np.stack(coarse_protos, axis=0)
    coarse_name_to_idx = {name: i for i, name in enumerate(coarse_names)}

    # Map labels to coarse indices
    new_labels = []
    for li in labels_idx:
        lab = label_list[int(li)]
        # already coarse label name
        if lab in coarse_name_to_idx:
            new_labels.append(coarse_name_to_idx[lab])
            continue
        mapped = None
        for coarse_lab, fine_labs in COARSE_GROUPS.items():
            if lab in fine_labs and coarse_lab in coarse_name_to_idx:
                mapped = coarse_name_to_idx[coarse_lab]
                break
        if mapped is None:
            new_labels.append(-1)
        else:
            new_labels.append(mapped)

    new_labels = np.asarray(new_labels, dtype=int)
    valid_mask = new_labels >= 0
    if not valid_mask.all():
        z_m = z_m[valid_mask]
        new_labels = new_labels[valid_mask]

    return z_m, new_labels, z_s_coarse, coarse_names


def text_to_motion_metrics(
    z_text: np.ndarray,
    z_motion: np.ndarray,
    labels_motion: np.ndarray,
    ks: List[int],
) -> Tuple[Dict[str, float], Optional[float], int]:
    sims = z_text @ z_motion.T  # (K, N)
    ks = sorted(set(ks))
    hits = {k: 0 for k in ks}
    ranks = []
    valid_queries = 0

    for k in range(z_text.shape[0]):
        pos_idx = np.where(labels_motion == k)[0]
        if len(pos_idx) == 0:
            continue
        valid_queries += 1
        order = np.argsort(-sims[k])  # (N,)
        inv_rank = np.empty_like(order)
        inv_rank[order] = np.arange(order.size)
        best_rank = int(inv_rank[pos_idx].min())
        ranks.append(best_rank + 1)
        for kk in ks:
            if best_rank < kk:
                hits[kk] += 1

    metrics = {f"R@{kk}": hits[kk] / max(valid_queries, 1) for kk in ks}
    medr = float(np.median(ranks)) if ranks else None
    return metrics, medr, valid_queries


def motion_to_text_metrics(
    z_text: np.ndarray,
    z_motion: np.ndarray,
    labels_motion: np.ndarray,
    ks: List[int],
) -> Tuple[Dict[str, float], Optional[float], int]:
    sims = z_motion @ z_text.T  # (N, K)
    order = np.argsort(-sims, axis=1)
    ranks = np.argsort(order, axis=1)
    correct_rank = ranks[np.arange(labels_motion.shape[0]), labels_motion] + 1

    ks = sorted(set(ks))
    metrics = {f"R@{kk}": float(np.mean(correct_rank <= kk)) for kk in ks}
    medr = float(np.median(correct_rank)) if correct_rank.size > 0 else None
    return metrics, medr, int(labels_motion.shape[0])


def compute_silhouette(z_m: np.ndarray, labels_idx: np.ndarray) -> Optional[float]:
    if silhouette_score is None:
        return None
    if z_m.shape[0] < 2:
        return None
    if len(np.unique(labels_idx)) < 2:
        return None
    return float(silhouette_score(z_m, labels_idx))

def compute_motion_to_text_confusion(
    z_text: np.ndarray,
    z_motion: np.ndarray,
    labels_motion: np.ndarray,
) -> Tuple[np.ndarray, Optional[float]]:
    if labels_motion.size == 0:
        return np.zeros((z_text.shape[0], z_text.shape[0]), dtype=np.int64), None
    sims = z_motion @ z_text.T  # (N, K)
    preds = sims.argmax(axis=1)
    cm = np.zeros((z_text.shape[0], z_text.shape[0]), dtype=np.int64)
    for true_id, pred_id in zip(labels_motion, preds):
        if 0 <= int(true_id) < z_text.shape[0] and 0 <= int(pred_id) < z_text.shape[0]:
            cm[int(true_id), int(pred_id)] += 1
    acc1 = float(np.mean(preds == labels_motion))
    return cm, acc1


def compute_text_to_motion_confusion(
    z_text: np.ndarray,
    z_motion: np.ndarray,
    labels_motion: np.ndarray,
) -> Tuple[np.ndarray, Optional[float]]:
    """Text->Motion confusion via class-matched top-N retrieval.

    For each text label i, retrieve top-N_i motion samples, where N_i is the number
    of motion samples whose true label is i. Then count the class distribution of
    retrieved motions into row i. This keeps row totals comparable to Motion->Text.
    """
    k_cls = z_text.shape[0]
    if labels_motion.size == 0:
        return np.zeros((k_cls, k_cls), dtype=np.int64), None

    sims = z_text @ z_motion.T  # (K, N)
    label_counts = np.bincount(labels_motion, minlength=k_cls)
    cm = np.zeros((k_cls, k_cls), dtype=np.int64)

    for text_id in range(k_cls):
        top_n = int(label_counts[text_id])
        if top_n <= 0:
            continue
        order = np.argsort(-sims[text_id])  # (N,)
        top_idx = order[:top_n]
        pred_labels = labels_motion[top_idx]
        for pred_id in pred_labels:
            pred_id = int(pred_id)
            if 0 <= pred_id < k_cls:
                cm[text_id, pred_id] += 1

    total = int(cm.sum())
    diag_ratio = float(np.trace(cm) / total) if total > 0 else None
    return cm, diag_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from latent snapshot.")
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to latent_snapshot_final.npz",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated list of splits to include (e.g., 'test' or 'train,test').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "fine", "coarse"],
        default="auto",
        help="Label granularity for evaluation.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["text2motion", "motion2text", "both"],
        default="both",
        help="Which retrieval direction to compute.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for R@K.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization on embeddings before evaluation.",
    )
    parser.add_argument(
        "--no-silhouette",
        action="store_true",
        help="Skip silhouette score computation.",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Compute confusion matrix (default: motion->text).",
    )
    parser.add_argument(
        "--confusion-out",
        type=str,
        default="",
        help="Optional output path for confusion matrix JSON.",
    )
    parser.add_argument(
        "--confusion-direction",
        type=str,
        choices=["auto", "motion2text", "text2motion"],
        default="auto",
        help=(
            "Direction of confusion matrix. "
            "auto: text2motion when --direction=text2motion, otherwise motion2text."
        ),
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    data = load_snapshot(snapshot_path)
    z_m = data["z_m"]
    labels_idx = data["labels_idx"]
    label_list = data["label_list"]
    z_s_cls = data["z_s_cls"]
    splits = data["splits"]

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else []
    z_m, labels_idx, splits = filter_by_splits(z_m, labels_idx, splits, split_list)

    if args.mode == "auto":
        mode = detect_label_mode(label_list)
    else:
        mode = args.mode

    if mode == "coarse":
        z_m, labels_idx, z_s_cls, label_list = build_coarse_view(z_m, labels_idx, label_list, z_s_cls)

    if not args.no_normalize:
        z_m = normalize(z_m)
        z_s_cls = normalize(z_s_cls)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    print(f"[Snapshot] {snapshot_path}")
    print(f"[Mode] {mode} | labels={len(label_list)} | samples={len(z_m)}")
    if split_list:
        print(f"[Splits] {', '.join(split_list)}")

    # Label counts (optional quick sanity check)
    counts = np.bincount(labels_idx, minlength=len(label_list))
    label_counts = ", ".join([f"{label_list[i]}:{counts[i]}" for i in range(len(label_list))])
    print(f"[Label counts] {label_counts}")

    if args.direction in ("text2motion", "both"):
        t2m, medr, n_q = text_to_motion_metrics(z_s_cls, z_m, labels_idx, ks)
        print("\n[Text -> Motion]")
        print(f"  queries: {n_q}")
        for k in ks:
            print(f"  R@{k}: {t2m[f'R@{k}']:.4f}")
        if medr is not None:
            print(f"  MedR: {medr:.1f}")

    if args.direction in ("motion2text", "both"):
        m2t, medr, n_q = motion_to_text_metrics(z_s_cls, z_m, labels_idx, ks)
        print("\n[Motion -> Text]")
        print(f"  queries: {n_q}")
        for k in ks:
            print(f"  R@{k}: {m2t[f'R@{k}']:.4f}")
        if medr is not None:
            print(f"  MedR: {medr:.1f}")

    if args.confusion:
        if args.confusion_direction == "auto":
            confusion_direction = "text2motion" if args.direction == "text2motion" else "motion2text"
        else:
            confusion_direction = args.confusion_direction

        if confusion_direction == "text2motion":
            cm, diag_ratio = compute_text_to_motion_confusion(z_s_cls, z_m, labels_idx)
            print("\n[Confusion Matrix: Text -> Motion]")
            if diag_ratio is not None:
                print(f"  DiagRatio@count: {diag_ratio:.4f}")
        else:
            cm, acc1 = compute_motion_to_text_confusion(z_s_cls, z_m, labels_idx)
            print("\n[Confusion Matrix: Motion -> Text]")
            if acc1 is not None:
                print(f"  Acc@1: {acc1:.4f}")

        out_path = Path(args.confusion_out) if args.confusion_out else snapshot_path.with_suffix(".confusion.json")
        payload = {
            "labels": label_list,
            "confusion": cm.tolist(),
            "confusion_direction": confusion_direction,
            "mode": mode,
        }
        if confusion_direction == "text2motion":
            payload["diag_ratio@count"] = diag_ratio
        else:
            payload["acc@1"] = acc1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")

    if not args.no_silhouette:
        sil = compute_silhouette(z_m, labels_idx)
        if sil is None:
            print("\n[Silhouette] skipped (insufficient labels or sklearn missing)")
        else:
            print(f"\n[Silhouette] {sil:.4f}")


if __name__ == "__main__":
    main()
