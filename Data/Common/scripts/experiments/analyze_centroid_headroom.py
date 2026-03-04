#!/usr/bin/env python3
"""Analyze centroid separability/headroom for hard-negative reward design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_rows(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _as_text_list(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in arr.tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore").strip())
        else:
            out.append(str(v).strip())
    return out


def _float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _quantile(vals: np.ndarray, q: float) -> float | None:
    if vals.size == 0:
        return None
    return float(np.quantile(vals, q))


def build_report(snapshot: Path, margin: float, split_filter: str | None = None) -> dict[str, Any]:
    data = np.load(snapshot)
    required = {"z_m", "labels_idx", "label_list"}
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in snapshot: {missing}")

    z_m = np.asarray(data["z_m"], dtype=np.float32)
    labels_idx = np.asarray(data["labels_idx"], dtype=np.int64)
    label_list = _as_text_list(np.asarray(data["label_list"]))

    if z_m.ndim != 2 or labels_idx.ndim != 1 or z_m.shape[0] != labels_idx.shape[0]:
        raise ValueError("Invalid snapshot shape: z_m and labels_idx are inconsistent.")

    if split_filter is not None:
        if "splits" not in data:
            raise KeyError("split_filter is set but 'splits' key is missing in snapshot.")
        splits = np.asarray(data["splits"]).astype(str)
        mask = splits == split_filter
        z_m = z_m[mask]
        labels_idx = labels_idx[mask]

    z_m = _normalize_rows(z_m)
    uniq_labels = sorted(set(int(x) for x in np.unique(labels_idx).tolist()))
    if len(uniq_labels) < 2:
        raise ValueError("Need at least 2 labels for separability analysis.")

    # Build centroids from z_m (same basis as style_module.class_centroids).
    id_to_col: dict[int, int] = {}
    centroid_names: list[str] = []
    centroids: list[np.ndarray] = []
    for col, lab in enumerate(uniq_labels):
        class_z = z_m[labels_idx == lab]
        c = class_z.mean(axis=0, dtype=np.float64)
        c = c / max(float(np.linalg.norm(c)), 1.0e-12)
        id_to_col[lab] = col
        centroids.append(c.astype(np.float32))
        centroid_names.append(label_list[lab] if 0 <= lab < len(label_list) else f"label_{lab}")
    cmat = np.stack(centroids, axis=0)  # (C, D)

    sample_sims = z_m @ cmat.T  # (N, C)
    target_cols = np.asarray([id_to_col[int(l)] for l in labels_idx], dtype=np.int64)
    target_sim = sample_sims[np.arange(sample_sims.shape[0]), target_cols]

    # r_neg: max over non-target centroids
    mask = np.ones_like(sample_sims, dtype=bool)
    mask[np.arange(sample_sims.shape[0]), target_cols] = False
    neg_sims = np.where(mask, sample_sims, -np.inf)
    r_neg = np.max(neg_sims, axis=1)

    # Sample-level margins/headroom
    headroom = target_sim - r_neg
    penalty = np.maximum(0.0, r_neg - target_sim + margin)

    # Centroid-level pairwise similarities
    centroid_pair = cmat @ cmat.T
    offdiag_mask = ~np.eye(centroid_pair.shape[0], dtype=bool)
    offdiag = centroid_pair[offdiag_mask]

    # Top-1 centroid classification
    pred_col = np.argmax(sample_sims, axis=1)
    top1_acc = float(np.mean(pred_col == target_cols))

    # Per-label diagnostics
    per_label: list[dict[str, Any]] = []
    for lab in uniq_labels:
        col = id_to_col[lab]
        name = label_list[lab] if 0 <= lab < len(label_list) else f"label_{lab}"
        row_mask = labels_idx == lab
        row_target = target_sim[row_mask]
        row_neg = r_neg[row_mask]
        row_headroom = headroom[row_mask]

        nearest_other = np.max(np.delete(centroid_pair[col], col))
        q25_self = _quantile(row_target, 0.25)
        label_headroom = None if q25_self is None else float(q25_self - nearest_other)
        margin_safe = None if label_headroom is None else bool(label_headroom >= margin)

        per_label.append(
            {
                "label_id": int(lab),
                "label": name,
                "count": int(row_mask.sum()),
                "self_sim_mean": _mean(row_target.tolist()),
                "self_sim_q25": q25_self,
                "neg_sim_mean": _mean(row_neg.tolist()),
                "headroom_mean": _mean(row_headroom.tolist()),
                "headroom_q10": _quantile(row_headroom, 0.10),
                "nearest_other_centroid_sim": float(nearest_other),
                "label_headroom_q25_minus_nearest": label_headroom,
                "margin_safe": margin_safe,
            }
        )

    safe_count = sum(1 for r in per_label if r["margin_safe"] is True)
    unsafe_count = sum(1 for r in per_label if r["margin_safe"] is False)

    verdict = "ok"
    if unsafe_count > 0:
        verdict = "caution"
    if unsafe_count >= max(1, len(per_label) // 2):
        verdict = "risky"

    suggested_margin = _quantile(headroom, 0.10)
    if suggested_margin is not None:
        suggested_margin = float(max(0.0, min(suggested_margin, margin)))

    return {
        "snapshot": str(snapshot),
        "sample_count": int(z_m.shape[0]),
        "class_count": int(len(uniq_labels)),
        "margin_assumed": float(margin),
        "split_filter": split_filter,
        "metrics": {
            "centroid_top1_accuracy": top1_acc,
            "sample_headroom_mean": _mean(headroom.tolist()),
            "sample_headroom_q10": _quantile(headroom, 0.10),
            "sample_headroom_q25": _quantile(headroom, 0.25),
            "penalty_trigger_rate": float(np.mean(penalty > 0.0)),
            "penalty_mean": _mean(penalty.tolist()),
            "centroid_pairwise_offdiag_mean": _mean(offdiag.tolist()),
            "centroid_pairwise_offdiag_min": _float(np.min(offdiag)),
            "centroid_pairwise_offdiag_max": _float(np.max(offdiag)),
            "label_margin_safe_count": int(safe_count),
            "label_margin_unsafe_count": int(unsafe_count),
        },
        "suggested_margin_max_q10_headroom": suggested_margin,
        "verdict": verdict,
        "per_label": per_label,
    }


def _print_summary(report: dict[str, Any]) -> None:
    m = report["metrics"]
    print("=== Centroid Headroom Check ===")
    print(f"snapshot: {report['snapshot']}")
    print(
        f"samples={report['sample_count']} classes={report['class_count']} "
        f"margin={report['margin_assumed']} split={report['split_filter']}"
    )
    print(
        "top1={:.4f} headroom_mean={:.4f} headroom_q10={} penalty_rate={:.4f} penalty_mean={:.4f}".format(
            float(m["centroid_top1_accuracy"]),
            float(m["sample_headroom_mean"]) if m["sample_headroom_mean"] is not None else float("nan"),
            m["sample_headroom_q10"],
            float(m["penalty_trigger_rate"]),
            float(m["penalty_mean"]) if m["penalty_mean"] is not None else float("nan"),
        )
    )
    print(
        "centroid_offdiag mean={:.4f} min={:.4f} max={:.4f}".format(
            float(m["centroid_pairwise_offdiag_mean"]) if m["centroid_pairwise_offdiag_mean"] is not None else float("nan"),
            float(m["centroid_pairwise_offdiag_min"]) if m["centroid_pairwise_offdiag_min"] is not None else float("nan"),
            float(m["centroid_pairwise_offdiag_max"]) if m["centroid_pairwise_offdiag_max"] is not None else float("nan"),
        )
    )
    print(
        f"label_margin_safe={m['label_margin_safe_count']} "
        f"unsafe={m['label_margin_unsafe_count']} verdict={report['verdict']}"
    )
    print(f"suggested_margin_max_q10_headroom={report['suggested_margin_max_q10_headroom']}")

    print("--- Unsafe labels (q25-self - nearest < margin) ---")
    margin = float(report["margin_assumed"])
    any_unsafe = False
    for row in report["per_label"]:
        lh = row["label_headroom_q25_minus_nearest"]
        if lh is not None and lh < margin:
            any_unsafe = True
            print(
                f"{row['label']}: label_headroom={lh:.4f}, "
                f"nearest_other={row['nearest_other_centroid_sim']:.4f}, self_q25={row['self_sim_q25']:.4f}"
            )
    if not any_unsafe:
        print("(none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze centroid headroom for hardneg reward.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to latent_snapshot_final.npz")
    parser.add_argument("--margin", type=float, default=0.05, help="Assumed hardneg margin.")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter, e.g., test")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional report JSON path.")
    args = parser.parse_args()

    report = build_report(args.snapshot, margin=args.margin, split_filter=args.split)
    _print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()

