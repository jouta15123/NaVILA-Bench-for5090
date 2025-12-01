import argparse
from pathlib import Path
import sys

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    INSTRUCTION_ONOMATOPEIA,
    apply_normalization_from_stats,
)

# Reuse MotionCLIP model builder from the joint training script
from hoyo_v1_1.models.train_motionclip_joint import (
    HOYO_ROOT,
    load_motionclip_full_model,
)


def _build_dataset(
    hoyo_root: Path,
    target_len: int,
    stats_path: Path,
) -> HoyoInstructionDataset:
    """Load HOYO dataset and apply the same normalization as in training."""
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)
    apply_normalization_from_stats(dataset, INSTRUCTION_ONOMATOPEIA, stats_path)
    return dataset


@torch.no_grad()
def _compute_metrics_for_array(model: torch.nn.Module, arr: np.ndarray, device: torch.device):
    """
    Compute reconstruction metrics for a single motion sequence.

    Args:
        model: MotionCLIP model (full, encoder+decoder).
        arr:  (T, J, 2) numpy array.
    Returns:
        mpjpe: float
        traj_err: float  (COM trajectory error)
        vel_corr: float  (correlation of COM speed time-series)
    """
    # (T, J, 2) -> (1, J, 2, T)
    coords = arr[np.newaxis, ...].transpose(0, 2, 3, 1)
    x = torch.from_numpy(coords).float().to(device)

    B, J, C, Tcur = x.shape
    mask = torch.ones((B, Tcur), dtype=torch.bool, device=device)
    lengths = torch.full((B,), Tcur, dtype=torch.long, device=device)

    batch = {
        "x": x,
        "mask": mask,
        "lengths": lengths,
        "y": torch.zeros((B,), dtype=torch.long, device=device),  # dummy
    }

    out = model(batch)
    rec = out.get("output", out.get("rec", None))
    if rec is None:
        raise RuntimeError("Model output does not contain 'output' or 'rec' for reconstruction.")

    # --- MPJPE (per-sequence) ---
    # diff: (B, J, 2, T)
    diff = rec - x
    per_joint_t = torch.norm(diff, dim=2)  # (B, J, T)
    mpjpe = per_joint_t.mean().item()

    # --- COM (mean joint) trajectory error ---
    # x, rec: (B, J, 2, T) -> (B, T, J, 2)
    orig = x.permute(0, 3, 1, 2)
    recon = rec.permute(0, 3, 1, 2)
    com_orig = orig.mean(dim=2)   # (B, T, 2)
    com_recon = recon.mean(dim=2)  # (B, T, 2)

    traj_err = (com_orig - com_recon).norm(dim=-1).mean().item()

    # --- COM speed time-series correlation (rhythm) ---
    if Tcur > 1:
        vel_orig = (com_orig[:, 1:, :] - com_orig[:, :-1, :]).norm(dim=-1).squeeze(0)   # (T-1,)
        vel_recon = (com_recon[:, 1:, :] - com_recon[:, :-1, :]).norm(dim=-1).squeeze(0)

        a = vel_orig - vel_orig.mean()
        b = vel_recon - vel_recon.mean()
        num = (a * b).sum()
        den = a.norm() * b.norm() + 1e-8
        vel_corr = (num / den).item()
    else:
        vel_corr = float("nan")

    return mpjpe, traj_err, vel_corr


def _plot_box_by_label(values, labels_idx, label_names, ylabel, out_path: Path):
    """Simple per-label boxplot."""
    import pandas as pd

    data = {
        "value": values,
        "label": [label_names[i] for i in labels_idx],
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="label", y="value")
    plt.ylabel(ylabel)
    plt.xlabel("Label")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {ylabel} boxplot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MotionCLIP reconstruction quality on HOYO.")
    parser.add_argument(
        "--hoyo-root",
        type=str,
        default=str(HOYO_ROOT),
        help="Root directory of HOYO data (default: inferred from project).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results" / "checkpoints" / "motionclip_full_joint_best.pth"),
        help="Path to MotionCLIP full model checkpoint (.pth).",
    )
    parser.add_argument(
        "--normalization-stats",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results" / "normalization_stats.json"),
        help="Path to normalization_stats.json used during training.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(HOYO_ROOT / "joint_training_results"),
        help="Directory to save analysis outputs (npz, plots).",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=60,
        help="Target sequence length (frames). Must match training.",
    )
    args = parser.parse_args()

    hoyo_root = Path(args.hoyo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Load dataset with the same normalization as training
    stats_path = Path(args.normalization_stats)
    if not stats_path.exists():
        print(f"[Warning] normalization_stats.json not found at {stats_path}", file=sys.stderr)
    dataset = _build_dataset(hoyo_root, args.target_len, stats_path)

    # 2) Load model
    model, _ = load_motionclip_full_model(device=device, target_len=args.target_len)
    ckpt_path = Path(args.checkpoint)
    print(f"Loading model checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3) Iterate over all samples and collect metrics
    all_mpjpe = []
    all_traj_err = []
    all_vel_corr = []
    all_labels_idx = []

    label_to_idx = {lab: i for i, lab in enumerate(INSTRUCTION_ONOMATOPEIA)}

    print("Computing reconstruction metrics per sequence...")
    for lab in INSTRUCTION_ONOMATOPEIA:
        samples = dataset.samples_by_label.get(lab, [])
        if not samples:
            continue
        lab_idx = label_to_idx[lab]

        for arr in samples:
            mpjpe, traj_err, vel_corr = _compute_metrics_for_array(model, arr, device)
            all_mpjpe.append(mpjpe)
            all_traj_err.append(traj_err)
            all_vel_corr.append(vel_corr)
            all_labels_idx.append(lab_idx)

    all_mpjpe = np.asarray(all_mpjpe, dtype=np.float32)
    all_traj_err = np.asarray(all_traj_err, dtype=np.float32)
    all_vel_corr = np.asarray(all_vel_corr, dtype=np.float32)
    all_labels_idx = np.asarray(all_labels_idx, dtype=np.int64)

    # 4) Save raw metrics
    out_npz = out_dir / "reconstruction_metrics.npz"
    np.savez(
        out_npz,
        mpjpe=all_mpjpe,
        traj_err=all_traj_err,
        vel_corr=all_vel_corr,
        labels_idx=all_labels_idx,
        label_list=np.asarray(INSTRUCTION_ONOMATOPEIA),
    )
    print(f"Saved reconstruction metrics to {out_npz}")

    # 5) Print summary
    def _summ(name, arr):
        return (
            f"{name}: mean={float(np.nanmean(arr)):.4f}, "
            f"median={float(np.nanmedian(arr)):.4f}, "
            f"std={float(np.nanstd(arr)):.4f}"
        )

    print("=== Global statistics ===")
    print(_summ("MPJPE", all_mpjpe))
    print(_summ("COM trajectory error", all_traj_err))
    print(_summ("COM speed correlation", all_vel_corr))

    print("\n=== Per-label MPJPE (mean) ===")
    for i, lab in enumerate(INSTRUCTION_ONOMATOPEIA):
        mask = all_labels_idx == i
        if not np.any(mask):
            continue
        print(f"  {lab}: mean MPJPE={float(all_mpjpe[mask].mean()):.4f}")

    # 6) Simple per-label boxplots
    _plot_box_by_label(all_mpjpe, all_labels_idx, INSTRUCTION_ONOMATOPEIA, "MPJPE", fig_dir / "mpjpe_boxplot.png")
    _plot_box_by_label(
        all_traj_err,
        all_labels_idx,
        INSTRUCTION_ONOMATOPEIA,
        "COM trajectory error",
        fig_dir / "traj_error_boxplot.png",
    )
    _plot_box_by_label(
        all_vel_corr,
        all_labels_idx,
        INSTRUCTION_ONOMATOPEIA,
        "COM speed correlation",
        fig_dir / "vel_corr_boxplot.png",
    )


if __name__ == "__main__":
    main()





