import argparse
import os
import sys
import random
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path so that `hoyo_v1_1` can be imported
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse utilities and constants from the joint-training script
from hoyo_v1_1.models.train_motionclip_joint import (
    HOYO_ROOT,
    COARSE_GROUPS,
    COARSE_LABELS,
    split_dataset,
    build_coarse_dataset,
    load_motionclip_full_model,
    evaluate_dataset,
    dump_latent_snapshot,
    Tee,
)

from hoyo_v1_1.models.common import (
    HoyoInstructionDataset,
    encode_semantics_sarashina,
    encode_semantics_siglip,
    INSTRUCTION_ONOMATOPEIA,
    normalize_dataset,
    apply_normalization_from_stats,
)


@torch.no_grad()
def compute_motion_prototypes(
    dataset: HoyoInstructionDataset,
    model: nn.Module,
    device: torch.device,
    labels,
    max_per_label: int = 200,
):
    """
    Compute motion-space prototypes for each label by averaging MotionCLIP latents.

    For each label g, we run the (frozen) MotionCLIP encoder on up to
    `max_per_label` samples and take the mean of the normalized μ vectors:
        m_g = mean_i normalize(mu_i)

    Returns:
        prototypes: (G, D) tensor on `device`
    """
    model.eval()
    labels = list(labels)

    all_protos = []
    for lab in labels:
        samples = dataset.samples_by_label.get(lab, [])
        if not samples:
            print(f"[Proto] Warning: label '{lab}' has no samples, using zero prototype.")
            all_protos.append(None)
            continue

        n_take = min(len(samples), max_per_label)
        chosen = random.sample(samples, n_take)

        latents = []
        for arr in chosen:
            # arr: (T, 14, 2) -> (1, J, C, T)
            coords = arr[np.newaxis, ...].transpose(0, 2, 3, 1)
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
            z_m = out["mu"]  # (1, D)
            z_m = F.normalize(z_m, dim=-1).squeeze(0)
            latents.append(z_m)

        if not latents:
            print(f"[Proto] Warning: label '{lab}' had no valid latents, using zero prototype.")
            all_protos.append(None)
            continue

        latents_tensor = torch.stack(latents, dim=0)  # (N, D)
        proto = latents_tensor.mean(dim=0)  # (D,)
        all_protos.append(proto)
        print(f"[Proto] Label '{lab}': used {len(latents)} samples.")

    # Replace missing prototypes (if any) with zeros
    dim = model.latent_dim
    for i, p in enumerate(all_protos):
        if p is None:
            all_protos[i] = torch.zeros(dim, device=device)

    prototypes = torch.stack(all_protos, dim=0)  # (G, D)
    prototypes = F.normalize(prototypes, dim=-1)
    print(f"[Proto] Computed motion prototypes with shape {tuple(prototypes.shape)}")
    return prototypes


def train_semantic_to_prototypes(
    train_dataset: HoyoInstructionDataset,
    test_dataset: HoyoInstructionDataset,
    model: nn.Module,
    sem_emb: torch.Tensor,
    motion_protos: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    steps: int = 2000,
    temp: float = 0.07,
    lr: float = 1e-4,
    log_interval: int = 100,
    eval_interval: int = 200,
    labels=None,
):
    """
    Prototype-based contrastive learning (案2).

    - MotionCLIP encoder is kept frozen.
    - For each coarse/fine label g, we have a motion prototype m_g in latent space.
    - We learn a linear projector sem_proj such that
          z_s(g) = sem_proj(e_text(g))
      is close to m_g and far from other prototypes.

    Concretely, we minimize cross-entropy over prototypes:
        logits[g, h] = exp(logit_scale) * <z_s(g), m_h>
        L = CE(logits, target = g)

    We do not update the VAE in this script; it is purely sem → motion alignment.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Projector for semantics -> motion latent space
    d_motion = model.latent_dim
    d_sem = sem_emb.shape[1]

    sem_proj = nn.Linear(d_sem, d_motion, bias=False).to(device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temp))

    optimizer = torch.optim.AdamW(
        [
            {"params": sem_proj.parameters(), "lr": lr},
            {"params": [logit_scale], "lr": lr},
        ]
    )

    if labels is None:
        labels = INSTRUCTION_ONOMATOPEIA
    labels = list(labels)

    sem_emb = sem_emb.to(device)
    motion_protos = motion_protos.to(device)

    best_test_acc = -1.0
    running_stats = {
        "total_loss": 0.0,
        "train_acc1": 0.0,
        "steps": 0,
    }

    print(f"[ProtoTrain] Start training sem → prototypes for {steps} steps...")
    print(
        f"[ProtoTrain] Config: steps={steps}, lr={lr}, temp={temp}, "
        f"n_labels={len(labels)}"
    )

    for step in range(1, steps + 1):
        model.eval()  # encoder is always frozen here
        sem_proj.train()

        # Forward: compute semantic projections and logits against motion prototypes
        z_s = sem_proj(sem_emb)  # (G, D)
        z_s = F.normalize(z_s, dim=-1)

        # (G, G) logits: each row g compares z_s(g) to all prototypes m_h
        logits = torch.matmul(z_s, motion_protos.t())
        logits = torch.exp(logit_scale) * logits

        targets = torch.arange(len(labels), dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)

        if torch.isnan(loss):
            print(f"[ProtoTrain] Loss is NaN at step {step}, stopping.")
            break

        # Train accuracy in prototype space
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc1 = (preds == targets).float().mean().item()
            current_temp = float(torch.exp(-logit_scale.detach()).item())

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients over sem_proj + logit_scale
        all_params = list(sem_proj.parameters()) + [logit_scale]
        torch.nn.utils.clip_grad_norm_(all_params, 0.5)
        optimizer.step()

        # Clamp logit_scale to keep temperature in a reasonable range.
        # Use a narrower range [1/10, 10] to avoid extreme temperatures.
        with torch.no_grad():
            logit_scale.data.clamp_(np.log(1.0 / 10.0), np.log(10.0))

        # Update running stats
        running_stats["total_loss"] += float(loss.item())
        running_stats["train_acc1"] += float(acc1)
        running_stats["steps"] += 1

        # Logging
        if step % log_interval == 0 or step == 1:
            denom = max(running_stats["steps"], 1)
            avg_loss = running_stats["total_loss"] / denom
            avg_acc1 = running_stats["train_acc1"] / denom

            print(
                f"[ProtoTrain][Step {step:05d}] "
                f"loss={avg_loss:.4f}, train_acc@1={avg_acc1:.3f}, "
                f"temp={current_temp:.4f}"
            )

            running_stats = {
                "total_loss": 0.0,
                "train_acc1": 0.0,
                "steps": 0,
            }

        # Periodic evaluation using motion→semantic classification
        if step % eval_interval == 0 or step == steps:
            print("[ProtoTrain] Evaluating on test set (motion → semantic prototypes)...")
            metrics = evaluate_dataset(
                test_dataset,
                model,
                sem_proj,
                sem_emb,
                device,
                labels=labels,
            )
            test_acc = metrics["acc@1"]
            print(
                f"  [TEST] Acc@1: {metrics['acc@1']:.3f} | "
                f"Acc@3: {metrics['acc@3']:.3f} | MPJPE: {metrics['mpjpe']:.4f}"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"  >>> New Best Test Acc: {best_test_acc:.3f}! Saving proto model...")
                torch.save(sem_proj.state_dict(), out_dir / "sem_proj_proto_best.pth")
                torch.save(
                    {"logit_scale": logit_scale.detach().cpu()},
                    out_dir / "logit_scale_proto_best.pt",
                )

    # Final save
    print(f"[ProtoTrain] Saving final proto models to {out_dir}")
    torch.save(sem_proj.state_dict(), out_dir / "sem_proj_proto_final.pth")
    torch.save(
        {"logit_scale": logit_scale.detach().cpu()},
        out_dir / "logit_scale_proto_final.pt",
    )

    # Save prototypes for later analysis / RL usage
    proto_np = motion_protos.detach().cpu().numpy()
    labels_np = np.asarray(labels)
    np.savez(out_dir / "motion_prototypes.npz", prototypes=proto_np, labels=labels_np)

    # Also dump a latent snapshot using the learned sem_proj
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
        max_per_label=50,
    )

    return sem_proj


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype-based sem ↔ motion training for HOYO + MotionCLIP (案2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["fine", "coarse"],
        default="coarse",
        help="Label granularity: 'fine' (11オノマトペ) or 'coarse' (4スタイル群)",
    )
    parser.add_argument(
        "--sem-encoder",
        choices=["sarashina", "siglip"],
        default="sarashina",
        help="Which semantic encoder to use for text embeddings.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.07,
        help="Contrastive temperature (shared via logit_scale).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval (steps).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=200,
        help="Evaluation interval (steps).",
    )
    parser.add_argument(
        "--max-proto-samples-per-label",
        type=int,
        default=200,
        help="Maximum number of motion samples per label used to build prototypes.",
    )
    parser.add_argument(
        "--motionclip-ckpt",
        type=str,
        default=None,
        help=(
            "Optional path to a pretrained MotionCLIP checkpoint "
            "(e.g., joint_training_results/motionclip_full_joint_best.pth). "
            "If not provided, the default HOYO-adapted MotionCLIP will be used."
        ),
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
    out_dir = hoyo_root / "proto_training_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging (tee) to a timestamped log file
    log_path = out_dir / f"train_proto_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"[Logger] Writing proto training logs to: {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    target_len = 60

    # Load dataset (fine labels)
    dataset = HoyoInstructionDataset(hoyo_root, INSTRUCTION_ONOMATOPEIA, target_len=target_len)

    # Split train / test
    train_dataset, test_dataset = split_dataset(dataset, ratio=0.8)

    # Semantic embeddings for fine labels
    if args.sem_encoder == "sarashina":
        print("Using Sarashina sentence embeddings for semantics.")
        sem_emb_fine = encode_semantics_sarashina(INSTRUCTION_ONOMATOPEIA, device=device)
    elif args.sem_encoder == "siglip":
        print("Using SigLIP text encoder for semantics.")
        sem_emb_fine = encode_semantics_siglip(INSTRUCTION_ONOMATOPEIA, device=device)
    else:
        raise ValueError(f"Unknown sem-encoder: {args.sem_encoder}")

    # Normalize TRAIN data and persist stats, then normalize TEST with the same stats
    stats_path = hoyo_root / "joint_training_results" / "normalization_stats.json"
    # Reuse stats from joint training if available; otherwise compute new ones.
    if stats_path.exists():
        print(f"[Norm] Using existing normalization stats: {stats_path}")
        apply_normalization_from_stats(train_dataset, INSTRUCTION_ONOMATOPEIA, stats_path)
        apply_normalization_from_stats(test_dataset, INSTRUCTION_ONOMATOPEIA, stats_path)
    else:
        print(f"[Norm] No existing stats found, computing new normalization stats at {stats_path}")
        normalize_dataset(train_dataset, INSTRUCTION_ONOMATOPEIA, stats_path)
        apply_normalization_from_stats(test_dataset, INSTRUCTION_ONOMATOPEIA, stats_path)

    # Optionally convert to coarse style groups
    if args.label_mode == "coarse":
        print("Using COARSE style labels:")
        for coarse, fines in COARSE_GROUPS.items():
            print(f"  {coarse}: {', '.join(fines)}")

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

    # Load MotionCLIP model (frozen)
    model, _ = load_motionclip_full_model(device, target_len)
    model.to(device)

    # Optionally load a pretrained HOYO-finetuned checkpoint
    if args.motionclip_ckpt is not None:
        ckpt_path = Path(args.motionclip_ckpt)
        if ckpt_path.is_file():
            print(f"[MotionCLIP] Loading checkpoint from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            try:
                model.load_state_dict(state_dict)
                print("[MotionCLIP] Loaded full state_dict from checkpoint.")
            except RuntimeError as e:
                print(f"[MotionCLIP] Failed to load full state_dict: {e}")
        else:
            print(f"[MotionCLIP] Warning: motionclip_ckpt '{ckpt_path}' not found; using default weights.")

    # Freeze the MotionCLIP encoder/decoder
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Compute motion prototypes on the TRAIN split
    motion_protos = compute_motion_prototypes(
        train_dataset,
        model,
        device,
        labels_for_training,
        max_per_label=args.max_proto_samples_per_label,
    )

    # Train semantic projector towards these prototypes
    train_semantic_to_prototypes(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        sem_emb=sem_emb,
        motion_protos=motion_protos,
        device=device,
        out_dir=out_dir,
        steps=args.steps,
        temp=args.temp,
        lr=args.lr,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        labels=labels_for_training,
    )


if __name__ == "__main__":
    main()


