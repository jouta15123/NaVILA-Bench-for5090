#!/usr/bin/env python3
"""
Pad base policy checkpoint to match current observation dimensions.

This script zero-pads the input dimension of actor/critic first layers
and (optionally) observation normalizer buffers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def pad_linear_weight(weight: torch.Tensor, target_in: int, truncate: bool) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    if target_in == in_dim:
        return weight
    if target_in < in_dim:
        if not truncate:
            raise ValueError(f"Target dim {target_in} < current dim {in_dim} (use --truncate to allow).")
        return weight[:, :target_in].contiguous()
    pad = torch.zeros((out_dim, target_in - in_dim), dtype=weight.dtype, device=weight.device)
    return torch.cat([weight, pad], dim=1)


def pad_norm_buffer(buf: torch.Tensor, target_dim: int, fill: float) -> torch.Tensor:
    # Expect shape (1, D) or (D,)
    if buf.ndim == 2 and buf.shape[0] == 1:
        buf = buf.squeeze(0)
    cur_dim = buf.shape[-1]
    if target_dim == cur_dim:
        return buf
    if target_dim < cur_dim:
        raise ValueError(f"Target dim {target_dim} < current dim {cur_dim} for norm buffer.")
    pad = torch.full((target_dim - cur_dim,), fill, dtype=buf.dtype, device=buf.device)
    return torch.cat([buf, pad], dim=0)


def pad_norm_state(norm_state: dict, target_dim: int) -> dict:
    if target_dim is None:
        return norm_state
    out = dict(norm_state)
    if "_mean" in out:
        out["_mean"] = pad_norm_buffer(out["_mean"], target_dim, fill=0.0).unsqueeze(0)
    if "_var" in out:
        out["_var"] = pad_norm_buffer(out["_var"], target_dim, fill=1.0).unsqueeze(0)
    if "_std" in out:
        out["_std"] = pad_norm_buffer(out["_std"], target_dim, fill=1.0).unsqueeze(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad base policy checkpoint input dims.")
    parser.add_argument("--src", required=True, help="Source checkpoint path.")
    parser.add_argument("--dst", required=True, help="Destination checkpoint path.")
    parser.add_argument("--actor-in", type=int, required=True, help="Target actor input dim.")
    parser.add_argument("--critic-in", type=int, required=True, help="Target critic input dim.")
    parser.add_argument("--obs-dim", type=int, default=None, help="Target obs dim for obs_norm_state_dict.")
    parser.add_argument("--critic-obs-dim", type=int, default=None, help="Target obs dim for critic_obs_norm_state_dict.")
    parser.add_argument("--truncate", action="store_true", help="Allow truncation when target dim is smaller.")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    for prefix, target in (("actor.0", args.actor_in), ("critic.0", args.critic_in)):
        w_key = f"{prefix}.weight"
        if w_key not in state:
            print(f"[WARN] Missing {w_key} in checkpoint; skipping.")
            continue
        w = state[w_key]
        old = tuple(w.shape)
        state[w_key] = pad_linear_weight(w, target, args.truncate)
        new = tuple(state[w_key].shape)
        print(f"[PAD] {w_key}: {old} -> {new}")

    # Pad normalizer buffers if present
    if "obs_norm_state_dict" in ckpt:
        ckpt["obs_norm_state_dict"] = pad_norm_state(ckpt["obs_norm_state_dict"], args.obs_dim)
        print("[PAD] obs_norm_state_dict updated.")
    if "critic_obs_norm_state_dict" in ckpt:
        ckpt["critic_obs_norm_state_dict"] = pad_norm_state(ckpt["critic_obs_norm_state_dict"], args.critic_obs_dim)
        print("[PAD] critic_obs_norm_state_dict updated.")

    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = state
        torch.save(ckpt, dst)
    else:
        torch.save(state, dst)
    print(f"Saved padded checkpoint to: {dst}")


if __name__ == "__main__":
    main()
