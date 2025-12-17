"""
Utility to pad a base policy checkpoint (Actor-Critic) so it matches
the current observation dimensions.

Usage example:
    python scripts/convert_base_ckpt.py \
        --ckpt logs/rsl_rl/h1_vision_rough/2024-11-03_15-08-09_height_scan_obst/model_4999.pt \
        --target_actor_in 365 \
        --target_critic_in 365 \
        --out logs/rsl_rl/h1_vision_rough/2024-11-03_15-08-09_height_scan_obst/model_4999_padded365.pt

Assumptions:
- The checkpoint was saved by rsl_rl OnPolicyRunner and contains either
  "model_state_dict" or the raw state_dict.
- Only the first Linear layers of actor/critic need in_features expansion.
- New columns are zero-initialized (safe for fine-tuning).
"""

import argparse
import os
import torch


def load_state_dict(ckpt_path: str):
    data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data, dict) and "model_state_dict" in data:
        return data, data["model_state_dict"]
    if isinstance(data, dict):
        return data, data
    raise ValueError("Unsupported checkpoint format")


def pad_linear_weight(weight: torch.Tensor, target_in: int):
    """Pad Linear weight (out, in) on input dimension with zeros."""
    out_features, in_features = weight.shape
    if target_in == in_features:
        return weight
    if target_in < in_features:
        # truncate if user requested smaller (rare)
        return weight[:, :target_in]
    padded = torch.zeros(out_features, target_in, dtype=weight.dtype)
    padded[:, :in_features] = weight
    return padded


def pad_linear_bias(bias: torch.Tensor, target_out: int):
    if target_out == bias.shape[0]:
        return bias
    if target_out < bias.shape[0]:
        return bias[:target_out]
    padded = torch.zeros(target_out, dtype=bias.dtype)
    padded[: bias.shape[0]] = bias
    return padded


def apply_padding(sd, target_actor_in: int, target_critic_in: int):
    # actor first layer
    a_w_key = "actor.0.weight"
    a_b_key = "actor.0.bias"
    c_w_key = "critic.0.weight"
    c_b_key = "critic.0.bias"

    if a_w_key not in sd or a_b_key not in sd:
        raise KeyError("actor first layer not found in state_dict")
    if c_w_key not in sd or c_b_key not in sd:
        raise KeyError("critic first layer not found in state_dict")

    sd[a_w_key] = pad_linear_weight(sd[a_w_key], target_actor_in)
    sd[a_b_key] = pad_linear_bias(sd[a_b_key], sd[a_w_key].shape[0])

    sd[c_w_key] = pad_linear_weight(sd[c_w_key], target_critic_in)
    sd[c_b_key] = pad_linear_bias(sd[c_b_key], sd[c_w_key].shape[0])

    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to base checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="output path for padded checkpoint")
    parser.add_argument("--target_actor_in", type=int, required=True, help="new actor in_features")
    parser.add_argument("--target_critic_in", type=int, required=True, help="new critic in_features")
    args = parser.parse_args()

    meta, sd = load_state_dict(args.ckpt)
    sd_padded = apply_padding(sd, args.target_actor_in, args.target_critic_in)

    # if original had model_state_dict, update it; else save sd directly
    if "model_state_dict" in meta:
        meta["model_state_dict"] = sd_padded
        to_save = meta
    else:
        to_save = sd_padded

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(to_save, args.out)
    print(f"Saved padded checkpoint to {args.out}")
    print(f"actor in_features: {args.target_actor_in}, critic in_features: {args.target_critic_in}")


if __name__ == "__main__":
    main()
