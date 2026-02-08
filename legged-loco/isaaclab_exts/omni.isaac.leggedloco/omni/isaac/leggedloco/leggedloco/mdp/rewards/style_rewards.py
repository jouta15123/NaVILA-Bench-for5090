
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_LOG_EVERY_N_STEPS = 10  # Log every N steps to avoid overhead

def style_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    beta_text: float = 0.0,
    beta_teacher_motion: float = 1.0,
    beta_centroid: float | None = None,
    ramp_steps: int = 0,
    style_reward_mode: str = "legacy",
    style_neg_weight: float = 0.0,
    style_neg_margin: float = 0.0,
) -> torch.Tensor:
    """
    Reward based on cosine similarity between agent's motion latent and target style latent.
    Updates the motion buffer in StyleCommandGenerator as a side effect.

    Args:
        command_name: Name of the style command generator in command manager.
    """
    # Use env.extras to store log counter (avoids global state issues in distributed training)
    if not hasattr(env, "extras") or "_style_log_counter" not in env.extras:
        if hasattr(env, "extras"):
            env.extras["_style_log_counter"] = 0
    
    # Access generator
    try:
        # env.command_manager is a CommandManager
        # _terms is a dict of CommandTerm
        cmd_gen = env.command_manager._terms[command_name]
    except KeyError:
        # Fallback if command manager behaves differently or command not found
        return torch.zeros(env.num_envs, device=env.device)

    # 1. Update Buffer with current state
    # Assumes 'robot' is the central articulation in the scene
    # This might need adjustment if robot name varies, but 'robot' is standard in leggedloco templates.
    if hasattr(env.scene, "robot"):
        robot = env.scene.robot
    elif "robot" in env.scene.keys():
        robot = env.scene["robot"]
    else:
        # Fallback loop
        robot = None
        for name in env.scene.keys():
            entity = env.scene[name]
            if hasattr(entity, "data") and hasattr(entity.data, "body_pos_w"):
                robot = entity
                break
    
    if robot is None:
        return torch.zeros(env.num_envs, device=env.device)

    body_pos_w = robot.data.body_pos_w # (B, NumBodies, 3)
    body_quat_w = robot.data.body_quat_w # (B, NumBodies, 4)
    root_quat_w = robot.data.root_quat_w # (B, 4)
    body_names = robot.data.body_names
    
    cmd_gen.style_module.update_buffer(body_pos_w, root_quat_w, body_names, body_quat_w)
    
    # 2. Get Targets
    target_z_onm = cmd_gen.style_latents
    target_teacher_motion = cmd_gen.teacher_motion_latents
    
    # 3. Compute positive reward (legacy formula)
    if beta_centroid is not None:
        beta_teacher_motion = beta_centroid
    reward, r_text, r_teacher_motion = cmd_gen.style_module.compute_current_reward(
        target_z_onm, target_teacher_motion, beta_text, beta_teacher_motion
    )
    reward_raw = reward
    requested_mode = str(style_reward_mode).strip().lower()
    if requested_mode not in {"legacy", "hardneg"}:
        requested_mode = "legacy"
    effective_mode = requested_mode
    if requested_mode == "hardneg" and style_neg_weight <= 0.0:
        effective_mode = "legacy"
    style_neg_sim = torch.zeros_like(reward_raw)
    style_neg_penalty = torch.zeros_like(reward_raw)

    # 4. Optional hard-negative penalty
    if requested_mode == "hardneg" and style_neg_weight > 0.0:
        try:
            if not hasattr(cmd_gen, "current_label_ids"):
                raise RuntimeError("current_label_ids is not available")
            class_centroids = getattr(cmd_gen.style_module, "class_centroids", None)
            if not isinstance(class_centroids, dict) or len(class_centroids) < 2:
                raise RuntimeError("class_centroids are not available")

            if isinstance(cmd_gen.current_label_ids, torch.Tensor):
                label_ids = cmd_gen.current_label_ids.to(device=env.device, dtype=torch.long).view(-1)
            elif isinstance(cmd_gen.current_label_ids, (list, tuple)):
                if len(cmd_gen.current_label_ids) != env.num_envs:
                    raise RuntimeError("current_label_ids size mismatch")
                label_ids = torch.tensor(cmd_gen.current_label_ids, device=env.device, dtype=torch.long)
            else:
                raise RuntimeError("current_label_ids type is unsupported")

            z_agent = cmd_gen.style_module.encode_buffer()
            z_agent = torch.nan_to_num(z_agent, nan=0.0)
            z_agent = torch.nn.functional.normalize(z_agent, dim=-1)

            centroid_ids = []
            centroid_vectors = []
            for label_id, centroid in class_centroids.items():
                try:
                    label_id_int = int(label_id)
                except Exception:
                    continue
                if not isinstance(centroid, torch.Tensor):
                    centroid = torch.tensor(centroid, device=z_agent.device, dtype=z_agent.dtype)
                centroid = centroid.to(device=z_agent.device, dtype=z_agent.dtype).view(-1)
                if centroid.numel() != z_agent.shape[-1]:
                    continue
                centroid = torch.nn.functional.normalize(centroid, dim=-1)
                centroid_ids.append(label_id_int)
                centroid_vectors.append(centroid)

            if len(centroid_vectors) < 2:
                raise RuntimeError("not enough valid centroids for hard negatives")

            centroid_bank = torch.stack(centroid_vectors, dim=0)  # (C, D)
            sims = z_agent @ centroid_bank.t()  # (B, C)

            id_to_col = {lab_id: col for col, lab_id in enumerate(centroid_ids)}
            target_cols = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
            for label_id, col in id_to_col.items():
                target_cols[label_ids == int(label_id)] = int(col)
            valid_rows = target_cols >= 0

            non_target_mask = torch.ones_like(sims, dtype=torch.bool)
            if valid_rows.any():
                non_target_mask[valid_rows, target_cols[valid_rows]] = False
            has_negatives = valid_rows & non_target_mask.any(dim=1)
            if not has_negatives.any():
                raise RuntimeError("no valid hard negatives for current labels")

            masked_sims = sims.masked_fill(~non_target_mask, float("-inf"))
            style_neg_sim[has_negatives] = masked_sims[has_negatives].max(dim=1).values
            style_neg_sim = torch.nan_to_num(style_neg_sim, nan=0.0, posinf=1.0, neginf=-1.0)
            style_neg_penalty = torch.relu(style_neg_sim - r_teacher_motion + style_neg_margin)
            style_neg_penalty = torch.nan_to_num(style_neg_penalty, nan=0.0, posinf=10.0, neginf=0.0)

            # Keep warmup behavior aligned with legacy reward.
            if hasattr(cmd_gen.style_module, "warmup_counter") and hasattr(cmd_gen.style_module, "warmup_frames"):
                warmup_mask = (cmd_gen.style_module.warmup_counter >= cmd_gen.style_module.warmup_frames).float()
                style_neg_penalty = style_neg_penalty * warmup_mask

            reward_raw = reward_raw - float(style_neg_weight) * style_neg_penalty
            reward_raw = torch.nan_to_num(reward_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        except Exception:
            # Fallback to legacy reward when labels/centroids are unavailable.
            effective_mode = "legacy"
    
    # 5. Optionally ramp up style reward to avoid early destabilization
    if ramp_steps is not None and ramp_steps > 0:
        step_count = getattr(env, "common_step_counter", 0)
        scale = min(1.0, float(step_count) / float(ramp_steps))
        reward = reward_raw * scale
    else:
        reward = reward_raw
        scale = 1.0

    # 6. Log metrics to env.extras (for IsaacLab internal logging)
    if hasattr(env, "extras"):
        env.extras["metrics/style_text_sim"] = r_text.mean()
        env.extras["metrics/style_teacher_motion_sim"] = r_teacher_motion.mean()
        env.extras["metrics/style_neg_sim"] = style_neg_sim.mean()
        env.extras["metrics/style_neg_penalty"] = style_neg_penalty.mean()
        env.extras["metrics/style_reward_mode"] = torch.tensor(
            1.0 if effective_mode == "hardneg" else 0.0, device=env.device
        )
        env.extras["metrics/style_reward_raw"] = reward_raw.mean()
        env.extras["metrics/style_reward_scaled"] = reward.mean()
        env.extras["metrics/style_reward_scale"] = torch.tensor(scale, device=env.device)
        
        # Also log warmup ratio for debugging
        if hasattr(cmd_gen.style_module, "warmup_counter"):
            warmup_ratio = (cmd_gen.style_module.warmup_counter >= cmd_gen.style_module.warmup_frames).float().mean()
            env.extras["metrics/style_warmup_ratio"] = warmup_ratio
    
    # Direct wandb logging for detailed debugging
    if hasattr(env, "extras"):
        env.extras["_style_log_counter"] = env.extras.get("_style_log_counter", 0) + 1
        log_counter = env.extras["_style_log_counter"]
    else:
        log_counter = 0
    if WANDB_AVAILABLE and wandb.run is not None and log_counter % _LOG_EVERY_N_STEPS == 0:
        try:
            step_env = getattr(env, "common_step_counter", None)
            log_payload = {
                "debug/style_text_sim": r_text.mean().item(),
                "debug/style_teacher_motion_sim": r_teacher_motion.mean().item(),
                "debug/style_neg_sim": style_neg_sim.mean().item(),
                "debug/style_neg_penalty": style_neg_penalty.mean().item(),
                "debug/style_reward_mode": effective_mode,
                "debug/style_reward_raw": reward_raw.mean().item(),
                "debug/style_reward_scaled": reward.mean().item(),
                "debug/style_reward_min": reward.min().item(),
                "debug/style_reward_max": reward.max().item(),
            }

            # Per-onomatopoeia logging
            if hasattr(cmd_gen, "current_texts"):
                current_texts = cmd_gen.current_texts
                r_text_values = r_text
                r_teacher_values = r_teacher_motion
                reward_raw_values = reward_raw
                reward_scaled_values = reward
                if r_text_values.dim() != 1:
                    r_text_values = r_text_values.view(-1)
                if r_teacher_values.dim() != 1:
                    r_teacher_values = r_teacher_values.view(-1)
                if reward_raw_values.dim() != 1:
                    reward_raw_values = reward_raw_values.view(-1)
                if reward_scaled_values.dim() != 1:
                    reward_scaled_values = reward_scaled_values.view(-1)

                label_logged = False
                if hasattr(cmd_gen, "current_label_ids") and hasattr(cmd_gen, "style_module"):
                    label_list = getattr(cmd_gen.style_module, "label_list", None)
                    label_ids = cmd_gen.current_label_ids
                    if isinstance(label_ids, torch.Tensor):
                        label_ids_tensor = label_ids.to(device=r_text_values.device, dtype=torch.long).view(-1)
                    elif isinstance(label_ids, (list, tuple)) and len(label_ids) == r_text_values.numel():
                        label_ids_tensor = torch.tensor(label_ids, device=r_text_values.device, dtype=torch.long)
                    else:
                        label_ids_tensor = None

                    if label_ids_tensor is not None and label_list:
                        normalized_labels = [str(label).strip() for label in label_list]
                        for label_id, label_name in enumerate(normalized_labels):
                            if not label_name:
                                continue
                            mask = label_ids_tensor == int(label_id)
                            if not mask.any():
                                continue
                            log_payload[f"debug/style_text_sim/{label_name}"] = r_text_values[mask].mean().item()
                            log_payload[f"debug/style_teacher_motion_sim/{label_name}"] = r_teacher_values[mask].mean().item()
                            log_payload[f"debug/style_reward_raw/{label_name}"] = reward_raw_values[mask].mean().item()
                            log_payload[f"debug/style_reward_scaled/{label_name}"] = reward_scaled_values[mask].mean().item()
                            log_payload[f"style/text_sim/{label_name}"] = r_text_values[mask].mean().item()
                            log_payload[f"style/teacher_motion_sim/{label_name}"] = r_teacher_values[mask].mean().item()
                            log_payload[f"style/label_count/{label_name}"] = float(mask.sum().item())
                            label_logged = True

                if not label_logged and isinstance(current_texts, (list, tuple)) and len(current_texts) == r_text_values.numel():
                    cfg_styles = getattr(getattr(cmd_gen, "cfg", None), "styles", None)
                    if cfg_styles:
                        styles = [str(style).strip() for style in list(cfg_styles)]
                    elif hasattr(cmd_gen, "style_module") and getattr(cmd_gen.style_module, "label_list", None):
                        styles = [str(style).strip() for style in list(cmd_gen.style_module.label_list)]
                    else:
                        styles = list(dict.fromkeys([str(text).strip() for text in current_texts]))

                    for style in styles:
                        if not style:
                            continue
                        mask_list = [text == style for text in current_texts]
                        if not any(mask_list):
                            continue
                        mask = torch.tensor(mask_list, device=r_text_values.device, dtype=torch.bool)
                        log_payload[f"debug/style_text_sim/{style}"] = r_text_values[mask].mean().item()
                        log_payload[f"debug/style_teacher_motion_sim/{style}"] = r_teacher_values[mask].mean().item()
                        log_payload[f"debug/style_reward_raw/{style}"] = reward_raw_values[mask].mean().item()
                        log_payload[f"debug/style_reward_scaled/{style}"] = reward_scaled_values[mask].mean().item()
                        log_payload[f"style/text_sim/{style}"] = r_text_values[mask].mean().item()
                        log_payload[f"style/teacher_motion_sim/{style}"] = r_teacher_values[mask].mean().item()
                        log_payload[f"style/label_count/{style}"] = float(mask.sum().item())

            if step_env is not None:
                log_payload["style/step_env"] = float(step_env)
                if hasattr(env, "extras") and not env.extras.get("_wandb_style_metric_defined", False):
                    try:
                        wandb.define_metric("style/*", step_metric="style/step_env")
                    except Exception:
                        pass
                    env.extras["_wandb_style_metric_defined"] = True

            current_step = getattr(wandb.run, "step", None)
            if current_step is not None:
                wandb.log(log_payload, step=int(current_step))
            else:
                wandb.log(log_payload)
        except Exception:
            pass  # Silently fail if wandb logging fails

    return reward
