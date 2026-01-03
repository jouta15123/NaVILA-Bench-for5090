
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
    beta_text: float = 0.5,
    beta_centroid: float = 0.5,
    ramp_steps: int = 0,
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
    target_centroid = cmd_gen.centroids
    
    # 3. Compute
    reward, r_text, r_centroid = cmd_gen.style_module.compute_current_reward(
        target_z_onm, target_centroid, beta_text, beta_centroid
    )
    reward_raw = reward
    
    # Optionally ramp up style reward to avoid early destabilization
    if ramp_steps is not None and ramp_steps > 0:
        step_count = getattr(env, "common_step_counter", 0)
        scale = min(1.0, float(step_count) / float(ramp_steps))
        reward = reward * scale
    else:
        scale = 1.0

    # Log metrics to env.extras (for IsaacLab internal logging)
    if hasattr(env, "extras"):
        env.extras["metrics/style_text_sim"] = r_text.mean()
        env.extras["metrics/style_centroid_sim"] = r_centroid.mean()
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
            wandb.log({
                "debug/style_text_sim": r_text.mean().item(),
                "debug/style_centroid_sim": r_centroid.mean().item(),
                "debug/style_reward_raw": reward_raw.mean().item(),
                "debug/style_reward_scaled": reward.mean().item(),
                "debug/style_reward_min": reward.min().item(),
                "debug/style_reward_max": reward.max().item(),
            }, commit=False)
        except Exception:
            pass  # Silently fail if wandb logging fails

    return reward
