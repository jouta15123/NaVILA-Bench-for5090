from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def motion_latent(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Observation term for MotionCLIP latent vector.
    This assumes that the environment has a 'motion_latent' attribute
    that is updated by the VLM/high-level planner.
    """
    if hasattr(env, "motion_latent"):
        return env.motion_latent
    else:
        # Default zero latent (512 dim for MotionCLIP)
        return torch.zeros((env.num_envs, 512), device=env.device)




