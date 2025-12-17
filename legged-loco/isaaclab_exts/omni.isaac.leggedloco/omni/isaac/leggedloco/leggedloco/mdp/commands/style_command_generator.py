
from __future__ import annotations

import sys
import torch
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

try:
    from isaaclab.managers import CommandTerm
    from isaaclab.envs import ManagerBasedRLEnv
except ModuleNotFoundError:
    from isaaclab.managers import CommandTerm
    from isaaclab.envs import ManagerBasedRLEnv

from omni.isaac.leggedloco.leggedloco.mdp.style_module import StyleModule, INSTRUCTION_ONOMATOPEIA
import random

class StyleCommandGenerator(CommandTerm):
    """
    Command generator that produces style latents based on onomatopoeia.
    Uses StyleModule to encode text into z_onm and retrieve centroids.
    """

    def __init__(self, cfg : StyleCommandGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        # Note: self.device is inherited from CommandTerm base class

        # Initialize StyleModule
        self.style_module = StyleModule(
            device=str(self.device), run_name=self.cfg.run_name, num_envs=self.num_envs
        )

        # Buffers
        # command: [z_onm (512), centroid (512)] -> 1024 dim
        # We might want to separate them, but CommandTerm usually returns a single tensor.
        # We will return concatenated tensor.
        self._command = torch.zeros(self.num_envs, 1024, device=self.device)
        self.style_latents = torch.zeros(self.num_envs, 512, device=self.device)
        self.centroids = torch.zeros(self.num_envs, 512, device=self.device)

        # Keep track of current text for debugging/logging
        self.current_texts = [""] * self.num_envs

    def __str__(self) -> str:
        return "StyleCommandGenerator"

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, 1024)."""
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Reset the command generator and motion buffer.
        
        This is called on episode reset to clear stale motion history.
        
        Args:
            env_ids: The list of environment IDs to reset. If None, resets all.
            
        Returns:
            Empty dict (for compatibility with CommandTerm interface).
        """
        # Call parent reset (handles resampling_time etc.)
        result = super().reset(env_ids)
        
        # Reset the motion buffer for these environments
        # This prevents stale motion data from affecting style reward
        if env_ids is None:
            self.style_module.reset_buffer(None)
        else:
            # Convert to tensor if needed
            if not isinstance(env_ids, torch.Tensor):
                env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = env_ids
            self.style_module.reset_buffer(env_ids_tensor)
        
        return result

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample command for specified environments."""
        # Sample onomatopoeia for reset environments

        # If user provides a fixed list of styles to sample from
        styles = self.cfg.styles if self.cfg.styles is not None else INSTRUCTION_ONOMATOPEIA

        for env_id in env_ids:
            text = random.choice(styles)
            self.current_texts[env_id] = text

            # Encode
            # Note: encode_instruction returns (1, D) tensors, squeeze to (D,)
            z_onm, centroid = self.style_module.encode_instruction(text)
            z_onm = z_onm.squeeze(0)  # (1, D) -> (D,)
            centroid = centroid.squeeze(0)  # (1, D) -> (D,)

            self.style_latents[env_id] = z_onm
            self.centroids[env_id] = centroid

        # Update command buffer
        self._command[env_ids, :512] = self.style_latents[env_ids]
        self._command[env_ids, 512:] = self.centroids[env_ids]

    def _update_metrics(self):
        pass

    def _update_command(self):
        # Static command until reset
        pass

from dataclasses import dataclass, field
try:
    from isaaclab.utils import configclass
    from isaaclab.managers import CommandTermCfg
except ModuleNotFoundError:
    from isaaclab.utils import configclass
    from isaaclab.managers import CommandTermCfg

@configclass
@dataclass
class StyleCommandGeneratorCfg(CommandTermCfg):
    class_type: type = StyleCommandGenerator
    run_name: str = "sarashina_full_fixed"
    styles: list[str] = None # If None, use all available in INSTRUCTION_ONOMATOPEIA
    resampling_time_range: tuple[float, float] = (1e9, 1e9) # Effectively never resample automatically, only on reset
