import gymnasium as gym

from .h1_low_base_cfg import H1BaseRoughEnvCfg, H1BaseRoughEnvCfg_PLAY, H1RoughPPORunnerCfg
from .h1_low_vision_cfg import (
    H1VisionRoughEnvCfg,
    H1VisionRoughEnvCfg_Legacy,
    H1VisionRoughEnvCfg_HeadingFixed,
    H1VisionRoughEnvCfg_HeadingFixed_ExpA,
    H1VisionRoughEnvCfg_WithoutSpeedInput,
    H1VisionRoughEnvCfg_WithoutSpeedInput_ExpB,
    H1VisionRoughEnvCfg_WithoutSpeedInput_ExpB_Fixed05,
    H1VisionRoughEnvCfg_WithoutSpeedInput_ExpC_Fixed03,
    H1VisionRoughEnvCfg_PLAY,
    H1VisionRoughPPORunnerCfg,
    H1VisionRoughPPORunnerCfg_FullFT,
)

##
# Register Gym environments.
##


gym.register(
    id="h1_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_base_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_legacy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_Legacy,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_heading_fixed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_HeadingFixed,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_heading_fixed_exp_a",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_HeadingFixed_ExpA,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_without_speedinput",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_WithoutSpeedInput,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_without_speedinput_exp_b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_WithoutSpeedInput_ExpB,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_without_speedinput_exp_b_fixed05",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_WithoutSpeedInput_ExpB_Fixed05,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_without_speedinput_exp_c_fixed03",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_WithoutSpeedInput_ExpC_Fixed03,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="h1_vision_without_speedinput_exp_b_fullft",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_WithoutSpeedInput_ExpB,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg_FullFT,
    },
)


gym.register(
    id="h1_vision_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)
