#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


class ActorCriticWithBaseInit(ActorCritic):
    """ActorCritic initialized from a base policy checkpoint (full fine-tune).

    This loads actor/critic weights from a base checkpoint. If the current
    observation dimension is larger (e.g., style appended), the first layer
    weights are padded with zeros for the extra dimensions.
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        base_policy_checkpoint=None,
        style_dim=0,
        **kwargs,
    ):
        self.base_policy_checkpoint = base_policy_checkpoint
        self.style_dim = style_dim
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        if base_policy_checkpoint is not None and len(base_policy_checkpoint) > 0:
            self._load_base_policy(base_policy_checkpoint)

    def _load_base_policy(self, checkpoint_path: str) -> None:
        print(f"Loading Base Policy (full FT init) from {checkpoint_path}...")
        try:
            loaded_dict = torch.load(checkpoint_path, map_location="cpu")
            state_dict = loaded_dict.get("model_state_dict", loaded_dict)
        except Exception as e:
            print(f"[WARN] Failed to load base policy checkpoint: {e}")
            return

        # Load actor/critic weights only (avoid critic mismatch if shapes differ).
        base_state = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("actor.") or k.startswith("critic.") or k in ("std", "log_std")
        }

        new_state = self.state_dict()
        for k, v in base_state.items():
            if k not in new_state:
                continue
            target = new_state[k]
            if k in ("actor.0.weight", "critic.0.weight") and v.ndim == 2 and target.ndim == 2:
                if v.shape[0] == target.shape[0] and v.shape[1] <= target.shape[1]:
                    padded = target.clone()
                    padded.zero_()
                    padded[:, : v.shape[1]] = v
                    new_state[k] = padded
                else:
                    print(f"[WARN] Skip init for {k}: base {v.shape} vs target {target.shape}")
            elif v.shape == target.shape:
                new_state[k] = v
            else:
                print(f"[WARN] Skip init for {k}: base {v.shape} vs target {target.shape}")

        self.load_state_dict(new_state, strict=False)
        print("Base Policy initialization done.")


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        return None


class ResidualActorCritic(ActorCritic):
    """
    Residual Policy that adds a style-conditioned residual to a frozen base policy.
    base_policy + alpha * style_policy(obs, style)
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 style_dim=512, base_policy_checkpoint=None, residual_scale=0.1,
                 unfreeze_base_last_layer: bool = False,
                 **kwargs):
        
        self.style_dim = style_dim
        self.residual_scale = residual_scale
        self.unfreeze_base_last_layer = unfreeze_base_last_layer
        
        # Initialize the residual policy (self.actor)
        # It takes the full observation (including style)
        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        
        self.base_policy = None
        if base_policy_checkpoint is not None and len(base_policy_checkpoint) > 0:
            print(f"Loading Base Policy from {base_policy_checkpoint}...")
            # We need to instantiate a base policy with correct dimensions
            # Base policy does not see style_dim
            base_actor_obs = num_actor_obs - style_dim
            # Assume critic obs also shrinks? Or critic sees everything?
            # Usually base critic sees base obs.
            # But we only need Actor for inference usually? 
            # If we want to continue training value function, we use `self.critic` (which is new).
            # The base policy's critic is not needed unless we use it for something.
            # We only use base_policy.actor for action.
            
            # Create a dummy container or load structure
            # To load state_dict, we need matching architecture.
            # We assume base policy is an instance of ActorCritic with default args?
            # This is risky without knowing exact config of base policy.
            # For now, we try to load it into a fresh ActorCritic instance.
            # We assume hidden dims are same as kwargs (or default).
            
            self.base_policy = ActorCritic(
                num_actor_obs=base_actor_obs,
                num_critic_obs=num_critic_obs, # Critic might vary, but we don't use base critic
                num_actions=num_actions,
                **kwargs
            )
            
            # Load weights
            try:
                loaded_dict = torch.load(base_policy_checkpoint, map_location=kwargs.get("device", "cpu"))
                # rsl_rl saves "model_state_dict" usually
                state_dict = loaded_dict["model_state_dict"] if "model_state_dict" in loaded_dict else loaded_dict
                # Load actor-only weights to avoid critic shape mismatch
                actor_only = {}
                for k, v in state_dict.items():
                    if k.startswith("actor."):
                        actor_only[k] = v
                    elif k in ("std", "log_std"):
                        actor_only[k] = v
                load_result = self.base_policy.load_state_dict(actor_only, strict=False)
                if load_result.missing_keys or load_result.unexpected_keys:
                    print(
                        "Base Policy Loaded with mismatched keys. "
                        f"Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}"
                    )
                else:
                    print("Base Policy Loaded.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load base policy from {base_policy_checkpoint}: {e}\n"
                    "ResidualActorCritic requires a valid base policy checkpoint. "
                    "Check that the file exists and has the correct format."
                )
            
            # Freeze base policy
            if self.base_policy is not None:
                self.base_policy.eval()
                for name, p in self.base_policy.named_parameters():
                    # Unfreeze last layer (actor.6) and maybe one before it?
                    # structure is [linear, act, linear, act, linear, act, linear] usually?
                    # Let's check self.base_policy.actor structure prints.
                    # Usually: [0: Linear, 1: Act, 2: Linear, 3: Act, 4: Linear] if 2 hidden layers.
                    # We want to unfreeze the LAST Linear layer.
                    
                    # Safer way: unfreeze if it belongs to the last layer module
                    # But named_parameters gives flattened names "actor.0.weight" etc.
                    p.requires_grad = False
                
                # Optionally unfreeze last layer (disabled by default to preserve base behavior)
                if self.unfreeze_base_last_layer:
                    if hasattr(self.base_policy, "actor") and isinstance(self.base_policy.actor, nn.Sequential):
                        # If last is just activation/layer, search back for Linear
                        for i in range(len(self.base_policy.actor)-1, -1, -1):
                            layer = self.base_policy.actor[i]
                            if isinstance(layer, nn.Linear):
                                print(f"Unfreezing Base Policy Layer: actor.{i} ({layer})")
                                for param in layer.parameters():
                                    param.requires_grad = True
                                break
        else:
            print("No base policy checkpoint provided. Training from scratch (Residual=Total).")

    def act(self, observations, **kwargs):
        """
        Sample action from the policy.

        For residual policy: action = base_mean + residual_scale * residual_sample
        The distribution is updated to reflect the combined policy for correct log_prob computation.
        """
        # observations: [base_obs, style_latent]
        self.update_distribution(observations)
        residual_mean = self.distribution.mean
        residual_action = self.distribution.sample()

        if self.base_policy is not None:
            obs_base = observations[:, :-self.style_dim]
            base_mean = self.base_policy.act_inference(obs_base)

            # Adjust distribution mean for correct log_prob computation
            residual_mean = self.distribution.mean
            combined_mean = base_mean + self.residual_scale * residual_mean
            self.distribution = Normal(combined_mean, self.std)
            return self.distribution.sample()

        return residual_action

    def act_inference(self, observations):
        residual_mean = self.actor(observations)
        if self.base_policy is not None:
             obs_base = observations[:, :-self.style_dim]
             base_mean = self.base_policy.act_inference(obs_base)
             return base_mean + self.residual_scale * residual_mean
        return residual_mean
