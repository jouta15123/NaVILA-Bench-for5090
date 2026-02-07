#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticWithBaseInit, ActorCriticRecurrent, ActorCriticDepthCNN, ActorCriticDepthCNNRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs

        if self.cfg.get("use_cnn", False):
            num_actor_obs_prop = self.env.unwrapped.observation_manager.compute_group("proprio").shape[1]
            self.policy_cfg["num_actor_obs_prop"] = num_actor_obs_prop * (self.policy_cfg.get("history_length", 0) + 1)
        
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent | ActorCriticDepthCNN | ActorCriticDepthCNNRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        # if not self.cfg.get("use_cnn", False):
        #     actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
        #         num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        #     ).to(self.device)
        # else:
        #     actor_critic: ActorCriticDepthCNN | Ac = ActorCriticDepthCNN(
        #         num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        #     ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        # Contribution logging config
        self.contrib_cfg = self.cfg.get("contrib", {})
        self.contrib_enabled = bool(self.contrib_cfg.get("enabled", False))
        self.contrib_interval = max(1, int(self.contrib_cfg.get("interval", 1)))
        self.contrib_topk = max(1, int(self.contrib_cfg.get("topk", 8)))

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            contrib_active = self.contrib_enabled and (it % self.contrib_interval == 0)
            reward_term_names = None
            reward_terms_sum = None
            reward_terms_abs_sum = None
            reward_terms_steps = None
            reward_terms_count = 0
            reward_terms_active_mask = None
            reward_step_dt = None
            joint_names = None
            action_sq_sum = None
            action_count = 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    # IMPORTANT!! clip total rewards to avoid early termination
                    rewards = torch.clamp(rewards, min=-5.0)
                    
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if contrib_active:
                        base_env = self.env.unwrapped
                        reward_mgr = getattr(base_env, "reward_manager", None)
                        if reward_mgr is not None and hasattr(reward_mgr, "active_terms") and hasattr(reward_mgr, "_step_reward"):
                            if reward_step_dt is None:
                                reward_step_dt = float(getattr(base_env, "step_dt", 1.0))
                            term_names = list(getattr(reward_mgr, "active_terms", []))
                            step_terms = reward_mgr._step_reward
                            if (
                                reward_term_names is None
                                and isinstance(step_terms, torch.Tensor)
                                and step_terms.ndim == 2
                                and step_terms.shape[1] == len(term_names)
                            ):
                                # add bootstrap term for time_outs to align with PPO rewards
                                reward_term_names = term_names + ["__bootstrap__"]
                                num_terms = len(reward_term_names)
                                reward_terms_sum = torch.zeros(num_terms, device="cpu", dtype=torch.float)
                                reward_terms_abs_sum = torch.zeros(num_terms, device="cpu", dtype=torch.float)
                                reward_terms_steps = torch.zeros(
                                    self.num_steps_per_env, self.env.num_envs, num_terms, device="cpu", dtype=torch.float
                                )
                                # Mask out terms with zero weight (RewardManager does not update _step_reward for them).
                                try:
                                    weights = []
                                    for name in term_names:
                                        try:
                                            weights.append(float(reward_mgr.get_term_cfg(name).weight))
                                        except Exception:
                                            weights.append(1.0)
                                    w = torch.tensor(weights, device=step_terms.device, dtype=torch.float)
                                    reward_terms_active_mask = (w != 0.0).to(dtype=torch.float)
                                except Exception:
                                    reward_terms_active_mask = None
                            if reward_term_names is not None and reward_terms_steps is not None:
                                per_step_terms = step_terms * reward_step_dt
                                if (
                                    reward_terms_active_mask is not None
                                    and reward_terms_active_mask.numel() == per_step_terms.shape[1]
                                ):
                                    per_step_terms = per_step_terms * reward_terms_active_mask
                                reward_terms_steps[i, :, : per_step_terms.shape[1]] = per_step_terms.detach().cpu()
                                reward_terms_sum[: per_step_terms.shape[1]] += per_step_terms.sum(dim=0).detach().cpu()
                                reward_terms_abs_sum[: per_step_terms.shape[1]] += per_step_terms.abs().sum(dim=0).detach().cpu()
                                reward_terms_count += int(per_step_terms.shape[0])
                                if "time_outs" in infos:
                                    time_outs = torch.as_tensor(infos["time_outs"], device=self.device).float()
                                    bootstrap = self.alg.gamma * self.alg.transition.values.squeeze(-1) * time_outs
                                    reward_terms_steps[i, :, -1] = bootstrap.detach().cpu()
                                    reward_terms_sum[-1] += bootstrap.sum().detach().cpu()
                                    reward_terms_abs_sum[-1] += bootstrap.abs().sum().detach().cpu()

                        if actions is not None:
                            if action_sq_sum is None:
                                action_sq_sum = torch.zeros(actions.shape[-1], device="cpu", dtype=torch.float)
                                try:
                                    robot = base_env.scene.get("robot")
                                    if robot is not None and hasattr(robot, "joint_names"):
                                        joint_names = list(robot.joint_names)
                                    else:
                                        joint_names = [f"joint_{j}" for j in range(actions.shape[-1])]
                                except Exception:
                                    joint_names = [f"joint_{j}" for j in range(actions.shape[-1])]
                            action_sq_sum += (actions ** 2).sum(dim=0).detach().cpu()
                            action_count += actions.shape[0]

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                if contrib_active and self.writer is not None:
                    self._log_contrib_metrics(
                        it,
                        reward_term_names,
                        reward_terms_sum,
                        reward_terms_abs_sum,
                        reward_terms_steps,
                        reward_terms_count,
                        action_sq_sum,
                        action_count,
                        joint_names,
                    )

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log_contrib_metrics(
        self,
        iteration: int,
        reward_term_names,
        reward_terms_sum,
        reward_terms_abs_sum,
        reward_terms_steps,
        reward_terms_count: int,
        action_sq_sum,
        action_count: int,
        joint_names,
    ) -> None:
        if self.writer is None:
            return

        def _sanitize(name: str) -> str:
            return name.replace("/", "_")

        # rate^mag
        if reward_term_names is not None and reward_terms_abs_sum is not None and reward_terms_count > 0:
            total_abs = float(reward_terms_abs_sum.sum().item())
            if total_abs > 0.0:
                rate_mag = reward_terms_abs_sum / total_abs
                k = min(self.contrib_topk, rate_mag.numel())
                topk = torch.topk(rate_mag, k)
                for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                    name = _sanitize(reward_term_names[idx])
                    self.writer.add_scalar(f"Train/Contrib/rate_mag/{name}", float(val) * 100.0, iteration)
                self.writer.add_scalar("Train/Contrib/reward_term_samples", reward_terms_count, iteration)

        # rate^adv (covariance with unnormalized advantages)
        if reward_term_names is not None and reward_terms_steps is not None:
            advantages = (self.alg.storage.returns - self.alg.storage.values).squeeze(-1).detach().cpu()
            flat_adv = advantages.reshape(-1)
            flat_terms = reward_terms_steps.reshape(-1, reward_terms_steps.shape[-1])
            if flat_terms.shape[0] == flat_adv.shape[0] and flat_terms.numel() > 0:
                r_mean = flat_terms.mean(dim=0)
                a_mean = flat_adv.mean()
                cov = ((flat_terms - r_mean) * (flat_adv - a_mean).unsqueeze(1)).mean(dim=0)
                cov_abs = cov.abs()
                total_cov = float(cov_abs.sum().item())
                if total_cov > 0.0:
                    rate_adv = cov_abs / total_cov
                    k = min(self.contrib_topk, rate_adv.numel())
                    topk = torch.topk(rate_adv, k)
                    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                        name = _sanitize(reward_term_names[idx])
                        self.writer.add_scalar(f"Train/Contrib/rate_adv/{name}", float(val) * 100.0, iteration)
                    self.writer.add_scalar("Train/Contrib/adv_samples", int(flat_adv.numel()), iteration)

        # share^E
        if action_sq_sum is not None and action_count > 0:
            total_energy = float(action_sq_sum.sum().item())
            if total_energy > 0.0:
                mean_action_sq = total_energy / float(action_count)
                self.writer.add_scalar("Train/Contrib/mean_action_sq", mean_action_sq, iteration)
                share_e = action_sq_sum / total_energy
                k = min(self.contrib_topk, share_e.numel())
                topk = torch.topk(share_e, k)
                for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
                    name = _sanitize(joint_names[idx] if joint_names else f"joint_{idx}")
                    self.writer.add_scalar(f"Train/Contrib/share_e/{name}", float(val) * 100.0, iteration)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
