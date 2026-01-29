# W&B Reward/Posture Analysis
## Overview
- Run path: jouta15123-osaka-univercity/StyleWalker_RL_fixed/07z76rp2
- Run name: StyleWalker_RL_fixed10
- Run state: running
- Created at: 2026-01-25T04:46:59Z
- Data points (reward terms): 1631
- Last window ratio: 0.10
- Step filter: _step >= 1000
- 評価範囲: step>=1000（学習安定期の区間）
- Analysis timestamp: 2026-01-25T14:21:08Z
- Note: Run is still running; values are partial snapshots.

## Experiment Settings (from W&B config)

### Runner / Training

| key | value |
| --- | --- |
| seed | 42 |
| max_iterations | 3000 |
| num_steps_per_env | 32 |
| num_envs | 2048 |
| save_interval | 200 |
| device | cuda:0 |
| logger | wandb |
| experiment_name | h1_vision_rough |
| run_name | 2026-01-25_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_seed42_dup3 |
| wandb_project | StyleWalker_RL_fixed |

### PPO Algorithm

| key | value |
| --- | --- |
| learning_rate | 0.001 |
| gamma | 0.99 |
| lam | 0.95 |
| clip_param | 0.2 |
| entropy_coef | 0.005 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| max_grad_norm | 1 |

### Environment

| key | value |
| --- | --- |
| dt | 0.005 |
| decimation | 4 |
| episode_length_s | 20 |
| base_velocity.lin_vel_x | [0.3, 0.3] |
| base_velocity.lin_vel_y | [0, 0] |
| base_velocity.ang_vel_z | [0, 0] |
| base_velocity.heading | [0, 0] |
| style_command.coord_mode | hoyo_front |

### Policy

| key | value |
| --- | --- |
| style_dim | 512 |
| residual_scale | 0.3 |
| activation | elu |
| actor_hidden_dims | [512, 256, 128] |
| critic_hidden_dims | [512, 256, 128] |
| base_policy_checkpoint | /workspace/NaVILA-Bench/logs/rsl_rl/h1_vision_rough/2024-11-03_15-08-09_height_scan_obst/model_4999_pad877_256.pt |
| unfreeze_base_last_layer | True |

### Reward Weights

| term | weight |
| --- | --- |
| termination_penalty | -200 |
| style_tracking | 2 |
| dof_pos_limits | -1 |
| flat_orientation_l2 | -1 |
| feet_stumble | -0.5 |
| track_lin_vel_xy_exp | 0.5 |
| feet_slide | -0.25 |
| feet_air_time | 0.25 |
| heading_tracking | 0.2 |
| ang_vel_xy_l2 | -0.05 |
| joint_deviation_hip | -0.05 |
| joint_deviation_arms | -0.05 |
| joint_deviation_torso | -0.02 |
| action_rate_l2 | -0.005 |
| dof_acc_l2 | -1.25e-07 |
| dof_torques_l2 | 0 |

### Reward Weight Changes (vs baseline)

- Baseline: h1_vision_heading_fixed

| term | baseline | current | delta |
| --- | --- | --- | --- |
| style_tracking | 5 | 2 | -3 |
| flat_orientation_l2 | -0.2 | -1 | -0.8 |
| heading_tracking | 2 | 0.2 | -1.8 |
| feet_air_time | 0.5 | 0.25 | -0.25 |

## Reward Term Contributions (Top 10 by |mean|)

| rank | term | mean | last_mean | abs_share | std |
| --- | --- | --- | --- | --- | --- |
| 1 | Episode_Reward/style_tracking | 0.939431 | 0.936553 | 49.8% | 0.026087 |
| 2 | Episode_Reward/track_lin_vel_xy_exp | 0.435390 | 0.407476 | 23.1% | 0.016036 |
| 3 | Episode_Reward/joint_deviation_arms | -0.137403 | -0.189252 | 7.3% | 0.048373 |
| 4 | Episode_Reward/flat_orientation_l2 | -0.096667 | -0.064468 | 5.1% | 0.034409 |
| 5 | Episode_Reward/action_rate_l2 | -0.082779 | -0.058927 | 4.4% | 0.011879 |
| 6 | Episode_Reward/joint_deviation_hip | -0.051220 | -0.074671 | 2.7% | 0.012729 |
| 7 | Episode_Reward/feet_slide | -0.037038 | -0.040092 | 2.0% | 0.005437 |
| 8 | Episode_Reward/heading_tracking | 0.034281 | 0.024859 | 1.8% | 0.013314 |
| 9 | Episode_Reward/dof_acc_l2 | -0.024046 | -0.009216 | 1.3% | 0.007935 |
| 10 | Episode_Reward/ang_vel_xy_l2 | -0.017075 | -0.008071 | 0.9% | 0.004512 |

Full list: reward_terms.csv

## Posture-Related Metrics

| metric | mean | last_mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| Episode_Reward/flat_orientation_l2 | -0.096667 | -0.064468 | 0.034409 | -0.204039 | -0.022713 |
| Episode_Reward/heading_tracking | 0.034281 | 0.024859 | 0.013314 | 0.011202 | 0.074501 |
| Metrics/base_velocity/error_vel_yaw | 0.000019 | 0.000010 | 0.000006 | 0.000007 | 0.000034 |

Full list: posture_metrics.csv

## Style Teacher Motion Similarity (per label)

| label | min | max | count |
| --- | --- | --- | --- |
| すたすた | 0.263804 | 0.311432 | 1631 |
| せかせか | 0.288592 | 0.360634 | 1631 |
| てくてく | 0.494763 | 0.550105 | 1631 |
| とぼとぼ | 0.483163 | 0.547718 | 1631 |
| どっしどっし | 0.527338 | 0.605262 | 1631 |
| のしのし | 0.497461 | 0.560905 | 1631 |
| のろのろ | 0.461906 | 0.523092 | 1631 |
| ぶらぶら | 0.454345 | 0.528491 | 1631 |
| よたよた | 0.434069 | 0.508825 | 1631 |
| よろよろ | 0.477962 | 0.554080 | 1631 |
| 通常 | 0.435031 | 0.491600 | 1631 |


### Reward Weight Changes (vs baseline)
- Baseline: h1_vision_heading_fixed

| term | baseline | current | delta |
| --- | --- | --- | --- |
| style_tracking | 5 | 2 | -3 |
| flat_orientation_l2 | -0.2 | -1 | -0.8 |
| heading_tracking | 2 | 0.2 | -1.8 |
| feet_air_time | 0.5 | 0.25 | -0.25 |


## Notes

- Reward terms are taken from keys prefixed with 'Episode_Reward/'.
- Posture-related metrics are detected by keyword match (orientation/yaw/roll/pitch/heading).
- If roll/pitch angles are needed, add explicit logging in training.
- style/teacher_motion_sim/* is reported per label when available.
