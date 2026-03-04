# MotionCLIP Latent Speed Separation Comparison

## Runs
- A (new): `20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full`
- B (old): `20260109_optuna_trial1_full`

## Primary Decision Metric
- primary_metric_name: `sample_rho`
- better_by_primary: `20260109_optuna_trial1_full`

## Key Metrics

| Metric | 20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full | 20260109_optuna_trial1_full | Delta (A-B) |
|---|---:|---:|---:|
| sample_spearman_rho | 0.907222 | 0.913492 | -0.006269 |
| regression_r2 | 0.834894 | 0.879943 | -0.045049 |
| regression_mae | 0.007198 | 0.006337 | 0.000860 |
| probe_macro_f1 | 0.843891 | 0.859193 | -0.015302 |
| probe_balanced_acc | 0.853741 | 0.870856 | -0.017114 |
| perm_p_rho | 0.000999 | 0.000999 | 0.000000 |
| perm_p_macro_f1 | 0.000999 | 0.000999 | 0.000000 |
| label_rank_rho_aux | 0.927273 | 0.927273 | 0.000000 |
| perm_p_label_rank | 0.000999 | 0.000999 | 0.000000 |
| semantic_macro_f1_aux | 0.933845 | 0.880369 | 0.053476 |
| perm_p_semantic_f1 | 0.000999 | 0.000999 | 0.000000 |
| knn_vs_global_ratio | 0.637541 | 0.579185 | 0.058356 |

## Threshold Decision
- A pass: `True`
- B pass: `True`

## Probe Bin Edges (Quantile 3-bin)
- A: `[0.0, 0.029280828312039375]`
- B: `[0.0, 0.029280828312039375]`

## Semantic Class Counts (Aux)
- A: `{'slow': 32, 'mid': 172, 'fast': 80}`
- B: `{'slow': 32, 'mid': 172, 'fast': 80}`

## Output Paths
- A outputs: `/home/jouta/NaVILA-Bench/docs/experiments/assets/20260207_motionclip_speed_latent/run_20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full`
- B outputs: `/home/jouta/NaVILA-Bench/docs/experiments/assets/20260207_motionclip_speed_latent/run_20260109_optuna_trial1_full`
- Comparison CSV: `/home/jouta/NaVILA-Bench/docs/experiments/assets/20260207_motionclip_speed_latent/comparison_metrics.csv`
