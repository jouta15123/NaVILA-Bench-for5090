# Contrastive Learning Main Experiment (English Summary)

## Overview
This experiment evaluates MotionCLIP-style contrastive learning on HOYO motion data with text-style labels (onomatopoeia). We train a joint motion–text embedding using **Supervised Contrastive (SupCon)** loss plus **VAE reconstruction**, and report retrieval metrics in both directions (Text→Motion and Motion→Text), along with reconstruction error (MPJPE) and silhouette score.

- **Two-stage training:** freeze → full
- **Sequence length:** target_len = 100 (2 seconds @ 50 Hz)
- **Centering:** first_frame_com
- **Label granularity:** fine (11 onomatopoeia classes)

## W&B Run Info
- Project: `hoyo_motion_main`
- Group: `cl_len100`
- Run name: `20260118_cl_len100_control_oldbest`
- Run IDs:
  - Full: `370bpe7p`
  - Freeze: `66lbw4tu`
- Run path (for API):
  - `jouta15123-osaka-univercity/hoyo_motion_main/370bpe7p`

## Training Setup (Final / Full Stage)
| Item | Value |
|---|---|
| Loss | SupCon + VAE |
| Sem Encoder | Sarashina |
| Label Mode | fine (11) |
| target_len | 100 |
| centering | first_frame_com |
| batch size | 64 |
| lr | 1.7e-5 |
| weight decay | 4.2e-6 |
| temperature τ | 0.023 |
| λ_vae | 1.0 |
| λ_contrastive | 0.3 |
| steps | freeze: 2000 / full: 14000 |
| optimizer | AdamW |
| seed | 42 |

## Results (Test Split)
| Metric | Freeze | Full |
|---|---:|---:|
| Acc@1 | 0.113 | 0.355 |
| Acc@3 | 0.258 | 0.742 |
| M2T R@1 | 0.113 | 0.355 |
| M2T R@3 | 0.258 | 0.742 |
| M2T R@5 | 0.452 | 0.952 |
| T2M R@1 | 0.182 | 0.545 |
| T2M R@3 | 0.273 | 0.636 |
| T2M R@5 | 0.455 | 1.000 |
| MedR (M2T) | 6 | 2 |
| MedR (T2M) | 6 | 1 |
| MPJPE | 1.381 | 0.152 |
| Silhouette | -0.446 | -0.049 |

## Notes
- The **full** stage substantially improves retrieval metrics and MPJPE compared to the **freeze** stage.
- The **silhouette score remains negative**, indicating class overlap in the embedding space despite improved retrieval performance.
