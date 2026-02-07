# H1 Vision Style Reward Analysis (2026-01-30)

## 目的
- 新旧 run の **寄与率（rate^mag / rate^adv / share^E）** を同条件で比較する
- style が効かない原因が **報酬設計 / encoder 差 / 行動暴れ**のどこにあるか切り分ける
- 学習戦略（重み・ランプ・入力・モデル）設計の土台を作る

---

## 寄与率の意味（指標の解釈）

### rate^mag（報酬の“鳴り”の寄与）
- 1ステップの報酬項 $r_{t,k}$ を使った **絶対値寄与**
- 解釈: **音量が大きい報酬はどれか**（鳴っている項目）

定義（概念）:
$$\mathrm{rate}^{\mathrm{mag}}_k = \frac{\mathbb{E}[|r_{t,k}|]}{\sum_j \mathbb{E}[|r_{t,j}|]}$$

---

### rate^adv（学習に効いている寄与）
- 1-step TD残差を Advantage 近似として使用
- 解釈: **どの報酬が方策更新に効いているか**

TD残差:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

共分散ベースの寄与（絶対値正規化）:
$$\mathrm{rate}^{\mathrm{adv}}_k = \frac{|\mathrm{Cov}(r_{t,k}, \delta_t)|}{\sum_j |\mathrm{Cov}(r_{t,j}, \delta_t)|}$$

> ※終端では $V(s_{t+1})$ を 0 にし、time_out は非終端扱いにする

---

### share^E（関節エネルギーの配分）
- 行動 $a$ の2乗をエネルギー指標とする
- 解釈: **どの関節が“努力”しているか**

$$\mathrm{share}^E_i = \frac{\mathbb{E}[a_i^2]}{\sum_j \mathbb{E}[a_j^2]}$$

**mean_action_sq**
- $\mathbb{E}[\|a\|^2]$ に相当
- share^E の偏りが「全体の暴れ」か「局所的な暴れ」かを見る補助指標

---

## 対象 run（新旧）
- **新 (dup1)**: `logs/rsl_rl/h1_vision_rough/2026-01-28_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_seed42_dup1`
- **旧 (dup3)**: `logs/rsl_rl/h1_vision_rough/2026-01-25_h1_vision_without_speedinput_exp_c_fixed03_exp_c_cmd03_seed42_dup3`

### encoder 差分（env.yaml）
- 差分は **style encoder の run_name のみ**
  - 新: `20260127_contrastive_vae_fine_lambda_vae_sarashina_fine_lv0.12_temp0.05_lc2.0_lv0.12_s43_f2000_full8000_full`
  - 旧: `20260109_optuna_trial1_full`

---

## 評価条件（共通）
- task: `h1_vision_without_speedinput_exp_c_fixed03`
- checkpoint: `model_2000.pt`, `model_2999.pt`
- `--eval_steps 500`
- `--log_reward_terms` 有効（寄与率をJSONへ）
- `--headless`

---

## 最新評価ファイル（今回の実行）
以下の4つを集計対象にした：
- `eval_results/motion/eval_motion_20260130_132957.json`
- `eval_results/motion/eval_motion_20260130_133151.json`
- `eval_results/motion/eval_motion_20260130_133346.json`
- `eval_results/motion/eval_motion_20260130_133541.json`

> どれが dup1/dup3・m2000/m2999 に対応するかは **run名の記録がないため未確定**。
> 現状は **mean_cos_centroid が 0.17帯 = 新（dup1）／0.60帯 = 旧（dup3）** と推定して比較。

---

## 主要結果（11スタイル平均）

| 推定グループ | ファイル | mean_cos_centroid | mean_style_score | mean_action_sq | rate^mag 上位 | share^E 上位 | rate^adv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **新(dup1?)** | `eval_motion_20260130_132957.json` | 0.1769 | -0.0192 | 7.50 | track_lin_vel_xy_exp 65.8% / style_tracking 12.5% / flat_orientation_l2 6.1% | joint_11 21.4% / joint_16 20.0% / joint_15 11.3% | **空** |
| **新(dup1?)** | `eval_motion_20260130_133151.json` | 0.1734 | -0.0126 | 11.59 | track_lin_vel_xy_exp 62.3% / style_tracking 13.1% / flat_orientation_l2 7.3% | joint_16 35.3% / joint_11 17.0% / joint_15 12.7% | **空** |
| **旧(dup3?)** | `eval_motion_20260130_133346.json` | 0.5967 | +0.0539 | 61.31 | track_lin_vel_xy_exp 32.0% / style_tracking 29.5% / joint_deviation_arms 14.4% | joint_5 48.1% / joint_8 8.5% / joint_6 6.6% | **空** |
| **旧(dup3?)** | `eval_motion_20260130_133541.json` | 0.5997 | -0.0032 | 101.72 | style_tracking 31.7% / joint_deviation_arms 27.6% / track_lin_vel_xy_exp 18.1% | joint_6 29.9% / joint_2 12.7% / joint_5 10.9% | **空** |

---

## 観察メモ（新旧の比較）

### 新(dup1推定)
- rate^mag は **track_lin_vel_xy_exp が支配的**（60%超）
- style_tracking は 12〜13% 程度で中位
- mean_action_sq が小さく、**全体の暴れは小さい**

### 旧(dup3推定)
- style_tracking と joint_deviation_arms が高めに出る
- share^E が **特定関節（joint_5 / joint_6）に集中**
- mean_action_sq が非常に大きく、**暴れ/高トルクの疑い**

---

## rate^adv が空の原因（推定）
- `values_pre` / `values_next` が取得できていない（critic未接続・evaluateの返り値不一致）
- `per_step_terms` が取れていない（reward_mgrの中身が空、または step_reward が更新されない）

**対処案（簡易）**
- `values_pre/next` の shape をログ出力して確認
- `reward_mgr._step_reward` の shape と `reward_term_names` を表示して確認

---

## 次のアクション（優先順）
1) **ファイル↔run/ckpt の対応を確定**（実行順ログ or 出力ファイル名に run/ckpt を埋め込む）
2) `rate^adv` が埋まるか確認（values_pre/next をログして原因切り分け）
3) 旧/新の比較表を最終版に更新（dup1/dup3, m2000/m2999 の確定後）

---

## 参考（学習中のW&Bの見方）
- `Train/Contrib/rate_adv/*` を最優先で見る
- `rate_mag` と `rate_adv` のギャップが大きい項目は“鳴ってるだけ”
- `share^E` と `mean_action_sq` が同時に大きい場合は暴れの兆候
