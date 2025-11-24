## スタイル報酬あり/なし RL 実験プロトコル（案）

本ドキュメントは，MotionCLIP ベースのスタイル報酬を
Navila + H1 RL に統合した際の **最初の A/B 実験プロトコル** をまとめたものである．
実験そのものはまだ実行しないが，後からそのまま実行できるように条件を固定する．

---

### 1. 共通設定

- **環境**: IsaacLab の `h1_matterport_vision` タスク  
  - `num_envs`: 16–64 程度（GPU メモリと相談）  
  - 1 エピソード長: 1,000–2,000 ステップ（約 20–40 秒）  
- **観測**:
  - 既存の視覚特徴 + ロボット状態に加えて，
    オノマトペ latent `z_onm`（固定ベクトル）を連結する．
- **ポリシ**: PPO（Navila 既存の設定に準拠）  
  - optimizer, learning rate, batch size などはベースラインと同じ．
- **初期重み**:
  - 可能なら Navila ベースラインの歩行ポリシから微調整，
    難しければランダム初期化からスタート．
- **エピソード設定**:
  - 各エピソードでゴール位置とオノマトペ \(w\)（例: 「すたすた」「のろのろ」）をサンプリング．
  - 1 run につき，オノマトペの分布は一様にする（各スタイルが同程度の頻度で出るように）．

---

### 2. 条件 A/B

#### 条件 A: スタイル報酬なし（ベースライン）

- 報酬:
  \[
    r_t^{(A)}
    = r_{\mathrm{task}, t} + r_{\mathrm{reg}, t}
  \]
- `style_reward_module` はロードしないか，呼び出しても常に 0 を返すスタブ実装とする．
- 目的:
  - タスク成功率・エピソード長・転倒率など，
    純粋なナビゲーション性能のベースラインを測る．

#### 条件 B: スタイル報酬あり

- 報酬:
  \[
    r_t^{(B)}
    = r_{\mathrm{task}, t} + r_{\mathrm{style}, t} + r_{\mathrm{reg}, t},
  \quad
    r_{\mathrm{style}, t}
    = \beta \cos\left(
        z_{\mathrm{mot(agent)}, t},\;
        z_{\mathrm{sem(onm)}}
      \right).
  \]
- \(\beta\) の候補値:
  - \(\beta \in \{0.1, 0.5, 1.0\}\) から 1–2 個を選び実験．
- 目的:
  - スタイル報酬を足してもタスク成功率が大きく劣化しないか．
  - 指示オノマトペに対して latent 類似度が上がるか．

---

### 3. ログ・評価指標

#### 3.1 ログに記録する量

- エピソード単位:
  - 成功 / 失敗フラグ
  - エピソード長（ステップ数）
  - 衝突回数・転倒回数
  - 各エピソードのオノマトペ \(w\)
  - エピソード平均の
    \(\overline{\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})}\)
    （style reward の元になっているコサイン類似度）．
- ステップ単位（必要に応じて間引き）:
  - \(r_{\mathrm{task}, t}\), \(r_{\mathrm{style}, t}\), \(r_{\mathrm{reg}, t}\)
  - buffer が 60 フレームに満たない区間かどうかのフラグ
  - 速度・揺れなどの簡易物理量（オプション）．

#### 3.2 主要な評価指標

- **タスク成功率**:
  - 各条件ごとの成功エピソード数 / 総エピソード数．
- **平均エピソード長**:
  - 成功エピソード・全エピソードそれぞれについて測定．
- **衝突 / 転倒率**:
  - 危険挙動が増えていないかの確認．
- **スタイル類似度**:
  - 条件 A/B で
    \(\overline{\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})}\)
    の分布を比較し，
    スタイル報酬が本当に latent を引っ張れているかを見る．

---

### 4. 実行コマンドの雛形

実際の RL トレーニングスクリプト名は，
`scripts/rsl_rl/` 以下の設計に依存するが，
コマンドラインは概ね以下のような形を想定する：

```bash
# 条件A: beta=0（スタイル報酬なし）
/workspace/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_h1_navila.py \
  --task=h1_matterport_vision \
  --num_envs=32 \
  --config=configs/h1_navila_base.yaml \
  --style-beta=0.0 \
  --style-enabled=false

# 条件B: beta>0（スタイル報酬あり）
/workspace/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_h1_navila.py \
  --task=h1_matterport_vision \
  --num_envs=32 \
  --config=configs/h1_navila_base.yaml \
  --style-beta=0.5 \
  --style-enabled=true \
  --style-config=configs/style_reward_motionclip.yaml
```

`train_h1_navila.py` や YAML ファイル名は仮称であり，
実際のコマンドは実装時に合わせて更新する．
本ドキュメントでは **A/B 設定の切り替えポイント** を固定しておくことを目的とする．

---

### 5. 成功・失敗パターンのチェックリスト

#### 5.1 うまくいっているときのサイン

- 条件 A/B でタスク成功率が同程度か，
  条件 B でわずかに改善している．
- 条件 B の方が
  \(\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})\) の平均値が
  一貫して高い．
- オノマトペごとの挙動に，目視で分かる質感の違いが現れる
  （例: 「のろのろ」で明らかに遅く歩く，など）．

#### 5.2 よくある失敗パターン

- \(\beta\) が大きすぎてタスク報酬を無視し，
  ゴールに向かわずに「それっぽい動き」だけしようとする．
  - 対策: \(\beta\) を小さくするか，style reward をクリップする．
- MotionCLIP リターゲットがうまく機能せず，
  \(\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})\) の分布がほぼランダム．
  - 対策: H1→HOYO 変換を単体テスト（既知の「速い/遅い」動作で latent を確認）．
- buffer が常に 60 フレームに満たず，
  学習中ほとんど style reward がゼロのまま．
  - 対策: エピソード長・制御周期・window サイズの関係を見直す．

---

このプロトコルに従って A/B 実験を行うことで，
「MotionCLIP latent を使った style reward が，
タスク性能を大きく損なわずに質感を制御できるか」を
系統的に評価できるようにする．


