## Navila × MotionCLIP 質感RL 接続仕様

このドキュメントは，MotionCLIP ベースのスタイル latent を
Navila + IsaacLab 上の H1 Humanoid RL に統合するための実装仕様をまとめたものである．
当面は **sem ↔ motion のみ** を用いた既知オノマトペに限定し，
phon encoder や未知語対応は次フェーズの拡張とする．

---

### 1. 全体像

#### 1.1 オフライン側（表現学習）

- HOYO 歩容データ（2D, 14 関節, 60 フレーム）を MotionCLIP VAE に入力し，
  モーション latent \(z_{\mathrm{mot}} \in \mathbb{R}^{D}\) を得る．
- 日本語テキスト埋め込み（Sarashina / SigLIP）から意味 latent
  \(z_{\mathrm{sem}}^{(\mathrm{fine})}\), \(z_{\mathrm{sem}}^{(\mathrm{coarse})}\) を生成し，
  線形射影 \(\mathrm{sem\_proj}\) を介して MotionCLIP 空間にマッピングする：
  \[
    z_s^{(k)}
    = \mathrm{normalize}\bigl(\mathrm{sem\_proj}(z_{\mathrm{sem}}^{(k)})\bigr).
  \]
- supervised contrastive + VAE 再構成損失で sem ↔ motion を joint 学習済み．

#### 1.2 オンライン側（RL / Navila）

- Navila の高レベル VLM / プランニングは既存実装を用い，
  目的地などのタスク指示テキストとは別に，
  オノマトペ \(w\) を VLM から（あるいは手動で）与える．
- オノマトペ \(w\) から意味 latent \(z_{\mathrm{onm}}\) を計算し，
  Navila の下位 policy に条件として渡す：
  \[
    z_{\mathrm{onm}}
    = \mathrm{sem\_proj}\bigl(e_{\mathrm{text}}(w)\bigr),
  \quad
    \pi_\theta(a_t \mid s_t, z_{\mathrm{onm}}).
  \]
- 一方で，H1 の実際の関節軌跡から
  MotionCLIP encoder を通じてエージェントのモーション latent
  \(z_{\mathrm{mot(agent)}, t}\) を推定し，
  オノマトペ意味 latent \(z_{\mathrm{sem(onm)}}\) との類似度に基づき
  スタイル報酬 \(r_{\mathrm{style}, t}\) を定義する．

---

### 2. オノマトペ latent の生成

#### 2.1 テキスト埋め込み

- オノマトペ \(w\) に対して，日本語文
  - 通常: 「\(w\)と歩いている。」
  - 「通常」ラベルのみ: 「普通に歩いている。」
  を生成する．
- Sarashina もしくは SigLIP テキストエンコーダを用いて
  文埋め込み \(e_{\mathrm{text}}(w) \in \mathbb{R}^{D_{\mathrm{sem}}}\) を取得し，
  L2 正規化する．

#### 2.2 sem\_proj による MotionCLIP 空間への射影

- joint 学習済みの線形層 \(\mathrm{sem\_proj} \in \mathbb{R}^{D \times D_{\mathrm{sem}}}\) を
  Navila 側にロードする（`motionclip_full_joint.pth` から該当部分のみ抽出）．
- オノマトペ latent \(z_{\mathrm{onm}}\) は
  \[
    z_{\mathrm{onm}}
    = \mathrm{normalize}\bigl(\mathrm{sem\_proj}(e_{\mathrm{text}}(w))\bigr)
  \]
  として計算する．
- RL policy 側では，
  \(z_{\mathrm{onm}}\) を
  - 観測ベクトル \(s_t\) との連結（`obs = concat(raw_obs, z_onm)`），もしくは
  - 中間層への FiLM 風の条件付け
  のいずれかで注入する．
  初期実装では単純な連結を採用する．

---

### 3. H1 → HOYO リターゲット（Option A）

MotionCLIP encoder は HOYO 形式
\((T^\*, J, C) = (60, 14, 2)\) の 2D 関節列を前提とするため，
H1 の 3D 関節軌跡を一時的に HOYO 形式に変換する．
この変換は style reward 計算のみに用い，RL policy 自体の制御には関与しない．

#### 3.1 座標系の正規化

1. H1 の前進運動学から，主要関節の 3D 位置 \(p_j \in \mathbb{R}^3\) を取得する．
2. 骨盤リンクを原点とし，前方方向を \(+x\) 軸，床面を \(x\)–\(z\) 平面とする
   ボディ座標系に変換する．
3. 高さ方向 \(y\) を無視し，
   \((x, z)\) の 2次元座標として 2D 関節位置を得る．

#### 3.2 関節対応

H1 の関節構造と HOYO の 14 関節との対応表を設計する．
初期版では，歩行スタイルに効く下半身を優先し，上半身は単純化する：

- 骨盤（pelvis） → HOYO: root
- 両股関節（left/right hip）
- 両膝（left/right knee）
- 両足首（left/right ankle）
- 胴体中心（torso / chest）
- 両肩 など

関節の数が合わない場合は，
近い位置のリンクをマージするか，
重心に近い代表点を用いて HOYO の 14 スロットにマッピングする．
詳細なマッピングは別ファイル（`docs/h1_to_hoyo_mapping.md` 等）で管理する予定．

#### 3.3 時間窓とリサンプリング

- IsaacLab 側の制御周期は約 50 Hz，
  HOYO は約 30 Hz を想定しているが，
  初期実装では「1 ステップ = 1 フレーム」とみなし，
  直近 60 ステップ分をそのまま 60 フレームとして扱う．
- 必要に応じて線形補間で 60 フレームに揃える：
  - buffer 長 < 60 の場合は zero padding もしくは複写で延長．
  - buffer 長 > 60 の場合は等間隔サンプリングで 60 フレームに間引く．

#### 3.4 正規化

- HOYO 学習時に用いた正規化パラメータ \(\mu, \sigma\) を
  そのまま H1→HOYO 変換結果に適用する：
  \[
    \hat{X} = (X - \mu) / \sigma.
  \]
- こうして得た \(\hat{X} \in \mathbb{R}^{60 \times 14 \times 2}\) を
  MotionCLIP encoder に入力し，
  \(z_{\mathrm{mot(agent)}, t} = f_{\mathrm{mc\_enc}}(\hat{X})\) を計算する．

---

### 4. スタイル報酬の定義

#### 4.1 定義式

- Navila タスク側の報酬を \(r_{\mathrm{task}, t}\) とし，
  MotionCLIP latent に基づくスタイル報酬を
  \[
    r_{\mathrm{style}, t}
    = \beta \cos\left(
        z_{\mathrm{mot(agent)}, t},\;
        z_{\mathrm{sem(onm)}}
      \right)
  \]
  と定義する．
- 最終的な一段時間あたりの報酬は
  \[
    r_t
    = r_{\mathrm{task}, t}
      + r_{\mathrm{style}, t}
      + r_{\mathrm{reg}, t}
  \]
  とし，\(r_{\mathrm{reg}, t}\) には姿勢崩壊・転倒抑制など既存の正則化項を含める．

#### 4.2 \(\beta\) のレンジ

初期の候補レンジ：

- \(\beta = 0\): ベースライン（スタイル報酬なし）
- \(\beta = 0.1\)
- \(\beta = 0.5\)
- \(\beta = 1.0\)

学習の安定性とタスク成功率を見ながら，
タスク報酬と同程度のスケールになるように調整する．

---

### 5. sliding window による毎ステップ更新

スタイル報酬の応答性を確保するため，
`T` ステップごとではなく，
**リングバッファによる 60 フレーム sliding window を毎ステップ更新** する．

#### 5.1 アルゴリズム概要

1. リングバッファ `buffer` を用意し，
   各ステップの HOYO 形式フレーム（14×2）を push していく．
2. `len(buffer) < 60` の間は
   \(r_{\mathrm{style}, t} = 0\) とし，タスク報酬のみで学習する．
3. `len(buffer) == 60` になったら，
   `buffer` の内容を 1 テンソルにまとめて MotionCLIP encoder に入力し，
   \(z_{\mathrm{mot(agent)}, t}\) を計算する．
4. 以後は各ステップで 1 フレーム追加 + 1 フレーム削除を行い，
   毎ステップ新しい 60 フレーム窓から
   \(z_{\mathrm{mot(agent)}, t}\) と \(r_{\mathrm{style}, t}\) を更新する．

---

### 6. MotionCLIP の更新方針（Phase 1 / Phase 2）

#### 6.1 Phase 1: 完全 freeze（\(\lambda = 0\)）

- 最初の RL 実験では，
  MotionCLIP encoder と \(\mathrm{sem\_proj}\) は完全に凍結し，
  **PPO のみ** を更新する．
- loss も
  \[
    \mathcal{L}_{\mathrm{total}}
    = \mathcal{L}_{\mathrm{PPO}}
  \]
  とし，対照損失 \(\mathcal{L}_{\mathrm{contrast}}\) は計算しない
  （\(\lambda = 0\) とおく）．
- 目的：
  - 既に HOYO で学習済みの latent を崩さずに，
    style reward が Navila の挙動にどう効くかを検証する．

#### 6.2 Phase 2: 部分的な解凍と online contrastive（将来案）

- Navila がある程度安定してタスクを解けるようになった後，
  以下のような拡張を検討する：
  - 成功エピソードのみから online roll-out データを抽出．
  - オフライン HOYO バッチと混合して対照学習を再開．
  - まずは \(\mathrm{sem\_proj}\) のみを更新し，
    問題なければ encoder の末端数層のみ解凍する．
- このフェーズでは
  \[
    \mathcal{L}_{\mathrm{total}}
    = \mathcal{L}_{\mathrm{PPO}} + \lambda \mathcal{L}_{\mathrm{contrast}}
  \]
  とし，\(\lambda\) を小さく保ちながら
  「H1 の挙動も MotionCLIP latent でうまく解釈できるか」
  を検証する．

---

### 7. 実装インターフェース案

#### 7.1 スタイルモジュール API

`scripts/style_reward_module.py` に以下のインターフェースを定義する：

- `init_style_module(config) -> StyleModule`  
  - MotionCLIP joint モデルと \(\mathrm{sem\_proj}\) のロード  
  - テキストエンコーダ（Sarashina/SigLIP）の初期化  
  - H1→HOYO リターゲット設定のロード（関節対応表など）
- `StyleModule.encode_instruction(onm_text: str) -> np.ndarray`  
  - オノマトペ文字列から \(z_{\mathrm{onm}}\) と \(z_{\mathrm{sem(onm)}}\) を計算．
- `StyleModule.update_and_compute_reward(h1_state, onm_text=None) -> float`  
  - H1 の現在状態から 1 フレーム分の HOYO 形式を生成し，
    リングバッファに push．
  - buffer が 60 フレーム揃っていれば
    MotionCLIP encoder を呼んで \(r_{\mathrm{style}}\) を返す．
  - onm\_text が None の場合は，エピソード開始時に与えられた
    オノマトペを内部に保持して用いる．

#### 7.2 Navila / RL 側からの呼び出しポイント

- RL 学習スクリプト（例: `scripts/rsl_rl/train_h1_navila.py`（仮））において：
  - 環境リセット時に `style_module.encode_instruction(onm_text)` を呼び，
    \(z_{\mathrm{onm}}\) を policy の観測に含める．
  - 各ステップで
    - `r_task` 計算後に `r_style = style_module.update_and_compute_reward(h1_state)` を呼び，
      合計報酬に加算する．
    - ログには `r_style` の平均値と
      \(\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})\) を記録する．

実際の RL コードのフック位置やファイル名は，
今後 `scripts/rsl_rl/` 以下を精査しながら具体化するが，
本ドキュメントはその際の仕様書として用いる．


