はい、承知いたしました。ご指定のテキストをObsidianに貼り付けることを想定したMarkdown形式で全文出力します。

---

## 0. ざっくり全体像

> やりたいことを一文でいうと：
> **NaVILA の 2階層構造（VLA + ロコモーション RL）に，MotionCLIP ベースの「スタイル latent」を足して，日本語オノマトペで歩行の質感を指定できる H1 ナビゲーションを作る**

そのための 3 フェーズ構成を提案する：

1.  **Phase 0: オフライン対照学習 (style encoder pretrain)**
    - HOYO 2D 歩容データ $X$ とオノマトペラベル $w$ から MotionCLIP VAE + テキスト射影 `sem_proj` を joint 学習して「**スタイル latent 空間**」を作る。
2.  **Phase 1: NaVILA + H1 RL（style encoder は完全 freeze）**
    - NaVILA の VLA が mid-level 行動（例: “move forward 0.75m”）を出すのはそのまま。[arXiv](https://arxiv.org/html/2412.04453v1)
    - 追加でオノマトペ $w$ を受け取り，$z_{\mathrm{onm}}$ を policy の条件付けに使い， MotionCLIP encoder で推定した $z_{\mathrm{mot(agent)}}$ との cos 類似度で **スタイル報酬** を足す。
    - 学習されるのは PPO policy だけ（style encoder は凍結）。
3.  **Phase 2: RL + オンライン対照学習（任意・発展）**
    - Phase 1 である程度タスクが解けたあと， H1 の roll-out からスタイルがそれっぽい軌跡を抜いてきて HOYO オフラインデータと混ぜて **encoder / sem_proj だけ** を対照学習で微調整。
    - policy の更新は PPO，style encoder は contrastive loss で別オプティマイザ。

以下，論文セクションっぽくちゃんと書いてく。

---

## 1. 問題設定と記号

### 1.1 データとラベル

- HOYO 歩容データセット $\mathcal{D}_{\mathrm{HOYO}} = \{ (X^{(n)}, y^{(n)}) \}_{n=1}^{N}$
    - $X^{(n)} \in \mathbb{R}^{T \times J \times 2}$： 時系列長 $T=60$，関節数 $J=14$，2D 座標（ルート中心・向き正規化済み）。
    - $y^{(n)} \in \{1,\dots,K\}$： オノマトペラベル（fine: 11 クラス / coarse: 4 クラス）。
- オノマトペのテキスト表現
    - ラベル集合を $\mathcal{W} = \{ w_k \}_{k=1}^{K}$ とする。
    - 各 $w_k$ について，日本語文:
        - 通常: 「$w_k$と歩いている。」
        - 「通常」だけ特例文: 「普通に歩いている。」
    - Sarashina or SigLIP テキスト encoder で $e_{\mathrm{text}}(w_k) \in \mathbb{R}^{D_{\mathrm{sem}}}$を取り，L2 正規化して使う。

### 1.2 latent とモジュール

- MotionCLIP encoder（HOYO 2D 対応版）
    $$
    f_{\mathrm{mot}} : \mathbb{R}^{T \times J \times 2} \to \mathbb{R}^{D}, \quad X \mapsto z_{\mathrm{mot}}
    $$
    - 実装的には VAE の平均 $\mu(X)$ を **スタイル latent** とみなす。
- MotionCLIP decoder
    $$
    g_{\mathrm{mot}} : \mathbb{R}^{D} \to \mathbb{R}^{T \times J \times 2}
    $$
    - 再構成損失用。RL では直接は使わない。
- テキスト encoder
    $$
    e_{\mathrm{text}} : \mathcal{W} \to \mathbb{R}^{D_{\mathrm{sem}}}
    $$
- セマンティクス射影
    $$
    W_{\mathrm{sem}} \in \mathbb{R}^{D \times D_{\mathrm{sem}}}, \quad z_s^{(k)} = \mathrm{normalize}\!\bigl(W_{\mathrm{sem}}\, e_{\mathrm{text}}(w_k)\bigr)
    $$
- NaVILA VLA（高レベル VLM）[arXiv](https://arxiv.org/html/2412.04453v1)
    - 入力:
        - ナビゲーション指示文 $u$
        - 過去の RGB フレーム列
        - 現在の RGB フレーム
    - 出力: mid-level 行動テキスト $a_{1:M}$： 例: 「move forward 0.75m」「turn right 30 degrees」など。
- ロコモーション policy（H1 用 RL）[arXiv](https://arxiv.org/html/2412.04453v1)
    $$
    \pi_\theta(a_t \mid s_t, v^{\mathrm{cmd}}_t, z_{\mathrm{onm}})
    $$
    - 観測 $s_t$: H1 の proprio, LiDAR から作る height map など（NaVILA と同様）。[arXiv](https://arxiv.org/html/2412.04453v1)
    - $v^{\mathrm{cmd}}_t$: VLA 出力をパースした目標速度（前進/横移動/回頭）。
    - $z_{\mathrm{onm}}$: オノマトペから計算したスタイル latent（後述）。

---

## 2. Phase 0: オフライン対照学習（style encoder pretrain）

### 2.1 入出力

-   **入力**
    -   HOYO シーケンス $X^{(n)}$
    -   ラベル $y^{(n)}$（fine or coarse）
    -   オノマトペテキスト $w_{y^{(n)}}$
-   **出力（学習後に Navila 側が使うもの）**
    -   MotionCLIP encoder $f_{\mathrm{mot}}$（パラメータ $\phi$）
    -   テキスト射影 $W_{\mathrm{sem}}$（パラメータ $\psi$）
    -   （オプション）各クラスの motion prototype $\mu_k$

### 2.2 アーキテクチャ

- MotionCLIP VAE
    - 入力: $X \in \mathbb{R}^{T \times J \times 2}$ を 実装では $(B, J, 2, T)$ に転置して encoder に投げる（いまのコード通り）。
    - encoder: 時系列 Transformer / temporal conv から latent $\mu(X), \log\sigma(X)$。
    - decoder: latent から 2D 関節列を再構成。
    - latent 次元: $D = 512$（MotionCLIP デフォルト）。
- テキスト側
    - Sarashina / SigLIP encoder で $e_{\mathrm{text}}(w_k)$（固定）を計算。
    - 線形射影 $W_{\mathrm{sem}}$（学習可能）で MotionCLIP 空間へ。
    $$
    z_s^{(k)} = \mathrm{normalize}(W_{\mathrm{sem}} e_{\mathrm{text}}(w_k))
    $$

### 2.3 損失設計

#### (a) VAE 再構成損失
MotionCLIP 側は
- 再構成誤差 $L_{\mathrm{rc}}$
- 速度一致 $L_{\mathrm{vel}}$
- KL 正則化 $L_{\mathrm{kl}}$
などを持っているので，HOYO 版でも
$$
L_{\mathrm{VAE}}(X) = \lambda_{\mathrm{rc}} L_{\mathrm{rc}} + \lambda_{\mathrm{vel}} L_{\mathrm{vel}} + \lambda_{\mathrm{kl}} L_{\mathrm{kl}}
$$
とする（KL は小さめでも OK）。

#### (b) CLIP 風の contrastive（提案）
さっき話した通り，いまの SupCon っぽい実装はちょっと回りくどいので， **モーション → クラスプロトタイプ** のクロスエントロピーにしちゃう：
1.  バッチで HOYO から $(X_i, y_i)$ をサンプル
2.  latent $z_{m,i} = \mathrm{normalize}(f_{\mathrm{mot}}(X_i))$
3.  クラスプロトタイプ（テキスト側）
    $$
    z_s^{(k)} = \mathrm{normalize}\bigl(W_{\mathrm{sem}} e_{\mathrm{text}}(w_k)\bigr) \quad (k = 1,\dots,K)
    $$
4.  ロジット
    $$
    \ell_{i,k} = \exp(\alpha)\, z_{m,i}^\top z_s^{(k)} = \frac{1}{T} z_{m,i}^\top z_s^{(k)}
    $$
    - $\alpha = \log(1/T)$ を学習可能パラメータ（実装の `logit_scale`）。
5.  CLIP 風 CE
    $$
    L_{\mathrm{clip}} = \frac{1}{B}\sum_{i=1}^{B} -\log \frac{\exp(\ell_{i, y_i})} {\sum_{k} \exp(\ell_{i,k})}
    $$
これなら
- 「ラベル k のモーション latent はラベル k のテキストプロトタイプに一番近い」
- 「他ラベルのプロトタイプからは離れる」
がストレートに効く。

#### (c) 全体 loss
Phase 0 では VAE より **分離優先** に振る：
$$
L_{\mathrm{pre}} = \lambda_{\mathrm{VAE}} L_{\mathrm{VAE}} + \lambda_{\mathrm{clip}} L_{\mathrm{clip}}
$$
- 例:
    - 最初: $\lambda_{\mathrm{VAE}} = 0.1, \lambda_{\mathrm{clip}} = 1.0$
    - latent が十分 separable になってきたら $\lambda_{\mathrm{VAE}}$ を 0.5〜1.0 に上げる

### 2.4 coarse → fine の二段階もあり

HOYO のラベルが
- 速い系 / 遅い系 / 重い系 / ふらふら系
みたいな coarse ではっきりしていて，fine はちょい曖昧なら：

1.  まず coarse (4 クラス) だけで Phase 0 を回して 「**倫理観レベルのスタイル軸**」を作る
2.  そのあと fine (11 クラス) を「coarse のサブクラスタ」として 追加で CE を回す

って形もアリ。

---

## 3. Phase 1: NaVILA + H1 RL（style encoder freeze）

ここから NaVILA と繋がるフェーズ。

### 3.1 NaVILA 側（高レベル）

NaVILA 本体はそのまま使う：
1.  ユーザから言語指示文 $u$ を受ける
    例: 「ドアを出て，右に曲がって，廊下の端の机まで歩いて」
2.  VLM (VILA ベース) が
    - 過去の RGB フレーム列（メモリ）
    - 現在フレーム
    - 指示文 $u$
    をプロンプトにして mid-level 行動テキスト列 $\{ a_1, a_2, \dots, a_M\}$ を生成する。[arXiv](https://arxiv.org/html/2412.04453v1)
3.  各 $a_m$ をパースして連続量コマンド $v^{\mathrm{cmd}} = (v_x, v_y, \omega, T_{\mathrm{horizon}})$ に変換。[arXiv](https://arxiv.org/html/2412.04453v1)
4.  そのコマンドを一定ステップ RL policy に渡す。

ここには**オノマトペは登場しない**。
オノマトペ $w$ は **ロコモーション policy 側の “スタイル条件”** として別経路で入れる。

### 3.2 StyleModule: 入出力

Navila RL に差し込むスタイルモジュールを，論文だとこんな感じで書ける：
> **StyleModule** $\mathcal{S}_\eta$ は
> オノマトペ $w$ と H1 の時系列状態から
> スタイル latent およびスタイル報酬を計算するモジュールである。

- `StyleModule.encode_instruction(onm_text: str)`
    - 入力: オノマトペ文字列 $w$
    - 出力:
        - $z_{\mathrm{onm}} = \mathrm{normalize}(W_{\mathrm{sem}} e_{\mathrm{text}}(w))$
        - $z_{\mathrm{sem(onm)}} = z_{\mathrm{onm}}$（reward 計算にも同じものを使う）

- `StyleModule.update_and_compute_reward(h1_state_t) -> (r_style_t, z_mot(agent),t)`
    1.  H1 の現在の 3D joint 位置から HOYO 形式 $X_t^{\mathrm{H1}}\in \mathbb{R}^{T \times J \times 2}$ を作る （リターゲット & 正規化はさっきの仕様どおり）。
    2.  リングバッファに push して，長さ $T=60$ なら MotionCLIP encoder で $z_{\mathrm{mot(agent)}, t} = \mathrm{normalize}(f_{\mathrm{mot}}(X_t^{\mathrm{H1}}))$ を計算。
    3.  スタイル報酬 $r_{\mathrm{style}, t} = \beta \cos\bigl( z_{\mathrm{mot(agent)}, t}, z_{\mathrm{onm}} \bigr)$ （バッファが 60 未満の間は $r_{\mathrm{style}, t}=0$）。

### 3.3 RL policy の入出力

低レベル policy は NaVILA と同じ構造に，スタイル latent を足した形：
- 観測 $o_t = [s_t, h_t, v^{\mathrm{cmd}}_t, z_{\mathrm{onm}}]$
    - $s_t$: H1 の proprio（joint 角・速度など）
    - $h_t$: LiDAR から作る height map（NaVILA と同様）[arXiv](https://arxiv.org/html/2412.04453v1)
    - $v^{\mathrm{cmd}}_t$: VLA から来るコマンド速度
    - $z_{\mathrm{onm}}$: エピソード開始時に固定されたスタイル latent
- policy $a_t \sim \pi_\theta(a_t \mid o_t)$
    - 出力は関節トルク or 目標 joint 位置（NaVILA の実装に合わせる）。
- 報酬 $r_t = r_{\mathrm{task}, t} + r_{\mathrm{style}, t} + r_{\mathrm{reg}, t}$
    - $r_{\mathrm{task}, t}$: ゴールまでの距離，衝突ペナルティなど（NaVILA と同じ設計）。[arXiv](https://arxiv.org/html/2412.04453v1)
    - $r_{\mathrm{reg}, t}$: 転倒ペナルティ，エネルギ正則化など。
- $\beta$ はタスク報酬と同オーダになるよう 0.1〜1.0 で sweep。
- 学習
    - PPO (clip-obj) で $\theta$ のみ更新。
    - $f_{\mathrm{mot}}$ と $W_{\mathrm{sem}}$ は **完全 freeze**。

### 3.4 Phase 1 トレーニングループ（擬似コード）

論文なら Algorithm 1 っぽく書けるやつ：
1.  Phase 0 で学習済みの $f_{\mathrm{mot}}, W_{\mathrm{sem}}$ をロード
2.  NaVILA の VLA をロード（pretrain 済み）
3.  policy パラメータ $\theta$ を初期化
4.  反復:
    1.  環境リセット:
        -   指示文 $u$ をサンプリング
        -   オノマトペ $w$ をサンプリング or 人間が指定
        -   $z_{\mathrm{onm}} = \mathcal{S}_\eta.\mathrm{encode\_instruction}(w)$
    2.  エピソードをロールアウト:
        -   適当なタイミングで VLA から新しい mid-level 行動を取得し，$v^{\mathrm{cmd}}_t$ を更新
        -   各ステップ:
            1.  policy から $a_t$ をサンプルし，環境に適用
            2.  H1 の状態から `StyleModule.update_and_compute_reward` で $r_{\mathrm{style}, t}$ と $z_{\mathrm{mot(agent)}, t}$ を計算
            3.  $r_t$ を計算してバッファに格納
    3.  集めた軌跡で PPO 更新（$\theta$ のみ）。

---

## 4. Phase 2: RL + 対照学習の同時回し（任意）

ここが Jouta が言ってた「$z_p$ を条件付けした RL と対照学習を同時に回す」フェーズやな。
（ここでは $z_p$ ≒ $z_{\mathrm{onm}}$ だと思ってもらって OK）

### 4.1 オンラインデータの収集

Phase 1 の policy がある程度安定したあと：
-   各エピソードで
    -   オノマトペ $w$
    -   H1 の sliding window から得た $X_t^{\mathrm{H1}}$
    -   対応する $z_{\mathrm{mot(agent)}, t}$
    -   スタイル報酬 $r_{\mathrm{style}, t}$
    をログしておく。
-   オンライン対照学習用データセット $\mathcal{D}_{\mathrm{H1}} = \{ (X^{(m)}_{\mathrm{H1}}, w^{(m)}, r^{(m)}_{\mathrm{style}}) \}_m$
    -   信頼できるサンプルだけ使いたければ $r^{(m)}_{\mathrm{style}} > \tau$ とか，成功エピソードのみとかでフィルタ。

### 4.2 損失

Phase 2 では，policy 更新と style encoder 更新を**分離**する：
- policy パラメータ: $\theta$（PPO 用）
- style encoder パラメータ: $\phi$（MotionCLIP encoder の末端）と $\psi$（$W_{\mathrm{sem}}$）

#### (a) policy 側
- 目標はそのまま
    $$
    \max_\theta \mathbb{E}\!\left[ \sum_t \gamma^t r_t \right]
    $$
- セマンティック側のパラメータには勾配を流さない。

#### (b) style encoder 側
- offline + online を混ぜた contrastive:
    $$
    L_{\mathrm{style}} = L_{\mathrm{clip}}^{\mathrm{HOYO}} + \lambda_{\mathrm{on}} L_{\mathrm{clip}}^{\mathrm{H1}}
    $$
- $L_{\mathrm{clip}}^{\mathrm{HOYO}}$: Phase 0 と同じ CE（HOYO データ）
- $L_{\mathrm{clip}}^{\mathrm{H1}}$: H1 から取ってきた $(X_{\mathrm{H1}}^{(m)}, w^{(m)})$ に対して
    $$
    \begin{split}
        z_{m}^{(m)} &= \mathrm{normalize}(f_{\mathrm{mot}}(X_{\mathrm{H1}}^{(m)})), \\
        L_{\mathrm{clip}}^{\mathrm{H1}} &= \frac{1}{M} \sum_m -\log \frac{\exp( \ell_{m, y^{(m)}} )} {\sum_k \exp(\ell_{m,k})}
    \end{split}
    $$
    を計算。
- これで 「H1 の軌跡も HOYO と同じ latent 空間でオノマトペに align されるか」 をオンラインに微調整していく感じ。

### 4.3 同時学習ループ（高レベル）

1.  Phase 1 の $\theta, \phi, \psi$ からスタート
2.  各イテレーションで:
    1.  いつも通り RL roll-out + PPO 更新（$\theta$）
    2.  HOYO バッチ + H1 バッチをサンプリングして $L_{\mathrm{style}}$ で $\phi, \psi$ を 1〜数ステップ更新
3.  $\lambda_{\mathrm{on}}$ は最初は 0 から始めて， policy が崩れないのを確認しつつ 0.1 とかに徐々に増やすイメージ。

---

## 5. ここまでを実装に落とすときのマッピング

今のコードと対応付けると：
-   **Phase 0**
    -   `train_joint(...)` の中の対照損失を， さっき書いた CLIP 風 CE に差し替え
    -   まずは `stage="full"` で `lambda_vae` 小さめ，`lambda_contrastive=1.0` から。
    -   学習完了後:
        -   `motionclip_full_joint_best.pth`
        -   `sem_proj_joint_best.pth`
        を保存 → `StyleModule` 初期化時にロード。
-   **StyleModule（Navila 側）**
    -   `init_style_module(config)` で
        -   MotionCLIP encoder + `sem_proj` ロード
        -   テキスト encoder 初期化
        -   H1→HOYO リターゲット設定ロード
    -   `encode_instruction` / `update_and_compute_reward` は さっきの 3.2 そのまま実装。
-   **RL 側**
    -   NaVILA の IsaacLab 用スクリプト（Go2/H1）の 「低レベル policy の観測構築部分」に `z_onm` を concat
    -   報酬計算部分で `r_style` を足す。
-   **Phase 2（やるなら）**
    -   別スレッドとかじゃなくて， 「PPO 更新 K 回ごとに style encoder 更新 1 回」 みたいな素朴なスケジューラを `train_h1_navila.py`（仮）に追加。

---

こんな感じで，
-   **Phase 0: 「スタイル latent 空間」を作る対照事前学習**
-   **Phase 1: NaVILA + RL にその latent をスタイル報酬として差し込む**
-   **Phase 2: H1 データで style encoder をちょい追学習**