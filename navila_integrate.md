---

## 0. ざっくり全体像

> やりたいことを一文でいうと：
> **NaVILA の 2階層構造（VLA + ロコモーション RL）に，MotionCLIP ベースの「スタイル latent」を足して，日本語オノマトペで歩行の質感を指定できる H1 ナビゲーションを作る**

そのための 3 フェーズ構成を提案する：

1.  **Phase 0: オフライン対照学習 (style encoder pretrain)** ✅ 完了
    - HOYO 2D 歩容データ $X$ とオノマトペラベル $w$ から MotionCLIP VAE + テキスト射影 `sem_proj` を joint 学習して「**スタイル latent 空間**」を作る。
    - **実験結果**: Sarashina エンコーダで 11クラス精度 **70.8%**、4クラス精度 **81.5%** を達成。
2.  **Phase 1: NaVILA + H1 RL（style encoder は完全 freeze）**
    - NaVILA の VLA が mid-level 行動（例: "move forward 0.75m"）を出すのはそのまま。[arXiv](https://arxiv.org/html/2412.04453v1)
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
    - **総サンプル数**: 292（Train: 227, Test: 65）
- オノマトペのテキスト表現
    - ラベル集合を $\mathcal{W} = \{ w_k \}_{k=1}^{K}$ とする。
    - 各 $w_k$ について，日本語文:
        - 通常: 「$w_k$と歩いている。」
        - 「通常」だけ特例文: 「普通に歩いている。」
    - **Sarashina** テキスト encoder (`sbintuitions/sarashina-embedding-v2-1b`) で $e_{\mathrm{text}}(w_k) \in \mathbb{R}^{D_{\mathrm{sem}}}$を取り，L2 正規化して使う。
    - ※ SigLIP との比較実験の結果、日本語オノマトペには **Sarashina を推奨**（11クラスで +13.9pp 優位）

### 1.2 オノマトペ分類

| オノマトペ | サンプル数 | Coarse グループ |
|-----------|-----------|----------------|
| 通常 | 32 | 遅い系 |
| すたすた | 32 | 速い系 |
| せかせか | 22 | 速い系 |
| てくてく | 22 | 速い系 |
| どっしどっし | 32 | 重い系 |
| とぼとぼ | 22 | 遅い系 |
| のしのし | 22 | 重い系 |
| のろのろ | 32 | 遅い系 |
| ぶらぶら | 22 | ふらふら系 |
| よたよた | 22 | ふらふら系 |
| よろよろ | 32 | ふらふら系 |

### 1.3 latent とモジュール

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
    - **推奨**: Sarashina (`sbintuitions/sarashina-embedding-v2-1b`)
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

## 2. Phase 0: オフライン対照学習（style encoder pretrain）✅ 完了

### 2.1 入出力

-   **入力**
    -   HOYO シーケンス $X^{(n)}$
    -   ラベル $y^{(n)}$（fine: 11クラス）
    -   オノマトペテキスト $w_{y^{(n)}}$
-   **出力（学習後に Navila 側が使うもの）**
    -   MotionCLIP encoder $f_{\mathrm{mot}}$（パラメータ $\phi$）
    -   テキスト射影 $W_{\mathrm{sem}}$（パラメータ $\psi$）
    -   （オプション）各クラスの motion prototype $\mu_k$

### 2.2 アーキテクチャ

- MotionCLIP VAE
    - 入力: $X \in \mathbb{R}^{T \times J \times 2}$ を 実装では $(B, J, 2, T)$ に転置して encoder に投げる。
    - encoder: 時系列 Transformer / temporal conv から latent $\mu(X), \log\sigma(X)$。
    - decoder: latent から 2D 関節列を再構成。
    - latent 次元: $D = 512$（MotionCLIP デフォルト）。
- テキスト側
    - **Sarashina** encoder で $e_{\mathrm{text}}(w_k)$（固定）を計算。
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

#### (b) Supervised Contrastive 損失 (SupCon)

モーション潜在ベクトルとテキスト埋め込みを結合した特徴行列で対照学習を行う：

$$
\mathbf{F} = [\mathbf{z}_{\mathrm{motion}}; \mathbf{e}_{\mathrm{text}}] \in \mathbb{R}^{(N+M) \times d}
$$

$$
\mathbf{S} = \exp(\tau) \cdot \mathbf{F} \mathbf{F}^\top
$$

ここで、$\mathbf{z}_{\mathrm{motion}}$ はモーション潜在ベクトル、$\mathbf{e}_{\mathrm{text}}$ はテキスト埋め込み、$\tau$ は学習可能な `logit_scale` パラメータ。

- **正例（Positive）**: 同じオノマトペラベルを持つペア（モーション-モーション、テキスト-テキスト、モーション-テキスト）
- **負例（Negative）**: 異なるオノマトペラベルを持つペア
- **温度パラメータ**: `temp=0.07`（学習可能な logit_scale で制御）

この損失により、テキストエンコーダの埋め込み空間とモーションエンコーダの潜在空間が整列し、オノマトペによるモーション検索・生成が可能になる。

#### (c) 全体 loss

$$
L_{\mathrm{total}} = \lambda_{\mathrm{VAE}} \cdot L_{\mathrm{VAE}} + \lambda_{\mathrm{cont}} \cdot L_{\mathrm{SupCon}}
$$

### 2.4 学習済みハイパーパラメータ（実験で検証済み）

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| `stage` | full | エンコーダ・デコーダ両方を学習 |
| `sem-encoder` | **sarashina** | SigLIP より +13.9pp 優位 |
| `label-mode` | fine | 11クラスで学習 |
| `steps` | 5000 | |
| `batch-size` | 32 | |
| `lr` | 5e-5 | プロジェクタ学習率 |
| `lr-encoder` | 2e-5 | エンコーダ学習率 |
| `lr-decoder` | 2e-5 | デコーダ学習率 |
| `lambda-vae` | 1.0 | VAE損失の重み |
| `lambda-contrastive` | 0.5 | 対照損失の重み |
| `temp` | 0.07 | 温度パラメータ |
| `seed` | 42 | 再現性確保 |

### 2.5 Phase 0 の実験結果

| 評価軸 | Sarashina | SigLIP | 差分 |
|--------|-----------|--------|------|
| **11クラス（細かい識別）** | **70.8%** | 56.9% | **+13.9pp** |
| **4クラス（粗い識別）** | **81.5%** | 78.5% | **+3.0pp** |

**結論**: 日本語オノマトペとモーションの対照学習において、日本語特化モデル **Sarashina** が多言語モデル SigLIP に対して明確な優位性を示した。

### 2.6 学習済みチェックポイント

Phase 0 の成果物（Phase 1 で使用）：
```
hoyo_v1_1/joint_training_results/sarashina_full_fixed/
├── checkpoints/
│   ├── motionclip_step_*.pth
│   └── sem_proj_step_*.pth
├── logs/
└── latent_snapshot_final.npz
```

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
オノマトペ $w$ は **ロコモーション policy 側の "スタイル条件"** として別経路で入れる。

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
    3.  スタイル報酬を計算（バッファが 60 未満の間は $r_{\mathrm{style}, t}=0$）。報酬の計算方法は下記 3.2.1 を参照。

### 3.2.1 スタイル報酬の設計選択肢

スタイル報酬 $r_{\mathrm{style}, t}$ の計算には3つの選択肢がある：

#### (A) テキストベース報酬
オノマトペのテキスト埋め込みを射影した latent と比較：
$$
r_{\mathrm{style}, t}^{\mathrm{(text)}} = \beta \cos\bigl( z_{\mathrm{mot(agent)}, t}, z_{\mathrm{onm}} \bigr)
$$
where $z_{\mathrm{onm}} = \mathrm{normalize}(W_{\mathrm{sem}} e_{\mathrm{text}}(w))$

**利点**: 見たことない表現にも対応可能、汎化性が高い
**欠点**: テキスト射影 $W_{\mathrm{sem}}$ の精度（70.8%）に依存

#### (B) 教師モーション重心ベース報酬
HOYO データセットから各オノマトペクラスの motion latent 重心を事前計算し、それと比較：
$$
r_{\mathrm{style}, t}^{\mathrm{(centroid)}} = \beta \cos\bigl( z_{\mathrm{mot(agent)}, t}, \mu_k \bigr)
$$
where $\mu_k = \mathrm{normalize}\!\left( \frac{1}{|C_k|}\sum_{i \in C_k} f_{\mathrm{mot}}(X^{(i)}) \right)$ は クラス $k$ の motion latent 重心

**利点**: motion 空間内で完結、テキスト射影の誤差を回避、より具体的な目標
**欠点**: HOYO データセットの motion に限定、HOYO(2D人間) ↔ H1(3Dロボット) のドメインギャップ

#### (C) ハイブリッド報酬（推奨）
テキストベースと教師モーション重心の両方を併用：
$$
r_{\mathrm{style}, t}^{\mathrm{(hybrid)}} = \beta_1 \cos\bigl( z_{\mathrm{mot(agent)}, t}, z_{\mathrm{onm}} \bigr) + \beta_2 \cos\bigl( z_{\mathrm{mot(agent)}, t}, \mu_k \bigr)
$$

**推奨理由**:
- テキストベースで大まかな意味的方向を与えつつ
- 教師モーション重心で具体的なスタイルに近づける
- どちらか一方が失敗しても、もう一方で補完できる

| 報酬モード | $\beta_1$ (text) | $\beta_2$ (centroid) | ユースケース |
|-----------|------------------|---------------------|-------------|
| text | 1.0 | 0.0 | ドメインギャップが大きい場合 |
| centroid | 0.0 | 1.0 | テキスト射影の精度に不安がある場合 |
| hybrid | 0.5 | 0.5 | 標準（推奨） |

#### クラス重心の事前計算

Phase 0 完了後、各クラスの motion latent 重心を計算して保存：

```python
# StyleModule 初期化時に事前計算
class StyleModule:
    def __init__(self, ...):
        # latent_snapshot_final.npz から読み込み
        snapshot = np.load("latent_snapshot_final.npz")
        z_all = torch.from_numpy(snapshot["z"])  # (N, D)
        labels = snapshot["labels"]  # (N,)

        # 各クラスの重心を計算
        self.class_centroids = {}
        for label in np.unique(labels):
            mask = labels == label
            z_class = z_all[mask]
            centroid = z_class.mean(dim=0)
            self.class_centroids[label] = F.normalize(centroid, dim=-1)
```

### 3.3 学習済みポリシーを活用したスタイル付与方式

**前提**: NaVILA には既に学習済みの歩行ポリシー $\pi_{\mathrm{base}}$ が存在する。このベースポリシーを活かしつつ、スタイル（質感）だけを追加で付与したい。

以下に3つの設計選択肢を示す。どれを採用するかは実験で決定。

---

#### (A) Residual Policy 方式

学習済みポリシーの出力に、スタイル調整用の小さな residual を足す。

$$
a_t = \pi_{\mathrm{base}}(s_t, v^{\mathrm{cmd}}_t) + \alpha \cdot \pi_{\mathrm{style}}(s_t, z_{\mathrm{onm}})
$$

- $\pi_{\mathrm{base}}$: NaVILA の学習済み歩行ポリシー（**完全 freeze**）
- $\pi_{\mathrm{style}}$: スタイル調整用の小さなネットワーク（**学習対象**）
- $\alpha$: residual のスケール係数（0.1〜0.3 程度から開始）

**アーキテクチャ例**:
```python
class ResidualStylePolicy(nn.Module):
    def __init__(self, base_policy, style_dim=512, hidden_dim=128):
        self.base_policy = base_policy  # freeze
        for p in self.base_policy.parameters():
            p.requires_grad = False

        # 小さな residual network
        self.style_net = nn.Sequential(
            nn.Linear(base_policy.obs_dim + style_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, base_policy.action_dim),
            nn.Tanh()  # 出力を制限
        )
        self.alpha = 0.1  # 学習可能にしてもよい

    def forward(self, s_t, v_cmd, z_onm):
        a_base = self.base_policy(s_t, v_cmd)  # no grad
        obs_style = torch.cat([s_t, z_onm], dim=-1)
        a_residual = self.style_net(obs_style)
        return a_base + self.alpha * a_residual
```

**利点**:
- ベースポリシーの安定した歩行を完全に保持
- スタイル調整の影響範囲を $\alpha$ で制御可能
- 学習が発散しにくい

**欠点**:
- 表現力が制限される（大きなスタイル変化は難しい）
- residual が大きくなりすぎると歩行が崩れる

---

#### (B) Fine-tuning with Style Conditioning 方式

学習済みポリシーを初期値として、スタイル条件付きで fine-tune。

$$
a_t = \pi_{\theta}(s_t, v^{\mathrm{cmd}}_t, z_{\mathrm{onm}})
$$

- $\theta$ は学習済み $\pi_{\mathrm{base}}$ の重みで初期化
- 入力に $z_{\mathrm{onm}}$ を concat して拡張
- 小さい学習率で fine-tune

**アーキテクチャ例**:
```python
class StyleConditionedPolicy(nn.Module):
    def __init__(self, base_policy, style_dim=512):
        # base_policy の構造をコピー
        self.policy = copy.deepcopy(base_policy)

        # 入力層を拡張（z_onm を受け取れるように）
        old_input_dim = self.policy.input_layer.in_features
        new_input_dim = old_input_dim + style_dim

        # 新しい入力層（古い重みを保持しつつ拡張）
        new_input_layer = nn.Linear(new_input_dim, self.policy.input_layer.out_features)
        with torch.no_grad():
            new_input_layer.weight[:, :old_input_dim] = self.policy.input_layer.weight
            new_input_layer.weight[:, old_input_dim:] = 0.01 * torch.randn(...)  # 小さな初期化
            new_input_layer.bias = self.policy.input_layer.bias
        self.policy.input_layer = new_input_layer

    def forward(self, s_t, v_cmd, z_onm):
        obs = torch.cat([s_t, v_cmd, z_onm], dim=-1)
        return self.policy(obs)
```

**学習設定**:
```python
# 小さい学習率で fine-tune
optimizer = Adam([
    {"params": policy.input_layer.parameters(), "lr": 1e-4},  # 新しい部分は少し大きめ
    {"params": other_layers, "lr": 1e-5},  # 既存部分は小さめ
])
```

**利点**:
- 表現力が高い（大きなスタイル変化も可能）
- end-to-end で最適化できる

**欠点**:
- 学習が不安定になる可能性（catastrophic forgetting）
- ベースの歩行能力が崩れるリスク

**対策**:
- KL 正則化: $L_{\mathrm{KL}} = D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}})$ を追加
- 小さい学習率 + 短いステップ数

---

#### (C) Style Adapter 方式（LoRA 風）

ベースポリシーは完全に freeze し、小さな adapter モジュールだけ学習。

$$
a_t = \pi_{\mathrm{base}}(s_t, v^{\mathrm{cmd}}_t) + \mathrm{Adapter}_\phi(s_t, z_{\mathrm{onm}}, h_{\mathrm{base}})
$$

- $h_{\mathrm{base}}$: ベースポリシーの中間特徴量
- Adapter は低ランク行列で構成（LoRA スタイル）

**アーキテクチャ例**:
```python
class StyleAdapter(nn.Module):
    def __init__(self, base_policy, style_dim=512, rank=16):
        self.base_policy = base_policy  # freeze
        for p in self.base_policy.parameters():
            p.requires_grad = False

        # LoRA 風の低ランク adapter
        hidden_dim = base_policy.hidden_dim  # 例: 256
        action_dim = base_policy.action_dim

        # Down projection (hidden + style -> rank)
        self.down = nn.Linear(hidden_dim + style_dim, rank)
        # Up projection (rank -> action)
        self.up = nn.Linear(rank, action_dim)

        # 小さな初期化
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, s_t, v_cmd, z_onm):
        # ベースポリシーの中間特徴を取得
        a_base, h_base = self.base_policy.forward_with_hidden(s_t, v_cmd)

        # Adapter で調整
        adapter_input = torch.cat([h_base, z_onm], dim=-1)
        delta = self.up(F.relu(self.down(adapter_input)))

        return a_base + delta
```

**利点**:
- パラメータ効率が高い（学習パラメータが少ない）
- ベースポリシーを完全に保持
- 複数のスタイルを別々の adapter として学習可能

**欠点**:
- ベースポリシーの中間特徴にアクセスする必要がある
- 実装がやや複雑

---

#### 方式比較サマリー

| 方式 | ベースポリシー | 学習パラメータ | 表現力 | 安定性 | 実装難度 |
|------|---------------|---------------|--------|--------|----------|
| **(A) Residual** | freeze | 小（style_net のみ） | 低〜中 | **高** | **低** |
| **(B) Fine-tune** | 初期値として使用 | 大（全パラメータ） | **高** | 低〜中 | 低 |
| **(C) Adapter** | freeze | **最小**（adapter のみ） | 中 | 高 | 中 |

**推奨**: まずは **(A) Residual** で実験し、表現力が足りなければ **(B) Fine-tune** を試す。

---

### 3.4 共通の報酬設計

どの方式を選んでも、報酬設計は共通：

- 報酬 $r_t = r_{\mathrm{task}, t} + r_{\mathrm{style}, t} + r_{\mathrm{reg}, t}$
    - $r_{\mathrm{task}, t}$: ゴールまでの距離，衝突ペナルティなど（NaVILA と同じ設計）[arXiv](https://arxiv.org/html/2412.04453v1)
    - $r_{\mathrm{style}, t}$: セクション 3.2.1 で定義したスタイル報酬
    - $r_{\mathrm{reg}, t}$: 転倒ペナルティ，エネルギ正則化など
- $\beta$（スタイル報酬の重み）はタスク報酬と同オーダになるよう 0.1〜1.0 で sweep
- $f_{\mathrm{mot}}$ と $W_{\mathrm{sem}}$ は **完全 freeze**

### 3.5 Phase 1 トレーニングループ（擬似コード）

論文なら Algorithm 1 っぽく書けるやつ：

```
Algorithm 1: Style-Conditioned Locomotion Training (Phase 1)

Input:
  - 学習済み style encoder f_mot, W_sem (from Phase 0)
  - 学習済み base policy π_base (from NaVILA)
  - 学習済み VLA (from NaVILA)
  - スタイル付与方式: mode ∈ {residual, finetune, adapter}

Initialize:
  if mode == "residual":
      π_style ← new ResidualStylePolicy(π_base)  # π_base は freeze
      trainable_params ← π_style.style_net.parameters()
  elif mode == "finetune":
      π ← new StyleConditionedPolicy(π_base)  # π_base を初期値に
      trainable_params ← π.parameters()
  elif mode == "adapter":
      adapter ← new StyleAdapter(π_base)  # π_base は freeze
      trainable_params ← adapter.parameters()

  StyleModule.load(f_mot, W_sem, class_centroids)

for episode = 1, 2, ... do:
    1. 環境リセット:
        - 指示文 u をサンプリング
        - オノマトペ w をサンプリング or 人間が指定
        - z_onm ← StyleModule.encode_instruction(w)
        - μ_k ← StyleModule.get_centroid(w)  # 教師motion重心

    2. エピソードをロールアウト:
        for t = 1, 2, ... do:
            - VLA から v_cmd_t を取得（適宜更新）

            - アクション計算:
              if mode == "residual":
                  a_t ← π_base(s_t, v_cmd) + α · π_style(s_t, z_onm)
              elif mode == "finetune":
                  a_t ← π(s_t, v_cmd, z_onm)
              elif mode == "adapter":
                  a_t ← π_base(s_t, v_cmd) + adapter(s_t, z_onm, h_base)

            - 環境に a_t を適用し、s_{t+1} を観測

            - スタイル報酬計算:
              z_agent ← StyleModule.encode_motion(X_t^H1)
              r_style ← β1 · cos(z_agent, z_onm) + β2 · cos(z_agent, μ_k)

            - 総報酬:
              r_t ← r_task + r_style + r_reg

            - バッファに (s_t, a_t, r_t, ...) を格納

    3. PPO 更新:
        - trainable_params のみ更新
        - f_mot, W_sem, π_base (residual/adapter の場合) は freeze
```

---

## 4. Phase 2: RL + 対照学習の同時回し（任意）

ここが「$z_p$ を条件付けした RL と対照学習を同時に回す」フェーズ。
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
    L_{\mathrm{style}} = L_{\mathrm{SupCon}}^{\mathrm{HOYO}} + \lambda_{\mathrm{on}} L_{\mathrm{SupCon}}^{\mathrm{H1}}
    $$
- $L_{\mathrm{SupCon}}^{\mathrm{HOYO}}$: Phase 0 と同じ SupCon（HOYO データ）
- $L_{\mathrm{SupCon}}^{\mathrm{H1}}$: H1 から取ってきた $(X_{\mathrm{H1}}^{(m)}, w^{(m)})$ に対して同様の SupCon を計算。
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
-   **Phase 0** ✅ 完了
    -   `train_motionclip_joint.py` で SupCon 損失を使用
    -   `stage="full"` で `lambda_vae=1.0`，`lambda_contrastive=0.5`
    -   **テキストエンコーダ**: Sarashina（実験で検証済み）
    -   学習完了後のチェックポイント:
        -   `hoyo_v1_1/joint_training_results/sarashina_full_fixed/checkpoints/`
        -   `latent_snapshot_final.npz`
        を保存 → `StyleModule` 初期化時にロード。
-   **StyleModule（Navila 側）**
    -   `init_style_module(config)` で
        -   MotionCLIP encoder + `sem_proj` ロード
        -   **Sarashina** テキスト encoder 初期化
        -   H1→HOYO リターゲット設定ロード
    -   `encode_instruction` / `update_and_compute_reward` は さっきの 3.2 そのまま実装。
-   **RL 側**
    -   NaVILA の IsaacLab 用スクリプト（Go2/H1）の 「低レベル policy の観測構築部分」に `z_onm` を concat
    -   報酬計算部分で `r_style` を足す。
-   **Phase 2（やるなら）**
    -   別スレッドとかじゃなくて， 「PPO 更新 K 回ごとに style encoder 更新 1 回」 みたいな素朴なスケジューラを `train_h1_navila.py`（仮）に追加。

---

## 6. 学習コマンド（参考）

### 6.1 Phase 0: Style Encoder Pretrain

```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/models/train_motionclip_joint.py \
  --stage full \
  --sem-encoder sarashina \
  --label-mode fine \
  --steps 5000 \
  --batch-size 32 \
  --lr 5e-5 \
  --lr-encoder 2e-5 \
  --lr-decoder 2e-5 \
  --lambda-vae 1.0 \
  --lambda-contrastive 0.5 \
  --temp 0.07 \
  --log-interval 100 \
  --eval-interval 200 \
  --seed 42 \
  --run-name sarashina_full_fixed
```

### 6.2 可視化・評価

```bash
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/viz/compare_encoder_results.py \
  --snapshots hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
  --out-dir hoyo_v1_1/viz/outputs/sarashina_eval
```

---

## 7. まとめ

-   **Phase 0: 「スタイル latent 空間」を作る対照事前学習** ✅ 完了
    - Sarashina + SupCon で 11クラス 70.8%、4クラス 81.5% を達成
-   **Phase 1: NaVILA + RL にその latent をスタイル報酬として差し込む** 🔜 次のステップ
-   **Phase 2: H1 データで style encoder をちょい追学習** （任意）

---

## 8. legged-loco RL フレームワーク詳細と統合計画

`legged-loco/` ディレクトリには NaVILA で使用される低レベル歩行ポリシーの RL 実装がある。ここでは Phase 1 の実装に必要な情報を整理する。

### 8.1 フレームワーク概要

| 項目 | 詳細 |
|------|------|
| **RL アルゴリズム** | PPO (Proximal Policy Optimization) |
| **フレームワーク** | rsl_rl (カスタム実装) + Isaac Lab |
| **並列環境数** | 4096 (デフォルト) |
| **ポリシー更新頻度** | 50Hz (sim dt=0.005s × decimation=4) |
| **エピソード長** | 20秒 |

### 8.2 ディレクトリ構成

```
legged-loco/
├── rsl_rl/                          # RSL-RL フレームワーク
│   └── rsl_rl/
│       ├── algorithms/ppo.py        # PPO アルゴリズム実装
│       ├── modules/
│       │   ├── actor_critic.py      # 基本 MLP Actor-Critic
│       │   ├── actor_critic_depth_cnn.py  # 深度画像対応版
│       │   └── actor_critic_recurrent.py  # RNN 版
│       └── runners/on_policy_runner.py    # 学習ループ
│
├── isaaclab_exts/omni.isaac.leggedloco/  # Isaac Lab 環境拡張
│   └── omni/isaac/leggedloco/
│       ├── config/
│       │   ├── go2/go2_low_base_cfg.py  # Go2 設定
│       │   ├── h1/h1_low_base_cfg.py    # H1 設定 ← **ターゲット**
│       │   └── g1/g1_low_base_cfg.py    # G1 設定
│       └── leggedloco/mdp/
│           ├── observations.py          # 観測定義
│           ├── rewards/objnav_rewards.py # 報酬関数
│           └── commands/                 # 速度コマンド生成
│
└── scripts/
    ├── train.py                     # 学習スクリプト
    └── play.py                      # 評価スクリプト
```

### 8.3 Actor-Critic アーキテクチャ

```python
# actor_critic.py より

# Actor (Policy) Network
Actor: [num_obs] → Linear → ELU → [512] → ELU → [256] → ELU → [128] → [num_actions]
  - 出力: アクション平均 μ（ガウシアンポリシー）
  - 標準偏差 σ: 学習可能パラメータ (init_noise_std=1.0)

# Critic (Value) Network
Critic: [num_obs] → Linear → ELU → [512] → ELU → [256] → ELU → [128] → [1]
  - 出力: 状態価値 V(s)
```

### 8.4 観測空間

**Policy 観測（Actor 入力）**:
```python
# PolicyCfg より
1. base_ang_vel          # 基底フレームでの角速度 (3D)
2. base_rpy              # ロール・ピッチ・ヨー (3D)
3. velocity_commands     # 速度コマンド [vx, vy, ω_z] (3D)
4. joint_pos_rel         # 相対関節位置 (12D: Go2, 19D: H1)
5. joint_vel_rel         # 相対関節速度 (12D: Go2, 19D: H1)
6. last_action           # 前回のアクション

# → スタイル latent z_onm (512D) を concat する場所
```

**Critic 観測（追加情報）**:
```python
# CriticObsCfg より（上記に加えて）
1. base_lin_vel          # 線形速度 (3D)
2. projected_gravity     # 重力ベクトル (3D)
3. height_scan           # 地形スキャン (66点)
```

### 8.5 報酬関数設計

**現在の報酬構成** (`objnav_rewards.py`):

| 報酬項 | 重み | 目的 |
|--------|------|------|
| `track_lin_vel_xy_exp` | 1.5 | 速度コマンド追従 |
| `track_ang_vel_z_exp` | 1.5 | 角速度コマンド追従 |
| `flat_orientation_l2` | -2.5 | 姿勢安定化 |
| `base_height` | -5.0 | 体高維持 |
| `feet_air_time` | 0.2 | 歩行リズム |
| `dof_torques_l2` | -0.0002 | エネルギー効率 |
| `action_smoothness_penalty` | -0.02 | 滑らかな動作 |

**スタイル報酬の追加**:
```python
# 追加する報酬項
style_reward = β1 · cos(z_mot(agent), z_onm) + β2 · cos(z_mot(agent), μ_k)
```

### 8.6 PPO ハイパーパラメータ

```python
# Go2RoughPPORunnerCfg より
clip_param = 0.2
gamma = 0.99
lam = 0.95  # GAE lambda
learning_rate = 1.0e-3
num_learning_epochs = 5
num_mini_batches = 4
entropy_coef = 0.01
num_steps_per_env = 24  # ロールアウト長
max_iterations = 5000
```

---

## 9. Phase 1 実装: 具体的な変更箇所

### 9.1 Step 1: StyleModule クラスの実装

**新規ファイル**: `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/style_module.py`

```python
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

class StyleModule:
    """オノマトペからスタイル latent を計算し、報酬を提供するモジュール"""

    def __init__(
        self,
        motionclip_encoder_path: str,
        sem_proj_path: str,
        text_encoder_name: str = "sbintuitions/sarashina-embedding-v2-1b",
        latent_dim: int = 512,
        window_size: int = 60,
        device: str = "cuda",
        reward_mode: str = "hybrid",  # "text", "centroid", "hybrid"
        beta_text: float = 0.5,
        beta_centroid: float = 0.5,
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.reward_mode = reward_mode
        self.beta_text = beta_text
        self.beta_centroid = beta_centroid

        # MotionCLIP encoder ロード (Phase 0 の成果物)
        self.motion_encoder = self._load_motion_encoder(motionclip_encoder_path)
        self.motion_encoder.eval()
        for p in self.motion_encoder.parameters():
            p.requires_grad = False

        # Semantic projection ロード
        self.sem_proj = self._load_sem_proj(sem_proj_path)
        self.sem_proj.eval()
        for p in self.sem_proj.parameters():
            p.requires_grad = False

        # Sarashina text encoder 初期化
        self.text_encoder = self._init_text_encoder(text_encoder_name)

        # クラス重心のロード (latent_snapshot_final.npz から)
        self.class_centroids = {}  # label -> normalized centroid

        # H1 motion buffer (環境ごと)
        self.motion_buffers = {}  # env_id -> deque of joint positions

    def encode_instruction(self, onomatopoeia: str) -> torch.Tensor:
        """オノマトペ文字列からスタイル latent を計算"""
        # テキストを埋め込み
        prompt = f"{onomatopoeia}と歩いている。" if onomatopoeia != "通常" else "普通に歩いている。"
        with torch.no_grad():
            text_emb = self.text_encoder.encode(prompt)  # (D_sem,)
            text_emb = F.normalize(torch.tensor(text_emb, device=self.device), dim=-1)
            z_onm = F.normalize(self.sem_proj(text_emb), dim=-1)
        return z_onm  # (latent_dim,)

    def update_motion_buffer(self, env_ids: torch.Tensor, joint_positions: torch.Tensor):
        """H1 の関節位置をバッファに追加"""
        # joint_positions: (num_envs, J, 2) - HOYO形式にリターゲット済み
        for i, env_id in enumerate(env_ids.tolist()):
            if env_id not in self.motion_buffers:
                self.motion_buffers[env_id] = deque(maxlen=self.window_size)
            self.motion_buffers[env_id].append(joint_positions[i].cpu().numpy())

    def compute_style_reward(
        self,
        env_ids: torch.Tensor,
        z_onm: torch.Tensor,
        label: str = None
    ) -> torch.Tensor:
        """スタイル報酬を計算"""
        num_envs = len(env_ids)
        rewards = torch.zeros(num_envs, device=self.device)

        for i, env_id in enumerate(env_ids.tolist()):
            buffer = self.motion_buffers.get(env_id, None)
            if buffer is None or len(buffer) < self.window_size:
                continue  # バッファが足りない間は報酬 0

            # Motion sequence を構築
            motion_seq = torch.tensor(
                np.stack(list(buffer)),
                device=self.device,
                dtype=torch.float32
            )  # (T, J, 2)

            # Motion latent を計算
            with torch.no_grad():
                z_agent = self.motion_encoder(motion_seq.unsqueeze(0))  # (1, latent_dim)
                z_agent = F.normalize(z_agent, dim=-1).squeeze(0)

            # 報酬計算
            if self.reward_mode == "text":
                rewards[i] = F.cosine_similarity(z_agent, z_onm, dim=0)
            elif self.reward_mode == "centroid":
                if label in self.class_centroids:
                    mu_k = self.class_centroids[label]
                    rewards[i] = F.cosine_similarity(z_agent, mu_k, dim=0)
            else:  # hybrid
                r_text = F.cosine_similarity(z_agent, z_onm, dim=0)
                r_cent = 0.0
                if label in self.class_centroids:
                    mu_k = self.class_centroids[label]
                    r_cent = F.cosine_similarity(z_agent, mu_k, dim=0)
                rewards[i] = self.beta_text * r_text + self.beta_centroid * r_cent

        return rewards

    def reset_buffer(self, env_ids: torch.Tensor):
        """指定環境のバッファをリセット"""
        for env_id in env_ids.tolist():
            if env_id in self.motion_buffers:
                self.motion_buffers[env_id].clear()

    def load_class_centroids(self, snapshot_path: str):
        """latent_snapshot_final.npz からクラス重心を計算"""
        data = np.load(snapshot_path)
        z_all = torch.tensor(data["z"], device=self.device)
        labels = data["labels"]  # string array

        for label in np.unique(labels):
            mask = labels == label
            z_class = z_all[mask]
            centroid = z_class.mean(dim=0)
            self.class_centroids[label] = F.normalize(centroid, dim=-1)
```

### 9.2 Step 2: 観測空間の拡張

**変更ファイル**: `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/observations.py`

```python
# PolicyCfg クラスに追加

@configclass
class StyleConditionedPolicyCfg(PolicyCfg):
    """スタイル条件付きポリシー用の観測設定"""

    # 既存の観測項に加えて
    style_latent = ObsTerm(
        func=lambda env: env.style_module.current_z_onm.expand(env.num_envs, -1),
        # shape: (num_envs, 512)
    )

    def __post_init__(self):
        super().__post_init__()
        # 観測次元の更新
        # 既存 obs + style_latent (512D)
```

### 9.3 Step 3: 報酬関数の拡張

**変更ファイル**: `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/rewards/objnav_rewards.py`

```python
# 新しい報酬関数を追加

def style_alignment_reward(
    env: ManagerBasedRLEnv,
    style_module: StyleModule,
    z_onm: torch.Tensor,
    label: str,
    scale: float = 1.0,
) -> torch.Tensor:
    """スタイル整合性報酬

    H1 の現在の歩行 latent とオノマトペ latent のコサイン類似度を報酬として返す。
    """
    env_ids = torch.arange(env.num_envs, device=env.device)
    rewards = style_module.compute_style_reward(env_ids, z_onm, label)
    return scale * rewards


# CustomRewardsCfg に追加
@configclass
class StyleConditionedRewardsCfg(CustomGo2RewardsCfg):
    """スタイル条件付き報酬設定"""

    style_alignment = RewTerm(
        func=style_alignment_reward,
        params={
            "style_module": None,  # 実行時に設定
            "z_onm": None,         # 実行時に設定
            "label": None,         # 実行時に設定
            "scale": 0.5,          # スタイル報酬の重み β
        },
        weight=1.0,
    )
```

### 9.4 Step 4: Actor-Critic の拡張（Residual 方式）

**新規ファイル**: `legged-loco/rsl_rl/rsl_rl/modules/actor_critic_styled.py`

```python
import torch
import torch.nn as nn
from .actor_critic import ActorCritic


class ResidualStylePolicy(nn.Module):
    """Residual 方式でスタイルを付与するポリシー"""

    def __init__(
        self,
        base_policy: ActorCritic,
        style_dim: int = 512,
        hidden_dim: int = 128,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.alpha = alpha

        # ベースポリシーを freeze
        for p in self.base_policy.parameters():
            p.requires_grad = False

        # Residual network (学習対象)
        obs_dim = base_policy.actor[0].in_features  # 入力次元
        action_dim = base_policy.actor[-1].out_features  # 出力次元

        self.style_net = nn.Sequential(
            nn.Linear(obs_dim + style_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 出力を [-1, 1] に制限
        )

        # 小さな初期化
        for m in self.style_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def act(self, observations: torch.Tensor, style_codes: torch.Tensor):
        """
        Args:
            observations: (batch, obs_dim) 通常の観測
            style_codes: (batch, style_dim) スタイル latent z_onm
        Returns:
            actions: (batch, action_dim)
        """
        # ベースポリシーのアクション (no grad)
        with torch.no_grad():
            a_base = self.base_policy.actor(observations)

        # Residual を計算
        obs_style = torch.cat([observations, style_codes], dim=-1)
        a_residual = self.style_net(obs_style)

        return a_base + self.alpha * a_residual

    def get_value(self, observations: torch.Tensor):
        """Critic は base_policy のものをそのまま使用"""
        return self.base_policy.critic(observations)

    def evaluate(self, observations: torch.Tensor, style_codes: torch.Tensor, actions: torch.Tensor):
        """PPO 更新用の評価"""
        actions_mean = self.act(observations, style_codes)

        # 標準偏差は base_policy から
        action_std = self.base_policy.std

        distribution = torch.distributions.Normal(actions_mean, action_std)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)

        value = self.get_value(observations)

        return value, log_prob, entropy
```

### 9.5 Step 5: 学習ループの拡張

**変更ファイル**: `legged-loco/rsl_rl/rsl_rl/runners/on_policy_runner.py`

```python
# learn() メソッド内で style_module を組み込む

def learn_with_style(self, num_learning_iterations: int, style_module: StyleModule, onomatopoeia: str):
    """スタイル条件付き学習ループ"""

    # オノマトペから z_onm を事前計算
    z_onm = style_module.encode_instruction(onomatopoeia)
    z_onm_batch = z_onm.unsqueeze(0).expand(self.env.num_envs, -1)  # (num_envs, 512)

    for it in range(num_learning_iterations):
        # ロールアウト収集
        with torch.no_grad():
            for step in range(self.num_steps_per_env):
                obs = self.obs

                # スタイル付きアクション
                if hasattr(self.alg.actor_critic, 'act'):
                    actions = self.alg.actor_critic.act(obs, z_onm_batch)
                else:
                    actions = self.alg.actor_critic.actor(obs)

                # 環境ステップ
                self.obs, rewards, dones, infos = self.env.step(actions)

                # Motion buffer 更新
                joint_positions = self._get_hoyo_format_joints()  # H1 → HOYO リターゲット
                style_module.update_motion_buffer(
                    torch.arange(self.env.num_envs),
                    joint_positions
                )

                # スタイル報酬を追加
                style_rewards = style_module.compute_style_reward(
                    torch.arange(self.env.num_envs),
                    z_onm,
                    label=onomatopoeia
                )
                rewards = rewards + style_rewards

                # リセット処理
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    style_module.reset_buffer(reset_ids)

                # ストレージに保存
                self.alg.storage.add_transitions(...)

        # PPO 更新
        self.alg.update()
```

### 9.6 Step 6: H1 → HOYO リターゲット

**新規ファイル**: `legged-loco/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/leggedloco/mdp/retarget.py`

```python
import torch

# H1 関節 → HOYO 14関節 のマッピング
H1_TO_HOYO_MAPPING = {
    # HOYO の 14 関節定義に対する H1 の対応関節
    # (詳細は HOYO データセットの定義に依存)
    0: "left_hip_yaw",
    1: "left_hip_roll",
    2: "left_hip_pitch",
    3: "left_knee",
    4: "left_ankle",
    5: "right_hip_yaw",
    6: "right_hip_roll",
    7: "right_hip_pitch",
    8: "right_knee",
    9: "right_ankle",
    10: "torso",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "head",  # または neck
}


def retarget_h1_to_hoyo(
    h1_joint_positions: torch.Tensor,  # (num_envs, num_h1_joints, 3)
    h1_base_position: torch.Tensor,    # (num_envs, 3)
    h1_base_orientation: torch.Tensor, # (num_envs, 4) quaternion
) -> torch.Tensor:
    """H1 の 3D 関節位置を HOYO の 2D 形式に変換

    Returns:
        hoyo_joints: (num_envs, 14, 2) - ルート中心・向き正規化済み
    """
    num_envs = h1_joint_positions.shape[0]

    # 1. ルート中心化 (腰を原点に)
    root_idx = 0  # H1 のルート関節インデックス
    root_pos = h1_joint_positions[:, root_idx, :2]  # (num_envs, 2)
    centered = h1_joint_positions[:, :, :2] - root_pos.unsqueeze(1)

    # 2. 向き正規化 (進行方向を Y 軸正方向に)
    # base_orientation から yaw を抽出して回転
    yaw = quaternion_to_yaw(h1_base_orientation)  # (num_envs,)
    rotated = rotate_2d(centered, -yaw)

    # 3. HOYO の 14 関節にマッピング
    hoyo_indices = [get_h1_joint_index(name) for name in H1_TO_HOYO_MAPPING.values()]
    hoyo_joints = rotated[:, hoyo_indices, :]  # (num_envs, 14, 2)

    return hoyo_joints
```

---

## 10. 学習コマンド（Phase 1）

### 10.1 ベースライン学習（スタイルなし）

```bash
# legged-loco ディレクトリで実行
cd /home/jouta/NaVILA-Bench/legged-loco

python scripts/train.py \
  --task=h1_base \
  --history_len=9 \
  --run_name=h1_baseline \
  --max_iterations=5000 \
  --save_interval=200 \
  --headless
```

### 10.2 スタイル条件付き学習

```bash
# 新しい学習スクリプト（作成予定）
python scripts/train_styled.py \
  --task=h1_styled \
  --style_encoder_path=../hoyo_v1_1/joint_training_results/sarashina_full_fixed/checkpoints/motionclip_step_5000.pth \
  --sem_proj_path=../hoyo_v1_1/joint_training_results/sarashina_full_fixed/checkpoints/sem_proj_step_5000.pth \
  --centroid_path=../hoyo_v1_1/joint_training_results/sarashina_full_fixed/latent_snapshot_final.npz \
  --onomatopoeia="すたすた" \
  --reward_mode=hybrid \
  --beta_text=0.5 \
  --beta_centroid=0.5 \
  --style_reward_scale=0.5 \
  --policy_mode=residual \
  --residual_alpha=0.1 \
  --history_len=9 \
  --run_name=h1_sutasuta \
  --max_iterations=3000 \
  --headless
```

### 10.3 複数スタイルの同時学習

```bash
# マルチスタイル学習（各環境で異なるスタイル）
python scripts/train_multistyle.py \
  --task=h1_multistyle \
  --styles="通常,すたすた,のろのろ,どっしどっし" \
  --style_sampling=uniform \
  --run_name=h1_multistyle_4class \
  --max_iterations=5000 \
  --headless
```

---

## 11. 実装優先順位

### Phase 1a: 最小実装（1週間目安）
1. ✅ StyleModule の基本実装
2. ✅ H1 → HOYO リターゲットの実装
3. ✅ 観測空間への z_onm 追加
4. ✅ スタイル報酬の追加
5. 動作確認（1スタイルで学習が回ることを確認）

### Phase 1b: Residual Policy（2週間目安）
1. ResidualStylePolicy の実装
2. 学習済みベースポリシーのロード
3. Residual ネットワークのみの学習
4. 各スタイルでの評価

### Phase 1c: マルチスタイル対応（3週間目安）
1. 環境ごとに異なるスタイルをサンプリング
2. スタイル埋め込みテーブルの導入
3. スタイル切り替え時の安定性評価
4. 11 スタイル全てでの定量評価

---

## 12. 評価指標

### 12.1 タスク性能
- **ゴール到達率**: ナビゲーション成功率
- **経路効率**: 最短経路比
- **衝突回数**: 障害物との接触

### 12.2 スタイル再現性
- **スタイル分類精度**: 生成歩行の latent を MotionCLIP encoder で埋め込み、最近傍クラスを判定
- **コサイン類似度**: 生成歩行 latent と目標スタイル latent の類似度
- **主観評価**: 人間評価者によるスタイルの自然さ

### 12.3 歩行品質
- **転倒率**: エピソード中の転倒回数
- **エネルギー効率**: 累積トルク / 移動距離
- **滑らかさ**: アクションの変動量
