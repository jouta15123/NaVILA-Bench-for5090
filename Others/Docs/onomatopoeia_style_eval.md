# オノマトペ指示に対するスタイル付与RLの評価指標まとめ（報告用）

## 0. 目的
本ドキュメントは、対照学習（MotionCLIP + Sarashina BERT）と強化学習を組み合わせた
オノマトペ指示のスタイル付与について、研究報告として再現可能な評価指標と
評価プロトコルを整理するものである。

想定タスクは H1 ロボットのナビゲーション／歩行であり、スタイル反映度と
タスク性能を分離して評価することを主眼とする。

---

## 1. 評価の基本方針

1. スタイルの意味整合（オノマトペらしさ）
2. モーションの見た目整合（参照モーションへの近さ）
3. タスク達成能力（成功率・到達時間）
4. 運動の安定性・効率（転倒、エネルギー、滑らかさ）

これらは互いにトレードオフがあるため、単一指標ではなく複数指標で報告する。

---

## 2. メジャー指標の適用方法（オノマトペ版）

### 2.1 スタイル意味整合（Teacher-Motion Alignment）

目的: 指示語に対応する教師モーションと実際の動きが一致しているかを測る。

- cos類似（Teacher-Style Sim）
  - cos(z_motion, z_teacher) をエピソード平均で評価。
  - z_teacher は「オノマトペごとの教師モーション」を MotionCLIP のモーションエンコーダに通した埋め込み。
  - スタイルの“近さ”を教師モーション基準で直接測る指標。
  - **注意**: 報酬と同一エンコーダだと「自己採点」になるため、**評価用エンコーダを分離**する。

**式（エピソード平均）**:
```math
\mathrm{StyleSim}
  = \frac{1}{T}\sum_{t=1}^{T}
    \frac{z^{(t)}_{\mathrm{motion}} \cdot z_{\mathrm{teacher}}}
         {\|z^{(t)}_{\mathrm{motion}}\|\,\|z_{\mathrm{teacher}}\|}
```

**z_motion の定義（再現性のため明文化）**:
```math
z^{(t)}_{\mathrm{motion}} = f_{\mathrm{motion}}(\mathbf{X}_{t:t+L-1})
```
```math
\mathrm{StyleSim} = \frac{1}{N}\sum_{k=1}^{N}
\cos\!\left(z^{(k)}_{\mathrm{motion}}, z_{\mathrm{teacher}}\right),
\quad \mathrm{std}(\cdot)\ \text{も併記}
```
実装例: 窓長 L=60, stride=10 でスライディングし、窓ごと埋め込みを平均・分散で集約。

- 補助: Text-Motion Retrieval（R@K / MedR）
  - 対照学習自体の性能評価として実施（RL評価の主指標ではない）。
  - モーションから正しいオノマトペを当てる Top-K 精度。

報告例:
mean cos(z_motion, z_teacher) を主指標にし、R@1 / R@5 / MedR は補助として報告。

**教師埋め込みの集約（複数教師がある場合）**:
```math
z_{\mathrm{teacher}} = \mathrm{L2Norm}\!\left(\frac{1}{M}\sum_{i=1}^{M} z^{(i)}_{\mathrm{teacher}}\right)
```
頑健化する場合: trimmed mean / medoid を採用。max一致は上限評価として併記する。

---

### 2.2 見た目の模倣一致（HOYO 2D 版）

目的: HOYO の教師モーションと見た目が近いかを定量化する。

前提: HOYO は 14関節・2D 座標 (T, 14, 2) を持つため、**関節回転は存在しない**。
そのため DeepMimic の r_p（関節回転）は **2D関節位置（または骨方向）誤差**で代用する。

**式（imitation reward の合成）**:
```math
r_I = w_p r_p + w_v r_v + w_e r_e + w_c r_c
```

（重み例: w_p=0.65, w_v=0.1, w_e=0.15, w_c=0.1）

- r_p 姿勢: 2D関節位置の一致度（推奨: 2D関節位置 or 骨方向ベクトル）
- r_v 角速度: 2D関節速度（フレーム差分）の一致度
- r_e EE位置: 手足の2D位置（右手4/左手7/右足10/左足13）の一致度
- r_c COM: 2D COM（全関節平均）の一致度

**各項の形（例: ガウス型）**:
```math
r_x = \exp\left( -\frac{\|\Delta_x\|^2}{\sigma_x^2} \right),
\quad x \in \{p, v, e, c\}
```
```math
\Delta_p = \mathbf{J}_{2D} - \mathbf{J}^{\ast}_{2D},
\quad
\Delta_v = \dot{\mathbf{J}}_{2D} - \dot{\mathbf{J}}^{\ast}_{2D}
```
```math
\Delta_e = \mathbf{E}_{2D} - \mathbf{E}^{\ast}_{2D},
\quad
\Delta_c = \mathbf{c}_{2D} - \mathbf{c}^{\ast}_{2D}
```

推奨実装（HOYO 正規化後の座標で計算）:
- すべての項目は exp(-||Δ||^2 / σ^2) 型で 0〜1 に正規化。
- r_p は「全関節2D位置」をそのまま使うか、**骨方向（隣接関節の単位ベクトル）**にすると
  平行移動・スケールの影響が抑えられる。
- HOYO の前処理に合わせて [x, y] 順・身長正規化・初期フレーム中心化を揃える。

運用上の注意:
- 参照が複数ある場合は「最大一致」または「平均一致」で評価。
- 位相ずれに弱いため、DTW や最小距離フレーム探索で位相合わせを行う。
- H1 → HOYO へのリターゲティングが未完成の場合、
  この指標は保留し、**cos(z_motion, z_teacher)** を主指標として使う。

**位相・座標系の頑健化（推奨）**:
- DTW 後の最小距離を r_I の各項に反映する（位相ずれ対策）。
- Procrustes alignment（回転・スケール最適化）後の誤差を使う（カメラ条件の影響低減）。
- 骨方向（単位ベクトル）＋骨長比で姿勢の相対構造を評価する。

---

### 2.3 スタイルの識別性能（Classifier 指標）

目的: 生成動作がそのオノマトペに見えるかを第三者的に判別する。

- HoYo のオノマトペラベル（11種）を使い、
  モーション専用分類器で Top-1 精度 / F1 を測定。
- **外部評価器**として使えるため、Style fidelity の主指標として扱うと説得力が高い。

**データ分割の注意**:
- 分類器の訓練に RL ロールアウトを混ぜない。
- 教師モーションは train/test を厳密分離（同一シーケンス由来が混ざらないようにする）。

---

### 2.4 スタイル空間の分離性・一貫性

- クラス分離性: Silhouette / Davies-Bouldin
  - 同じオノマトペがまとまり、別ラベルが離れているか。
- 一貫性（Intra-style variance）
  - 同一オノマトペでの z_motion 分散が小さいほど安定。

---

### 2.5 タスク性能

- 成功率
- 到達時間（平均ステップ数）
- 失敗率／転倒率

**式（報酬の合成例）**:
```math
r_t = r_{\mathrm{task}, t} + \beta\, r_{\mathrm{style}, t} + r_{\mathrm{reg}, t}
```

スタイル報酬あり・なしで比較して、スタイルが上がってもタスクが落ちすぎないかを見る。

---

### 2.6 運動学的補助指標（オノマトペ特徴の説明用）

指標そのものは普遍的だが、オノマトペの説明と結び付けやすい。

- 平均速度 / 歩幅 / 周期 / 接地率
- 体幹の揺れ（pitch/roll）
- エネルギー指標（COTなど）

例: 「のろのろ」は速度低下と接地率増加が観察される、など。

---

### 2.7 教師データにない未知オノマトペへの対応（Open-set）

目的: 教師モーションが存在しないオノマトペに対して、どのように推論・評価するかを定める。

運用方針（推奨の優先順）:

1. **近傍写像（Text -> Known Style）**
   - z_text を既知オノマトペのテキスト埋め込みに最近傍マッチし、最も近い既知スタイルを代用する。
   - 評価は「代用された既知スタイル」に対して cos(z_motion, z_teacher) を計算。

2. **混合スタイル（Convex Mix）**
   - z_text と近い上位K個の既知スタイルを重み付き平均し、疑似 z_teacher を作る。
   - 連続的なニュアンスを表現しやすいが、解釈は補助的に扱う。

3. **ゼロショット評価（Text Alignment のみ）**
   - 教師が存在しない場合は cos(z_motion, z_text) を参考値として記録し、
     既知スタイル群との相対比較で「近さ」を示す。

注意点:
- 未知語は **Open-set** として扱い、通常の精度指標（Top-1/F1）は適用しない。
- 事前に「未知語セット」を作り、既知語への写像の妥当性を小規模に検証しておく。
- 言語的に近い語があっても運動的に近いとは限らないため、報告では必ず注釈を付ける。

評価の出し方（簡易）:
- 未知語ごとに「最近傍の既知スタイル名」を提示。
- cos(z_motion, z_teacher_approx) と cos(z_motion, z_text) を併記。

**信頼度付きの reject option（推奨）**:
```math
\max_i \cos(z_{\mathrm{text}}, z_{\mathrm{known}, i}) < \tau \Rightarrow \text{unknown}
```
閾値未満は未知語として扱い、cos(z_motion, z_text) のみ参考値で報告する。

---

## 3. 推奨レポート構成（先生への報告用）

### 3.1 結果テーブル（最小構成）

| 指標カテゴリ | 指標 | ベースライン（styleなし） | styleあり |
|---|---|---|---|
| 意味整合 | cos(z_motion, z_teacher) |  |  |
| Retrieval（補助） | R@1 / R@5 / MedR |  |  |
| 模倣一致 | r_I（＋内訳） |  |  |
| タスク | 成功率 / 到達時間 |  |  |
| 安定性 | 転倒率 / 衝突回数 |  |  |

### 3.2 図の推奨

- オノマトペごとの cos(z_motion, z_teacher) 分布（箱ひげ図）
- 主要オノマトペの速度・歩幅比較
- 埋め込み空間の UMAP/t-SNE 可視化
- β をスイープした「成功率 vs Style 指標」の Pareto 図（トレードオフの可視化）

---

## 4. 実験設計（最低限）

1. ベースライン: style報酬なし
2. style報酬あり
3. 平面で評価（推奨）
4. オノマトペ一様サンプリング

補足:
階段や障害物は生存行動が支配的になり、スタイルが見えにくいため、
評価は原則平面で実施する。

**統計の最低限（推奨）**:
- 3〜5 seed
- 平均 ± 標準誤差 or 95% CI
- 簡易検定（bootstrap など）

---

## 5. まとめ（要点）

- オノマトペ評価は意味整合、模倣一致、タスク性能を分離して報告する。
- **Classifier 指標を主軸**にし、StyleSim は内部スコアとして補助的に位置づけると説得力が高い。
- cos(z_motion, z_teacher) + imitation reward は内部整合の確認として併記する。
- 可視化はオノマトペ別分布と埋め込み空間の分離が説得力が高い。

---

## 6. 付記: 参考指標の出典

- DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills (ACM TOG 2018)
  - imitation reward（姿勢/速度/EE/COM）とタスク報酬の併用の代表例
  - https://arxiv.org/abs/1804.02717
- AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control (ACM TOG 2021)
  - style reward（モーション事前分布）+ task reward の枠組み
  - https://arxiv.org/abs/2104.02180
- Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions (IROS 2022)
  - style reward を現実ロボットに適用した実証例
  - https://arxiv.org/abs/2203.15103
- Generating Diverse and Natural 3D Human Motions From Text (CVPR 2022)
  - text-to-motion の標準評価（R-Precision / FID / Diversity / Multimodality）
  - https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html
- TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis (ICCV 2023)
  - retrieval 指標（R@K / MedR）の代表例
  - https://arxiv.org/abs/2305.00976

本研究ではオノマトペ指示に合わせて「意味整合・模倣一致・タスク性能」を統合的に報告する。
