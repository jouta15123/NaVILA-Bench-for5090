## プロジェクト設計の現状整理（project.md vs 実装）

このドキュメントでは，`project.md` で定義した全体構想と，
現在実装が到達している段階の差分を整理する．

---

### 1. 全体構想（project.md の要約）

#### 1.1 研究の大目標

- 日本語オノマトペの曖昧な質感を **身体運動（Motion）** で理解・生成できるモデルを作る．
- 未知オノマトペに対しても「それっぽい歩き方」を出せることがゴール．
- テーマは「移動系オノマトペ × 身体性」であり，
  単なる Text-to-Motion でも純粋な Navigation でもなく，
  **身体運動を通した語意味理解** が主眼．

#### 1.2 三本エンコーダ構想（phon / sem / motion）

- **phon encoder（音韻）**
  - オノマトペの表記（かな列）から音素・リズム・繰り返し構造などの音韻 latent を抽出．
  - 未知語対応のための中核（音が似ている未知オノマトペから意味を類推する）．

- **sem encoder（意味）**
  - 文脈説明文や LLM 生成文から意味 latent を抽出．
  - Sarashina / SigLIP などの日本語埋め込みを活用し，
    文脈に応じた意味の違いを表現できる空間を想定．

- **motion encoder（モーション）**
  - MotionCLIP をベースに，HOYO 2D gait データで fine-tune した latent \(z_{\mathrm{mot}}\) を用いる．
  - 人間歩行の多様な「質感」（速さ，重さ，ふらつきなど）を潜在空間で表現．

#### 1.3 三方向の対照学習

1. **sem ↔ motion（主軸）**
   - 意味 latent とモーション latent の alignment により，
     「さらさら vs ざらざら」など質感の違いを運動で分離．
2. **phon ↔ phon（補助）**
   - 表記ゆれや繰り返し構造を学習し，音が似る語同士を音韻空間で近づける．
3. **phon → sem（未知語ブリッジ）**
   - phon latent から sem latent を推定し，未知オノマトペでも意味 latent を経由して
     モーション latent にアクセスできるようにする．

#### 1.4 RL（Navila + PPO）との統合像

- policy を
  \[
    \pi(a \mid s, z_{\mathrm{onm}})
  \]
  の形にし，オノマトペ latent \(z_{\mathrm{onm}}\) を条件入力とする．
- エージェントのモーション latent \(z_{\mathrm{mot(agent)}}\) と
  オノマトペ意味 latent \(z_{\mathrm{sem(onm)}}\) のコサイン類似度に基づく
  **style reward** を導入：
  \[
    r
    = r_{\mathrm{task}}
      + \beta \cos\left(
          z_{\mathrm{mot(agent)}},\;
          z_{\mathrm{sem(onm)}}
        \right)
  \]
- 最終的には
  \[
    \mathcal{L}
    = \mathcal{L}_{\mathrm{PPO}} + \lambda \mathcal{L}_{\mathrm{contrast}}
  \]
  の形で，PPO と対照学習を joint optimization する構想．
- LangWBC を参考に，
  freeze → encoder-only → full fine-tune の段階的学習で安定化を図る．

---

### 2. 現状の実装ステータス

#### 2.1 実装済みコンポーネント

- **データと前処理**
  - HOYO 歩容データから，日本語オノマトペ 11 語を抽出し，
    それらを 4 種類の粗いスタイル（速い系／遅い系／重い系／ふらふら系）にマージ．
  - 2D 関節列（14 関節，60 フレーム）への整形と，
    train split に基づいた正規化（平均・分散）を実装済み．

- **motion encoder（MotionCLIP）**
  - MotionCLIP ベースの VAE を HOYO 形式に合わせて改変し，
    再構成損失 \(\mathcal{L}_{\mathrm{VAE}}\) を維持しつつ fine-tune 済み．
  - エンコーダ出力を latent \(z_{\mathrm{mot}} \in \mathbb{R}^{512}\) として利用．

- **sem encoder（意味）**
  - Sarashina（`sbintuitions/sarashina-embedding-v2-1b`）と
    SigLIP テキストエンコーダ（`google/siglip-base-patch16-256-multilingual`）の
    2 系統を実装．
  - 各オノマトペ \(w\) に対して
    「\(w\)と歩いている。」「普通に歩いている。」といった日本語文を生成し，
    L2 正規化済みの意味ベクトル \(z_{\mathrm{sem}}^{(\mathrm{fine})}(w)\) を取得．
  - 11 語を coarse 4 スタイルに集約し，
    fine ベクトルの平均として \(z_{\mathrm{sem}}^{(\mathrm{coarse})}(g)\) を定義．

- **sem → motion 射影と対照学習（sem ↔ motion）**
  - バイアス無し線形層 \(\mathrm{sem\_proj}\) により，
    \(z_s^{(k)} = \mathrm{normalize}(\mathrm{sem\_proj}(z_{\mathrm{sem}}^{(k)}))\) を
    MotionCLIP latent 空間上のクラスプロトタイプとして構成．
  - MotionCLIP latent と semantic プロトタイプを並べた
    supervised contrastive（InfoNCE 変種）を実装し，
    \(\mathcal{L}_{\mathrm{VAE}} + \lambda_{\mathrm{contrast}} \mathcal{L}_{\mathrm{contrast}}\)
    を joint 学習．
  - freeze → encoder 段階の 2 段階学習を実装し，
    `train_motionclip_joint.py` でステージ管理と checkpoint の引き継ぎを行うように改良済み．

- **coarse 4 スタイル latent の評価・解析**
  - `evaluate_dataset` による Acc@1, MPJPE 風再構成誤差の評価．
  - coarse 4 スタイルでの混同行列と PCA 可視化スクリプト
    （`eval_joint_best.py`, `visualize_latent_snapshot.py`）を整備．
  - 新規スクリプト `analyze_latent_coarse.py` により，
    クラスタ統計（mean/var），クラス間距離，Fisher-like 比，
    簡易 latent editing（スタイル間平均差ベクトル）を算出し，
    `latent_analysis/README.md` にレポートを作成済み．
  - 解析結果から，coarse 4 スタイルは
    「完全分離ではないが，スタイル報酬の基盤として妥当な程度には分かれている」
    というレベルの構造を持つことを確認．

- **文書化**
  - `tex/graduate_essay.tex` において，
    手法（データ，モデル，損失関数），評価結果，
    Navila との統合設計（style reward，joint optimization，Option A リターゲット）を
    LaTeX で詳細に記述済み．

#### 2.2 まだ実装していない部分

- **phon encoder と phon↔phon / phon→sem**
  - オノマトペ文字列から音韻 latent を抽出する encoder は未実装．
  - phon↔phon の音韻空間での self-supervised 学習や，
    phon→sem MLP による未知語ブリッジは設計段階のまま．

- **CLIP 画像／テキスト latent との alignment**
  - MotionCLIP 論文にあるような，CLIP 空間との alignment を補助損失として導入する案は
    まだ実験されていない．

- **RL（Navila + PPO）との実際の接続**
  - policy への latent 条件付け（\(\pi(a \mid s, z_{\mathrm{onm}})\)）と
    style reward の実装は仕様レベル（tex, plan）までで，
    まだ IsaacLab/Navila 上にはコードとして組み込まれていない．
  - H1 → HOYO 形式へのリターゲット（Option A）は
    仕様としては固まっているが，
    実際の実装（前進運動学＋関節対応表＋2D 投影）は未着手．

- **RL 中の online contrastive（Phase 2）**
  - freeze された MotionCLIP を RL 中に部分的に解凍し，
    online roll-out データを対照学習に混ぜる Phase 2 は，
    まだ評価設計の段階で，実装・実験とも未着手．

---

### 3. 差分と近日着手予定

#### 3.1 「今あるもの」と「目標」とのギャップ

- **実装済み**:
  - sem ↔ motion の対照学習と latent 解析は完了し，
    coarse 4 スタイルレベルで有意なスタイル構造が得られている．
  - MotionCLIP latent を質感表現として使う準備は整っている．

- **未着手 or 部分的**:
  - phon encoder と未知オノマトペ対応はまったく手を付けていない．
  - RL 接続は，tex/plan レベルでは設計されているが，
    まだコードや設定として落ちていない．
  - joint optimization（PPO + contrastive）は，
    いまのところ「オフライン HOYO 学習」と「オンライン RL 学習」が分離しており，
    実際には同時最適化になっていない．

#### 3.2 近日着手予定（本ブランチのゴール）

このブランチでまず目指すのは，
**「Navila + MotionCLIP latent によるスタイル付きナビゲーション」を
sem ↔ motion のみで実現すること** である．

具体的には：

1. **coarse 4 スタイル latent の信頼性を解析で固める**  
   - クラスタ統計・物理量との相関・簡易 latent editing は完了済み．
   - これを前提として，style reward の設計に使う．

2. **RL 接続仕様とスタイル報酬モジュールの雛形を整備する**  
   - `docs/rl_style_integration.md` に，
     - どの latent を policy に渡すか
     - H1 → HOYO リターゲット（Option A）の処理
     - style reward の式と \(\beta\) のレンジ
     - sliding window（60 フレーム）の仕様
     - Phase 1: MotionCLIP 完全 freeze（\(\lambda=0\)），
       Phase 2: 段階的解凍  
     を明文化する．
   - IsaacLab/Navila 側には，
     `init_style_module(config)` / `compute_style_reward(h1_state_history, onm_text)` といった
     関数インターフェースを持つスタブモジュールを追加する．

3. **最初の RL 実験プロトコルを設計する**  
   - \(\beta = 0\)（style なし） vs \(\beta > 0\)（style あり）の A/B 比較実験を
     `docs/rl_style_experiments.md` としてプロトコル化する．
   - ログ出力・評価指標（タスク成功率，エピソード長，衝突数，
     \(\cos(z_{\mathrm{mot(agent)}}, z_{\mathrm{sem(onm)}})\)）を整理する．

4. **phon/未知オノマトペ対応は次フェーズに回す**  
   - 現段階では phon encoder は完全に未実装のため，
     一旦「既知オノマトペ（/ coarse 4 スタイル）」に限定して
     Navila + MotionCLIP の接続を検証する．
   - その上で，LangWBC や LLM オノマトペ分析の知見を踏まえ，
     phon/sem/motion 三本構造をどのタイミングで導入するかを再検討する．

---

### 4. 位置づけのまとめ

- **project.md のゴール**:
  - phon / sem / motion の三本エンコーダを揃え，
    未知オノマトペにも対応する「質感付きナビゲーションシステム」を作ること．

- **現状の到達点**:
  - sem ↔ motion の alignment と coarse 4 スタイル latent の解析までは完了し，
    「既知オノマトペ × HOYO 歩容」の範囲では
    質感 latent をある程度信頼してよい段階にいる．

- **このブランチのゴール**:
  - まずは **sem ↔ motion + RL** だけで，
    Navila H1 に「歩行の質感」を与えられるかどうかを検証する．
  - phon encoder や未知語対応は，その上に乗せる「次フェーズの拡張」として扱う．


