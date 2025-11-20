# 研究方針決定

# 🔥 **研究全体の方針（最新版）**

## **0. まず大前提**

本質的にやりたいのは：

> 「オノマトペの曖昧性を、身体性（モーション）側で理解させる」
> 
> 
> ＝「未知オノマトペにも“らしい動き”を生成できる知的システム」
> 

ナビゲーションに閉じすぎるより、

**“移動系オノマトペに対応する身体表現を学ぶ”** 方向が主軸

「LLMはなんとなくオノマトペを理解できているが、実際に動きとしてはオノマトペを理解できていないのではないか」

---

# 1️⃣ **技術的コア：オノマトペ × 音韻 × 意味 × モーションの対照学習**

## **1-1. 三本のエンコーダを用意する**

- **phon encoder（音韻）**
    
    オノマトペ単体。文字・音の似さを捉える役。
    
- **sem encoder（意味）**
    
    周辺文脈や説明文から意味的特徴を抽出。
    
- **motion encoder（モーション）**
    
    MotionCLIP の motion tower を LoRA or 追加学習で拡張。
    

→ **音韻と意味を分離した二因子モデル**が今のところ最適。

## **1-2. 対照学習は三方向**

1. **意味 ↔ モーション（主役・最重要）**
    
    → “さらさら” vs “ざらざら” みたいな意味差はここでガッツリ分離。
    
2. **音韻 ↔ 音韻（補助）**
    
    → オノマトペ単体の構造を学ぶ（表記揺れ、促音、伸ばし棒など）。
    
3. **音韻 → 意味（弱いブリッジ）**
    
    → 未知オノマトペの意味を推定するためのマッピング。
    
    （“スルスル”が smooth 系、“ゴツゴツ”が rough 系になるように）
    

→ こうすれば

**“音は似てるけど意味が真逆”**問題を完全に回避できる。

---

# 2️⃣ **MotionCLIP の活用方針**

### MotionCLIP の motion encoder を：

- **日本語版 CLIP のテキスト塔（あるなら）と alignment**
- **LoRA で軽く finetune**
- **対照学習の target motion space** として使う

さらに：

- **motion encoder の潜在 $z_p$ を low-level policy の入力に入れる**
    
    → “速度/加速度だけ渡す” より質感が表現できる。
    
    → 身体性宿るのはこの latent やから。
    

これは **速度・加速度入力に違和感ある**問題の解決策として最適。

---

# 3️⃣ **RL（PPO）との結合**

### **方針**

- low-level policy は
    
    $$
    π(a∣s,zonm)\pi(a|s, z_{\text{onm}})
    $$
    
    みたいに、**オノマトペ embedding を条件として動く**。
    
- reward は
    
    $$
    r=rtask+β⋅cos⁡(zmot(agent),zsem)r = r_\text{task} + \beta \cdot \cos(z_\text{mot(agent)}, z_\text{sem})
    $$
    
    みたいに style consistency を追加。
    

### **強化学習＋対照学習を joint に流す**

PPO の loss に

$$
LPPO+λLcontrast\mathcal{L}_\text{PPO} + \lambda \mathcal{L}_\text{contrast}
$$

を入れる。

→ 「RLで動きを学びながら、潜在空間もオノマトペに寄っていく」構造。

### **報酬関数は“意味側”を使う**

- “ざらざら”と“さらさら”をちゃんと分けるのは sem embedding。
- phon は未知語処理の補助なので reward には使わない。

---

# 4️⃣ **研究としての整理ポイント**

## **4-1. 研究目的を再定義**

> 曖昧な移動指示（特にオノマトペ）を
身体的モーションを通じて解釈・生成するモデルの提案
> 

→ Text-to-Motion に寄りすぎず、

→ Navigation-only に縛られず、

→ **身体性による意味理解** がテーマ。

## **4-2. 関連研究の再整理ポイント**

- 動作×言語の Text-to-Motion（MotionCLIP, TMR）
- 和語オノマトペの音韻構造と意味構造
- 曖昧指示（“ふわっと動いて”系）の人間ロボットインタラクション研究
- navigation の high-level → low-level mapping
- RL＋対照学習（CURL, C2RL, etc.）

---

# 5️⃣ **データ面の方向性**

## **5-1. 今すぐできること**

- 歩容（gait）データセットを可視化
    - joint range
    - 速度分布
    - 加速度
    - オノマトペ別の差分

→ ここで **オノマトペ×歩容の“地図”** を作る。

## **5-2. humanoid への remap**

- HOYO や sim-to-real ツールでリターゲット
- 人間の歩容 → humanoid の可動範囲へマッピング
- motionclip latent を変えず、関節 remap の層を作る

---

# 6️⃣ **未知オノマトペへの対応戦略**

### **音韻 → 意味 → モーション**の三段構え：

1. **音韻 p(w)p(w)p(w)**
    
    → 音から構造を学び、新語でも embedding 可能
    
2. **音韻→意味 F(p(w))F(p(w))F(p(w))**
    
    → 未知語の意味を予測
    
3. **意味→モーション**
    
    → 対照学習した latent space に投影
    
4. **low-level RL**
    
    → その latent に一致する動きを作る
    

---

# 7️⃣ **最終的なプロジェクト方針（まとめ）**

### ✔ オノマトペ理解の中心は「意味 × モーション」

（音韻は補助・未知語対応）

### ✔ MotionCLIP を追加学習してモーション空間を獲得

（速度/加速度をそのまま低レベルに渡すのはNG）

### ✔ PPO low-level は motion latent を使用

（身体性と質感を policy が使える形で渡す）

### ✔ 強化学習と対照学習を joint optimize

（対照損失は PPO loss に足す）

### ✔ ナビゲーションに縛られない

（移動系オノマトペの身体表現が本質）

---

# 🧭 **To-Do リスト（実務ベース）**

## 今週（モデル決定）

- [ ]  **phon / sem / motion encoder の構成を確定**
- [ ]  motion clip精読、境野論文精読, langWBC
- [ ]  **対照損失（phon→sem, sem↔motion, phon↔phon）の係数決定**
- [ ]  日本語CLIPのオノマトペ理解度の軽い予備実験
    
    → “さらさら”“ざらざら”“どしん”などで距離を調べる
    
- [ ]  MotionCLIP のコードベースどれ使うか決める
1. motion clip精読、motionclip実装を見て理解を深める
2. 埋め込みモデルがmotionclipにつなげれるのか？どうかの検討
3. 強化学習にどうHOYOのデータを報酬として組み込むのか
- [x]  歩容 dataset の可視化

## 来週（実装開始）

- [ ]  motion latent と semantic latent の距離分布をチェック
- [ ]  オノマトペ別の歩容差分を解析（速度・振幅・周期）
- [ ]  phon encoder の文字レベル前処理を実装
- [ ]  phon→sem マッピング MLP を軽くテスト

## 2–3 週後（対照学習実行）

- [ ]  三本の encoder を合同で pretrain
- [ ]  latent space を t-SNE 可視化
- [ ]  “ざらざら vs さらさら” が sem space で分離してるか確認
- [ ]  未知語（例：“ズリズリ”“フワリン”）の sem 予測が妥当か確認

## 4 週後（RL 接続）

- [ ]  motion latent を low-level policy へ入力
- [ ]  PPO + contrastive の joint training
- [ ]  style reward のチューニング
- [ ]  “らしさ”の定量化メトリクスを作る（rough/smooth 属性）

最初に大胆に、ざっくりと挑戦

それで労力を減らし、経過を丁寧に観察する

それで当たりをつけてから、細心に条件を設定する