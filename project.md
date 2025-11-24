# 研究方針決定

---

# 🔥 研究全体の方針（改訂版）

## 0. 大前提

やりたいこと：

> オノマトペの曖昧さを**身体運動（Motion）**で理解・生成できるモデルを作る  
> ＝ 未知オノマトペでも“それっぽい歩き方”を出せるようにする

テーマは **「移動系オノマトペ × 身体性」**。  
Text-to-Motion でも単なる Navigation でもなく、語の意味を身体表現で理解する立場に立つ。

---

# 1️⃣ 音韻 × 意味 × モーションの三本エンコーダ

## 1-1. 役割

- **phon encoder（音韻）**  
  オノマトペの文字列→音素/記号パターン。促音「っ」や長音「ー」などをきちんと扱う。未知語対応の要。
- **sem encoder（意味）**  
  文脈説明や LLM 生成文→意味ベクトル。Sarashina, SigLIP 等の日本語埋め込みを活用。
- **motion encoder（モーション）**  
  MotionCLIP をベース。HOYO 2D gait データで fine-tune して latent \(z_p\) を得る。

## 1-2. 三方向の対照学習

1. **sem ↔ motion（主軸）**  
   – “さらさら” vs “ざらざら” を身体動作で分離。  
2. **phon ↔ phon（補助）**  
   – 表記ゆれや繰り返し構造を学習し、音が似る語同士を近づける。  
3. **phon → sem（未知語ブリッジ）**  
   – phon latent から sem latent を推定し、未知オノマトペの意味推定を可能にする。

> これで「音は似てるのに意味が真逆」問題を避けつつ、未知語の推論まで行ける。

---

# 2️⃣ MotionCLIP latent の役割

- 日本語埋め込み（Sarashina / SigLIP）と MotionCLIP を alignment させる。  
- **低レベルRLには速度・加速度ではなく latent \(z_p\)** を渡す。  
  – “ささっと”の軽さ、 “もたもた”の遅さ/ふらつき など質感を policy が直接扱える。
- LangWBC でいう teacher/student 構造を参考に、  
  1. **Teacher**: MotionCLIP 既存VAEを凍結 → 再構成性能を保持。  
  2. **Student**: CVAE/contrastive でオノマトペ含む latent を獲得。

---

# 3️⃣ RL (PPO) との統合

## 3-1. Policy 設計

オノマトペ latent を条件にした policy：

$$
\pi(a \mid s,\; z_{\text{onm}})
$$


## 3-2. Style Reward

エージェントの生成モーション \(z_{\text{mot(agent)}}\) が  
指示オノマトペの意味ベクトル \(z_{\text{sem(onm)}}\) に近いほど、ご褒美：

$$
r = r_{\text{task}}
  + \beta \cdot 
  \cos\left(
      z_{\text{mot(agent)}},\;
      z_{\text{sem(onm)}}
  \right)
$$


## 3-3. Joint Optimization

動作学習（PPO）と意味一致（対照学習）を同時に最適化：

$$
\mathcal{L}
=
\mathcal{L}_{\text{PPO}}
+
\lambda\;
\mathcal{L}_{\text{contrast}}
$$

LangWBC と同様に  
**freeze → encoder-only → full fine-tune**  
の段階的学習で安定化を図る。


---

# 4️⃣ 研究としての整理

## 4-1. 目的（再定義）
> 曖昧な移動指示（オノマトペ）を身体モーションを通じて解釈・生成するモデルの構築

- Text-to-Motion に寄りすぎず、Navigation限定でもない。  
- **身体性による語意味理解** が主テーマ。

## 4-2. 関連研究
MotionCLIP / TMR, オノマトペ研究、LangWBC, RobotMDM, CURL/C2RL など。

---

# 5️⃣ データとリターゲット

1. **HOYO / gait データの解析**  
   – joint range, velocity, acceleration, step length, 周期などをオノマトペ別に地図化。  
2. **Humanoid へのリターゲット**  
   – HOYO 2D → H1/Isaac skeleton へ。MotionCLIP latent を壊さない補正層を追加。

---

# 6️⃣ 未知オノマトペ対応

音韻 → 意味 → モーション → RL の四段構え

1. phon encoder で文字列を latent 化  
2. phon→sem マッピングで意味推定  
3. sem→motion latent へ投影  
4. RL が latent に沿う動きを生成

これで “ズリズリ” “フワリン” など未登録語でも“それっぽい”行動に。

---

# 7️⃣ 最終まとめ

- ✔ 「意味 × モーション」の alignment が中心（音韻は未知語補助）  
- ✔ MotionCLIP latent を policy 入力に使う（質感を直接渡す）  
- ✔ PPO + Contrastive を同時に最適化  
- ✔ ナビゲーションではなく“歩きの質感”がテーマ

---

# 🧭 To-Do（改訂版）

## ✅ 今週：要件定義 & Teacher/Student 設計
- [ ] phon / sem / motion encoder 構成を確定  
- [ ] MotionCLIP / LangWBC を精読。freeze→encoder→full の段階設計をまとめる  
- [ ] 三方向の contrastive loss と重みを決定  
- [ ] Sarashina vs SigLIP のオノマトペ距離を評価  
- [ ] MotionCLIP コードベース統合方針を決める  
- [x] HOYO 歩容の可視化（済）

## ▶ 来週：データ整備 + 前処理
- [ ] motion / semantic latent の距離分布をチェック  
- [ ] オノマトペ別の歩容差分（速度・周期・揺れ）を解析  
- [ ] phon encoder 前処理（文字正規化・音素化）  
- [ ] phon→sem MLP のプロトタイプ

## ▶ 2–3 週後：対照学習
- [ ] 三本の encoder（phon/sem/motion）を pretrain  
- [ ] latent space を t-SNE/PCA で可視化  
- [ ] “さらさら vs ざらざら” の分離を検証  
- [ ] 未知語（例：ズリズリ, フワリン）の sem 予測実験

## ▶ 4 週後：RL 接続
- [ ] motion latent を low-level policy に入力  
- [ ] PPO + contrastive の joint training  
- [ ] Style reward のチューニング  
- [ ] “歩行の質感”を測るメトリクスを定義

> 最初に大胆に、経過を丁寧に、最後は細心に条件を整える。