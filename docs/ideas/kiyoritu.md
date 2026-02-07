# PPO × 関節トルク制御：寄与率の取り方まとめ（学習時 / 評価時）

## ざっくり方針（迷ったらこれ）
- ① 報酬内訳（mag）：基本は **評価時**
- ② Advantage寄与（adv）：**評価時でも可**（要: $V(s)$ と $A_t$）。**学習時はデバッグ最強**
- ③ 関節マスク因果（abl-act）：**評価時オンリー**
- ④ エネルギー割合（share^E）：基本は **評価時**。学習時にも取ると「暴れ癖」検知に便利

---

## 前提：ロールアウトで取れるもの
時刻 $t=0,\dots,T-1$ について、
- 状態 $s_t$
- 行動（関節トルク） $a_t \in \mathbb{R}^d$
- 報酬の項別分解 $r_t^{(k)}$（$k=1,\dots,K$）
- 割引率 $\gamma$
- PPO critic の値 $V(s_t)$
- 評価指標 $J$（成功率、平均Return、time-to-success、安全違反率など）

---

# 1) 報酬内訳寄与率 rate^mag（「どの報酬項がどれだけ鳴ってた？」）

## 定義
平均絶対寄与（相殺を避ける）：
$$
m_k = \mathbb{E}\left[|r_t^{(k)}|\right]
$$

寄与率（%）：
$$
\mathrm{rate}^{mag}_k
= \frac{m_k}{\sum_{j=1}^{K} m_j}\times 100
$$

## いつ取る？
- 基本：評価時（推奨）
- 学習時：デバッグ用途ならアリ（shaping支配、罰則鳴りっぱなし等）

---

# 2) Advantage寄与率 rate^adv（「意思決定/学習信号に効いてる報酬項は？」）

## Advantage の作り方（評価向け：MCが分かりやすい）
モンテカルロ・リターン：
$$
G_t=\sum_{l=0}^{T-1-t}\gamma^l r_{t+l}
$$

Advantage：
$$
A_t = G_t - V(s_t)
$$

## 寄与率の定義（共分散ベース）
影響度スコア：
$$
I_k = \left|\mathrm{Cov}\left(r_t^{(k)}, A_t\right)\right|
$$

共分散：
$$
\mathrm{Cov}(x,y)=\mathbb{E}\left[(x-\mathbb{E}[x])(y-\mathbb{E}[y])\right]
$$

寄与率（%）：
$$
\mathrm{rate}^{adv}_k
= \frac{I_k}{\sum_{j=1}^{K} I_j}\times 100
$$

## いつ取る？
- 評価時：説明用に使える（「罰則は大きいのにAと無関係」などが見える）
- 学習時：本命（どの報酬項が勾配を支配してるかの診断）

---

# 3) 関節トルク因果寄与率 rate^abl-act（評価時マスクで因果っぽく）

## 定義
通常評価：$J_{full}$  
関節 $i$ を潰した評価：$J_{-i}$

性能低下：
$$
\Delta_i = J_{full} - J_{-i}
$$

寄与率（%）：
$$
\mathrm{rate}^{abl\text{-}act}_i
= \frac{\max(\Delta_i,0)}{\sum_{j=1}^{d}\max(\Delta_j,0)}\times 100
$$

## マスク（置換）
関節 $i$ だけ置換して、
$$
\tilde a^{(i)}_{t,j}=
\begin{cases}
a_{t,j} & (j\neq i)\\
g_i(s_t) & (j=i)
\end{cases}
$$

おすすめの $g_i(s_t)$：
- 平均行動：$g_i(s_t)=\mu_i(s_t)$（方策の平均）
- 次点：ホールド $a_{t-1,i}$
- 最後：$0$（不自然になりやすい）

## いつ取る？
- 評価時オンリー（学習中にやると別タスクになって学習が歪む）
- 全関節は重いので、まずは候補関節からでOK

---

# 4) 関節エネルギー割合 share^E（効率/暴れの可視化）

## 定義（簡易）
$$
E_i=\mathbb{E}\left[a_{t,i}^2\right]
$$

割合（%）：
$$
\mathrm{share}^E_i=\frac{E_i}{\sum_{j=1}^{d}E_j}\times 100
$$

（可能なら）パワー寄り：
$$
P_i=\mathbb{E}\left[|a_{t,i}\dot q_{t,i}|\right],\quad
\mathrm{share}^P_i=\frac{P_i}{\sum_{j}P_j}\times 100
$$

## いつ取る？
- 基本：評価時
- 学習時：発散/副作用の早期検知に便利

---

# 実務でのおすすめ運用（迷ったらこれ）

## 学習中（毎Nイテレーション）
- rate^adv：更新信号の診断
- share^E：暴れ/省エネの診断
- 余裕あれば rate^mag：報酬の鳴り方モニタ

## 評価（モデル比較・レポート）
- 成功率 / 平均Return / time-to-success / 安全違反
- rate^mag：報酬内訳
- share^E：関節の努力配分
- 追加で強い：rate^abl-act：関節寄与の決定版

---

## 注意
評価時の $A_t$ は、学習で使うGAEと完全一致せんでもOK。  
説明目的なら MC の $A_t=G_t-V(s_t)$ が分かりやすい。
