# 定量評価の方法

## 実行コマンド（現状）

### 強化学習（Docker）

```bash
docker run --rm -it --name navila-gui --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY=$DISPLAY \
  -e LIVESTREAM=0 \
  -e QT_X11_NO_MITSHM=1 \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/jouta/.Xauthority:/root/.Xauthority:ro \
  -v /home/jouta/isaac/cache/kit:/isaac-sim/kit/cache \
  -v /home/jouta/.cache/ov:/root/.cache/ov \
  -v /home/jouta/.nv/ComputeCache:/root/.nv/ComputeCache \
  -v /home/jouta/isaac/logs:/root/.nvidia-omniverse/logs \
  -v /isaac/Documents:/root/Documents \
  -v /home/jouta/NaVILA-Bench:/workspace/NaVILA-Bench \
  -w /workspace/NaVILA-Bench \
  navila_docker
```

### 対照学習（/home/jouta/venvs/motionclip）

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

## **対照学習（CL）の評価：表現の健全性**

オノマトペとモーションの埋め込みが、潜在空間で正しく対応しているかを評価する。

### **主要指標（Retrieval）**

- **R@K**
    - 「トップK件（1位, 3位, 5位）以内に正解が含まれている確率」。
    - **高いほど良い**（検索精度が高い）。
- **MedR (Median Rank)**
    - 「正解が出てくる順位の真ん中（中央値）」。
    - **低いほど良い**（例: MedR=1 なら、だいたい1位に正解が出てくる優秀なモデル）。

### **補助指標（Clustering）**

 散布図上で、同じスタイルの点がちゃんと固まっているか？

- **Silhouette係数**
    - 1〜+1 の値をとる。
    - **+1に近い**: スタイルごとに綺麗に分かれている
    - **0に近い**: 境界が曖昧で混ざっている。
    
    **複数実験する（パラメータを変える？）**

---

## **3. 強化学習の評価：**

獲得したポリシーが、指示通りかつタスクを完遂できているかを評価する。

### **評価指標**

| 指標名 | 評価内容 |
| --- | --- |
| cos⁡(z_motion, z_teacher) | 生成された挙動が教師モーションの埋め込みとどれだけ近いか。 |
| 転倒率 | 物理的な安定性。 |
| HOYOとの誤差 | HOYOデータセットの教師動作に対して、物理的な追従度を確認する。
データセットとの関節位置、速度、重心の誤差を評価する

どう位相を合わせるのか？
**DTW (Dynamic Time Warping)を使えばいいか？** |
| 人間の評価 | motionを見て、人がどのオノマトペに近いかの分類確率 |

---

> オノマトペ認識の個人差
> 
- 「ふらふら」という言葉から想起される動きには個人差があり、また同じ動きに対しても人によって付けるオノマトペが異なる

これはオノマトペとロボットの運動を結びつけるうえで解決すべき1つの課題
どうアプローチするのか

# **５. 今後の課題**

- [ ]  未知のオノマトペの対応をどうするか
- [ ]  パラメータもっと変えて実験
- [ ]  style付与の論文のsurveyをし、定量評価がどう行われているか確認
- [ ]  motionclip以外のモデルでstyle付与する実験も必要
- [x]  階段じゃなくて、平面でも実験する
