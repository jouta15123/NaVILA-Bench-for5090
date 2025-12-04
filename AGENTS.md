## 概要
- このリポジトリは、VLM「NaVILA」を用いて Isaac Sim 上でロボットのナビゲーションを行うための実装です。
- ベンチマーク実装（評価用の挙動）は `original_scripts` 内にある元コードに含まれています。
- `scripts` には任意の言語指示で動かすための試験的な実装がありましたが、完走率が低く未完成でした。

## 目的（このブランチ／拡張でやりたいこと）
- ベンチマークの挙動（NaVILA + 低レベル方策 + VLM 推論）を「任意の言語指示」で再現すること。
- ベンチマークのように複数エピソードは不要。単一エピソードでの実行でよい。
- 別途、任意の部屋環境（シーン）をロードできるようにする。
- original_scripts/のファイルは変更しない

## ディレクトリ構成と追加スクリプト
- `original_scripts/`　元のnavila開発者のお手本code
- `scripts/`  改変した未完成code(navila_interactive.py , navila_vla_utils.py)
 

## シングルエピソードで任意指示を再現する
- 目的: ベンチマークの挙動（VLM 推論→速度指令→低レベル方策）を保ちつつ、指示だけを任意に差し替える
- 実行フローはベンチマーク準拠（画像バッファ生成→VLM問い合わせ→速度コマンド→環境ステップ）
- 指示は部屋環境を確認してから送れる方法で行いたい（socket通信を行うか、isaac sim実行ターミナルから直接送るか）




## 使い方

### Isaac sim側

Dockerfile（Isaac Sim 4.5.0 ベース、IsaacLab/NaVILA-Bench/NaVILA をセットアップ）
```bash
#Dockerfile
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# set up terminal
ENV TERM=xterm-256color
SHELL ["/bin/bash", "-lc"]

# tool installation
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl build-essential ca-certificates python3-venv python3-dev\
 && rm -rf /var/lib/apt/lists/*

# miniconda install
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR \
 && rm -f /tmp/mc.sh
RUN conda config --system --set auto_update_conda false \
 && conda config --system --set show_channel_urls true \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# isaac-lab navila-bench navila download
WORKDIR /workspace
RUN git clone -c advice.detachedHead=false -b v2.0.2 --depth=1 https://github.com/isaac-sim/IsaacLab.git \
 && git clone -c advice.detachedHead=false -b isaaclab-2.0 --depth=1 https://github.com/yang-zj1026/NaVILA-Bench.git \
 && git clone --depth=1 https://github.com/AnjieCheng/NaVILA.git

# connect isaac_lab with isaac_sim and NaVILA-bench
ENV ISAACSIM_PATH="/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
RUN ln -s ${ISAACSIM_PATH} /workspace/IsaacLab/_isaac_sim \
 && ln -s /workspace/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce /workspace/IsaacLab/source/ \
 && ln -s /workspace/NaVILA-Bench/isaaclab_exts/omni.isaac.matterport /workspace/IsaacLab/source/
RUN /workspace/IsaacLab/isaaclab.sh -i none \
 && /workspace/IsaacLab/isaaclab.sh -p -m pip install -e /workspace/NaVILA-Bench/scripts/rsl_rl

# setup torch for cu128 and install gym
RUN /isaac-sim/python.sh -m pip uninstall -y torch torchvision torchaudio \
 && /isaac-sim/python.sh -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio \
 && /isaac-sim/python.sh -m pip install numpy==1.26.4 opencv-python==4.11.0.86 gym rl-games stable_baselines3 tensordict

# create assets directory to mount usds
RUN mkdir /workspace/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets

# enable WebRTC
ENV LIVESTREAM=2

# create conda environment for NaVILA
RUN conda create -n navila-eval -y python=3.10 pip setuptools wheel \
 && conda clean -afy

# set for start
WORKDIR /workspace/NaVILA-Bench
ENTRYPOINT ["/bin/bash"]
```

Docker イメージのビルドと起動
```bash
docker build -t navila_docker -f Dockerfile .

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

デモプランナー起動例
```bash
cd /workspace/NaVILA-Bench/scripts
/workspace/IsaacLab/isaaclab.sh -p demo_planner.py --task=h1_matterport_vision --enable_cameras --load_run=2024-11-03_15-08-09_height_scan_obst
```

単一エピソード（ベンチマーク実装相当）
```bash
/workspace/IsaacLab/isaaclab.sh -p original_scripts/navila_eval.py \
  --task=h1_matterport_vision --num_envs=1 \
  --load_run=2024-11-03_15-08-09_height_scan_obst --enable_cameras --episode_idx=0
```

任意指示対応のシングルエピソード実行（`scripts/navila_single_episode.py`）

デフォルトではベンチマーク同様にデータセットの指示を使用しますが、同時に `--instruction_host` / `--instruction_port` で指定したソケットを常にリッスンしており、送信された指示はその場で上書きされます。
```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/navila_single_episode.py \
  --task=h1_matterport_vision --num_envs=1 \
  --load_run=2024-11-03_15-08-09_height_scan_obst --enable_cameras --episode_idx=0
```

部屋を見渡してから初回指示を出したい場合は `--instruction_mode=socket` を指定します。シーンがロードされるとロボットは停止したまま待機し、ソケットから受け取った文が初回指示として使われます。
```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/navila_single_episode.py \
  --task=h1_matterport_vision --num_envs=1 \
  --load_run=2024-11-03_15-08-09_height_scan_obst --enable_cameras --episode_idx=0 \
  --instruction_mode=socket --instruction_host=127.0.0.1 --instruction_port=5557

# 別ターミナル例
echo "Go to the sofa and stop." | nc 127.0.0.1 5557
```

手元CLIから直接文字列を渡す場合は `--instruction_mode=text --instruction_text="..."` を利用できます（ソケットで送った指示はその後でも上書き可能）。

部屋環境（シーン）を変更
```bash
/workspace/IsaacLab/isaaclab.sh -p scripts/navila_interactive.py \
  --task=h1_matterport_vision --num_envs=1 \
  --load_run=2024-11-03_15-08-09_height_scan_obst --enable_cameras \
  --episode_idx=0 --scene_id_override=QUCTc6BB5sX
```

### VLM サーバ側

サーバ起動
```bash
python original_scripts/vlm_server.py --model_path ~/models/navila-llama3-8b-8f/ --port=54321
```

参考パッケージ（例）
```bash
absl-py==2.3.1
accelerate==0.27.2
... (省略せず載せる場合は、提示されたリストを使用)
```

### データセット準備（Matterport3D）
```bash
sudo apt-get update && sudo apt-get install -y aria2
cd ~/NaVILA/evaluation/data/scene_datasets
mkdir -p mp3d
cd mp3d
aria2c -c -x16 -s16 -k1M \
  "http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip" \
  -o mp3d_habitat.zip
```

## ポート設計
- **VLM 用ポート**: `vlm_server.py --port` と評価側 `--vlm_port` を一致させる（デフォルト 54321）
- 指示の送信方法は運用で選択（例: Isaac Sim 実行ターミナル操作、将来的にソケット連携など）

## 現在進行中のタスク：HoYo + MotionCLIP 対照学習
オノマトペ（意味的特徴）とモーション（動作的特徴）の共有潜在空間を構築し、オノマトペ指示に基づく動作生成や評価を行うための学習実験です。

### 目的
- 日本語オノマトペ（例：「すたすた」「のろのろ」）と、それに対応する歩行モーションの関係を学習する。
- **MotionCLIP** のモーションエンコーダと、テキストエンコーダ（**Sarashina BERT**）を Joint Embedding 空間へ射影し、対照学習（Contrastive Learning）を行う。
- これにより、「オノマトペ → モーション」の生成や、「現在のモーション → オノマトペっぽいか」の評価（報酬計算）を可能にする。

### 実験環境
- 仮想環境パス: `/home/jouta/venvs/motionclip/`
- データセット: **HoYo Dataset**（11種類のオノマトペラベル付き歩行モーション）
- 実装ファイル:
    - `hoyo_v1_1/train_motionclip_joint.py`: 学習メインスクリプト
    - `hoyo_v1_1/hoyo_sem_motion_contrastive_motionclip.py`: データセット定義、モデルユーティリティ

### 学習コマンド例
```bash
# motionclip環境のpythonを使用
/home/jouta/venvs/motionclip/bin/python hoyo_v1_1/train_motionclip_joint.py \
  --stage full \
  --steps 5000 \
  --lambda-contrastive 0.1 \
  --batch-size 32 \
  --lr 1e-5
```

---

## 今後の計画：強化学習による質感（Style）の付与
学習した HoYo + MotionCLIP モデルを報酬関数として利用し、ロボットのナビゲーション動作にオノマトペの質感を付与します。

### 目的
- ユーザーが「てくてく歩いて」と指示した際に、ナビゲーションの軌道追従だけでなく、その歩き方のスタイル（質感）も反映させる。

### アプローチ
- **強化学習 (RL)** を用いて、既存のナビゲーション方策（あるいは低レベル歩行制御）に追加学習または Fine-tuning を行う。
- **スタイル報酬 (Style Reward)** を導入する。

### 実装方針
- 実装箇所: `scripts/style_reward_module.py` (現在スタブ実装)
- **報酬計算の仕組み**:
    1. **Observation**: ロボットの関節角度やベース速度などの履歴（Window Size: 60フレーム等）。
    2. **Encoding**:
        - 履歴を HOYO フォーマット（14関節, 2D座標）にリターゲティング変換。
        - 学習済み MotionCLIP Encoder に通して潜在ベクトル $z_{motion}$ を取得。
    3. **Target**:
        - 指示オノマトペ（例：「のろのろ」）をテキストエンコーダに通し、投影層を経て潜在ベクトル $z_{text}$ を取得。
    4. **Reward**:
        - $Reward_{style} = \beta \cdot \text{CosineSimilarity}(z_{motion}, z_{text})$
        - 動作が指示オノマトペの意味内容に近いほど高い報酬を与える。

### 課題
- **リターゲティング**: H1ロボット（人型）の関節構造から、HOYOデータセット（簡易スケルトン）への変換ロジックの実装。
- **推論コスト**: 毎ステップ（あるいは数ステップごと）に Transformer Encoder を回すため、リアルタイム性の確保が必要。

## 参考資料
- `NaVILA_paper.md`: NaVILA の元論文まとめ
- `original_scripts/navila_eval.py`: ベンチマークの元となる評価実装
- `scripts/`: 参考用の従来スクリプト（未完成）
