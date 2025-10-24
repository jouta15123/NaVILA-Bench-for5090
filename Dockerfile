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