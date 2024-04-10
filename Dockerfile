FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# System Installations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake curl ffmpeg git \
    libatlas-base-dev libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev libboost-system-dev libboost-test-dev \
    libhdf5-dev libcgal-dev libeigen3-dev libflann-dev libfreeimage-dev libgflags-dev libglew-dev libgoogle-glog-dev \
    libmetis-dev libprotobuf-dev libqt5opengl5-dev libsqlite3-dev libsuitesparse-dev \
    nano protobuf-compiler python-is-python3 python3-dev python3-pip qtbase5-dev \
    sudo vim-tiny build-essential libcudnn8$ \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Python installations (from requriements.txt)
RUN pip3 install --upgrade pip
RUN python3 -m pip install --upgrade pip

RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA Architectures:
# 
# | CUDA Architecture | Example Graphics Cards                 |
# |-------------------|----------------------------------------|
# | 20                | NVIDIA Tesla C2050, GeForce GTX 48 0   |
# | 21                | GeForce GTX 560 Ti, Tesla K20          |
# | 30                | GeForce GTX 680, Tesla K10             |
# | 35                | GeForce GTX 780 Ti, Tesla K40          |
# | 37                | Tesla K80                              |
# | 50                | GeForce GTX 970, Tesla M40             |
# | 52                | GeForce GTX 980 Ti, Tesla M60          |
# | 53                | Tesla M4                               |
# | 60                | GeForce GTX 1080, Tesla P100           |
# | 61                | Tesla P40                              |
# | 62                | Tesla P4                               |
# | 70                | Tesla V100                             |
# | 72                | Tesla T4                               |
# | 75                | GeForce RTX 2080, Tesla T4             |
# | 80                | GeForce RTX 3080, Tesla A100           |
# | 86                | GeForce RTX 3090                       |
# | 89                | 40X0                                   |
# | 90                | H100                                   |
# |-------------------|----------------------------------------|

ENV CUDA_ARCHITECTURES=86
ENV TCNN_CUDA_ARCHITECTURES=86
RUN export CUDA_ARCHITECTURES=86
RUN export TCNN_CUDA_ARCHITECTURES=86
RUN pip3 install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN pip3 install \
    mmdet==2.20.0 mmengine==0.8.4 mmsegmentation==0.20.0 mmcls==0.25.0 mmcv-full==1.5.0 \
    pyyaml==6.0.1 imageio==2.33.1 imageio-ffmpeg==0.4.9 lpips==0.1.4 \
    pytorch-msssim==1.0.0 kornia==0.7.0 yapf==0.40.1 seaborn==0.13.2 \
    crc32c pandas tensorboardX jupyterlab matplotlib \
    opencv-python scikit-image tqdm torchmetrics

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=6789","--allow-root","--no-browser"]

EXPOSE 6789

# taking roughly 30 minutes 
# BUILD: docker build -t 6img-to-3d .