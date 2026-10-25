# WorldStereo 2.0 (github.com/FuchengSu/WorldStereo; code Apache 2.0, weights MIT) for a Nebius H100 job:
# PyTorch's own 2.7.1 devel image (upstream pins torch 2.7.1) plus WorldStereo's code at a fixed commit and
# pytorch3d compiled for Ada (L40S, 8.9), Hopper (H100, 9.0) and Blackwell (RTX PRO 6000, 12.0; needs CUDA 12.8). No weights: the job downloads them from Hugging
# Face. Built and pushed by build.sh.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG WORLDSTEREO_COMMIT=5ba8dc3d37c48ecef99bbd603815ced513a4a431
ENV TORCH_CUDA_ARCH_LIST="8.9 9.0 12.0" MAX_JOBS=12 FORCE_CUDA=1 PIP_NO_CACHE_DIR=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq git curl libgl1 libglib2.0-0 libgomp1 > /dev/null \
    && rm -rf /var/lib/apt/lists/*
RUN git clone -q https://github.com/FuchengSu/WorldStereo.git /opt/WorldStereo \
    && git -C /opt/WorldStereo checkout -q $WORLDSTEREO_COMMIT
WORKDIR /opt/WorldStereo

# Upstream's requirements against the image's torch; the headless OpenCV (no X in the job). decord has no
# wheel for every Python: eva-decord is its packaged fork.
RUN pip list --format=freeze | grep -iE "^(torch|torchvision)==" > /opt/keep.txt && cat /opt/keep.txt \
    && sed -e 's/^opencv-python==/opencv-python-headless==/' -e '/^decord/d' requirements.txt > /tmp/req.txt \
    && pip install -q -r /tmp/req.txt open3d==0.19.0 hf_transfer -c /opt/keep.txt \
    && (pip install -q decord || pip install -q eva-decord)
RUN pip install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
# MapAnything (code Apache 2.0, weights facebook/map-anything-apache Apache 2.0): the per-leg depth of ws_cache_job.py.
# Without its declared dependencies: rerun-sdk (a viewer) and torchaudio would pull numpy 2.x over upstream's pin;
# only what inference imports is added.
RUN pip install -q --no-deps "git+https://github.com/facebookresearch/map-anything.git@3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9" \
        uniception==0.1.7 \
    && pip install -q hydra-core natsort orjson python-box jaxtyping scikit-learn termcolor torchmetrics minio \
        pillow-heif numpy==1.26.4 -c /opt/keep.txt

RUN python -c "import cv2, decord, open3d, kornia, pytorch3d.renderer, diffusers, transformers, imagesize; \
import sys; sys.path.insert(0, '.'); from models.worldstereo_wrapper import WorldStereo; \
from src.retrieval_wm import SimpleMemoryBank; from mapanything.models import MapAnything; from mapanything.utils.image import preprocess_inputs; \
import numpy; assert numpy.__version__ == '1.26.4'; print('imports ok')"
