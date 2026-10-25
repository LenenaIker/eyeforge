# Lyra 1 (github.com/nv-tlabs/lyra, Lyra-1/; code Apache 2.0, weights NVIDIA Open Model License) for a Nebius
# L40S or H100 job: NVIDIA's PyTorch container (the one GEN3C already ran in, gen3c_walk.IMAGE) plus Lyra's code
# at a fixed commit and every package that needs compiling, built here once for Ada (L40S, 8.9) and Hopper
# (H100, 9.0). No weights: the job downloads them from Hugging Face. Built and pushed by build.sh.
FROM nvcr.io/nvidia/pytorch:24.10-py3

ARG LYRA_COMMIT=9fffc9adc37004091ecf26ef03abfb3abdf4d59a
ENV TORCH_CUDA_ARCH_LIST="8.9 9.0" MAX_JOBS=12 \
    PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_CONSTRAINT= PIP_NO_CACHE_DIR=1

RUN git clone -q https://github.com/nv-tlabs/lyra.git /opt/lyra && git -C /opt/lyra checkout -q $LYRA_COMMIT
WORKDIR /opt/lyra/Lyra-1

# The container's torch, Transformer Engine and Apex stay as they are (an upgraded torch broke TE for GEN3C,
# 2026-10-18), so every install below is constrained to them.
RUN pip list --format=freeze | grep -iE "^(torch|torchvision|transformer.engine|transformer_engine|apex|flash.attn|flash_attn)==" > /opt/keep.txt \
    && cat /opt/keep.txt \
    && grep -vE "^(torch|torchvision|opencv-python)==" requirements_gen3c.txt > /tmp/req_gen3c.txt \
    && pip install -q -r /tmp/req_gen3c.txt hf_transfer -c /opt/keep.txt

# MoGe of March 2025 (GEN3C's time; today's wants numpy 2) and its utils3d, without their dependencies.
RUN pip install -q --no-deps "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
    && pip install -q click trimesh -c /opt/keep.txt

# The 3DGS decoder's packages (requirements_lyra.txt); the compiled ones without build isolation, against
# the container's torch. flash-attn only if the container lacks it.
RUN python -c "import flash_attn" 2>/dev/null || pip install -q --no-build-isolation flash_attn==2.7.4.post1
RUN pip install -q --no-build-isolation "git+https://github.com/Dao-AILab/causal-conv1d@v1.4.0" -c /opt/keep.txt
RUN pip install -q --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.4" -c /opt/keep.txt
RUN pip install -q --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@73fad53c31ec4d6b088470715a63f432990493de" \
    "git+https://github.com/rahul-goel/fused-ssim/@8bdb59feb7b9a41b1fab625907cb21f5417deaac" -c /opt/keep.txt
RUN pip install -q timm==1.0.19 kiui==0.2.17 lru-dict==1.3.0 mpi4py==4.1.0 plyfile==1.1.2 deepspeed==0.17.5 \
    accelerate==1.10.0 openexr==3.2.3 -c /opt/keep.txt

# The container's OpenCV and the headless one installed over it mix into a broken cv2: only the headless one.
RUN pip uninstall -y -q opencv opencv-python opencv-contrib-python opencv-python-headless; \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2 \
    && pip install -q --no-deps opencv-python-headless==4.10.0.84

# megatron.core and transformer_engine ask for a GPU on import, which docker build lacks: checked in the job.
RUN python -c "import cv2, moge, decord, hydra, gsplat, mamba_ssm, causal_conv1d, \
fused_ssim, flash_attn, deepspeed, accelerate, kiui; from moge.model.v1 import MoGeModel; print('imports ok')"
