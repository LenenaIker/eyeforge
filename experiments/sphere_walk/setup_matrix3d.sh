#!/bin/bash
# Walk test arm D: Matrix-3D (MIT) environment and weights on Crucial, only what panoramic_image_to_video.py needs
# with --use_5b_model (no 3D reconstruction, no FLUX text-to-panorama, no super-resolution).
set -e
C=/media/lenena-iker/Crucial
ENV=$C/envs/matrix3d
M=$C/tools/Matrix-3D
export PIP_CONFIG_FILE=/dev/null PIP_CACHE_DIR=$C/tools/tmp/pipcache TMPDIR=$C/tools/tmp HF_HUB_CACHE=$C/tools/hf_cache
mkdir -p $TMPDIR
[ -d $ENV ] || ~/miniforge3/condabin/conda create -y -q -p $ENV python=3.10
~/miniforge3/condabin/conda install -y -q -p $ENV -c nvidia/label/cuda-12.8.1 cuda-nvcc cuda-cudart-dev cuda-crt
P=$ENV/bin/pip
$P install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
export CUDA_HOME=$ENV
cd $M && git submodule update --init --depth 1 submodules/nvdiffrast
$P install -q --no-build-isolation $M/submodules/nvdiffrast
$P install -q "setuptools<80" wheel
$P install -q --no-build-isolation -e $M/code/DiffSynth-Studio
$P install -q plyfile decord trimesh pyrender ninja xfuser diffusers==0.34.0 open3d py360convert peft easydict einops \
  "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" pandas modelscope==1.28.2 transformers==4.56.0 imageio-ffmpeg==0.6.0 \
  scikit-image scipy "opencv-python-headless==4.10.0.84" matplotlib==3.8.4 omegaconf
rm -rf $C/tools/tmp/pipcache
# Weights: Wan2.2-TI2V-5B where DiffSynth looks for it, the 5B LoRA (the script reads wan-lora/), MoGe v1.
cd $M/code
$ENV/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir="checkpoints/Wan-AI/Wan2.2-TI2V-5B")
hf_hub_download("Skywork/Matrix-3D", "checkpoints/pano_video_gen_720p_5b.safetensors", local_dir="checkpoints/_m3d")
hf_hub_download("Ruicheng/moge-vitl", "model.pt", local_dir="checkpoints/moge")
PY
mkdir -p checkpoints/Wan-AI/wan-lora
ln -sf ../../_m3d/checkpoints/pano_video_gen_720p_5b.safetensors checkpoints/Wan-AI/wan-lora/
echo SETUP DONE
