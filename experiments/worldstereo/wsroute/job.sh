#!/bin/bash
# The WorldStereo job, run in the image on the GPU VM (vm.py). cloud.job_script fills the placeholders (double
# underscores) and puts run_ws.py inside the heredoc below; LOG_URL comes as an environment variable.
exec > /tmp/job.log 2>&1
set -xeo pipefail  # a failed step ends the job instead of burning GPU time on the next one
mkdir -p /tmp/scene /tmp/out
up() { curl -s -X PUT --upload-file /tmp/job.log "$LOG_URL" > /dev/null || true; }
# Whatever legs exist go up every 5 min and at exit: the job timeout kills the container without running traps.
up_out() { tar -C /tmp -cf /tmp/out_up.tar out && curl -s -X PUT --upload-file /tmp/out_up.tar "__OUT_URL__" > /dev/null || true; }
trap 'echo "exit status $?"; up_out; up' EXIT
( set +xe; i=0; while true; do sleep 60; up; i=$((i + 1)); [ $((i % 5)) -eq 0 ] && up_out; done ) &
cd /opt/WorldStereo
nvidia-smi
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
export HF_XET_HIGH_PERFORMANCE=1 TORCHDYNAMO_DISABLE=1  # torch.compile off: minutes of compiling for 8 runs
T0=$(date +%s)
python - <<'PY'
from huggingface_hub import snapshot_download
# The DMD checkpoint holds the whole transformer: the base one's 65 GB are never read (run_ws.py below).
snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                  ignore_patterns=["transformer/diffusion_pytorch_model*"] if "dmd" in "__MODEL__" else None)
snapshot_download("hanshanxue/WorldStereo", allow_patterns=["__MODEL__/config.json", "__MODEL__/model.safetensors"])
snapshot_download("facebook/dinov2-base")
PY
echo "weights seconds $(( $(date +%s) - T0 ))"
curl -sf "__SCENE_URL__" | tar -C /tmp/scene -xf -
ls /tmp/scene/__CASE__
# Resume: earlier legs' results go back where the runner looks; it skips them and loads them into its memory.
if [ -n "__RESUME_URL__" ]; then curl -sf "__RESUME_URL__" | tar -C /tmp/out -xf -; find /tmp/out -name "*_result.mp4"; fi
T0=$(date +%s)
# Upstream keeps the fp32 umT5 (23 GB) on the GPU next to the 17B transformer: 80 GB ran out on the first leg
# (2026-10-21). The wrapper parks the text encoder on the CPU and lends it the GPU only while it encodes.
cat > /tmp/run_ws.py <<'PY'
__RUN_WS__
PY
torchrun --nproc_per_node=1 /tmp/run_ws.py --model_type __MODEL__ --task_type panorama \
  --input_path /tmp/scene/__CASE__ --output_path /tmp/out
# (no --local_files_only: with it transformers cannot load the base model's tokenizer, it looks for a root
# config.json the Diffusers repo lacks)
echo "generation seconds $(( $(date +%s) - T0 ))"
find /tmp/out -type f | head -200
