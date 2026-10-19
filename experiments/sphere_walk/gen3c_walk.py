"""Walk test arm E, GEN3C (github.com/nv-tlabs/GEN3C; code Apache 2.0, weights NVIDIA Open Model License) on a
Nebius L40S. Its single-image mode builds a 3D cache from MoGe's depth of one view and generates a 121-frame video
along a camera path. Here the view is cut from our erased panorama (pitch 0, 16:9 as GEN3C wants), MoGe's depth is
scaled to ours (the job patches GEN3C to read GEN3C_METRIC_DEPTH), and the path goes straight ahead to FAR_M, so
frames at 3, 5, 8 and 13 m line up with the other arms' forward views. Run in edge-sdg (boto3 for the bucket):

    python experiments/sphere_walk/gen3c_walk.py launch --case <place>:<heading> [--case ...] [--max-hours 3]
    python experiments/sphere_walk/gen3c_walk.py fetch

The job's log and videos go to s3://<SIM2CHIP_CACHE_BUCKET or sim2chip-cache>/gen3c_test/out/, fetched into
data/sphere_test/<place>/walk/E_gen3c/."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip import config  # noqa: E402,F401  (loads .env)
from sim2chip.cloud_sync import S3Store  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

BUCKET = os.environ.get("SIM2CHIP_CACHE_BUCKET", "sim2chip-cache")
PREFIX = "gen3c_test"
W, H, HFOV = 1280, 704, 80.0
FAR_M = 13.0
FRAMES = 121
IMAGE = "nvcr.io/nvidia/pytorch:24.10-py3"  # Python 3.10, torch, Transformer Engine and Apex already in
WHERE = ("gpu-l40s-d", "1gpu-16vcpu-96gb")  # T5-11B's 45 GB load in RAM: the 32 GB preset is too small
PROMPT = "A walk through a street on a sunny day, realistic, sharp, natural light."

JOB = r'''#!/bin/bash
exec > /tmp/job.log 2>&1
set -x
up() { curl -s -X PUT --upload-file /tmp/job.log "$LOG_URL" > /dev/null; }
trap up EXIT
( while true; do sleep 120; up; done ) &
# NVIDIA's container points pip at an index Nebius cannot resolve: plain PyPI instead.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_CONSTRAINT  # NGC pins its own versions there
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple
pip config list; cat /etc/pip.conf 2>/dev/null
cd /tmp && git clone -q --depth 1 https://github.com/nv-tlabs/GEN3C.git && cd GEN3C
grep -vE "^(torch|torchvision|opencv-python)==" requirements.txt > req.txt
# The container's torch and Transformer Engine stay as they are (numpy goes 1.24 to GEN3C's 1.26) (an upgraded torch broke TE, 2026-10-18).
pip list --format=freeze | grep -iE "^(torch|torchvision|transformer.engine|transformer_engine|apex)==" > keep.txt; cat keep.txt
pip install -q -r req.txt opencv-python-headless==4.10.0.84 hf_transfer -c keep.txt
# MoGe of March 2025 (GEN3C's time; today's wants numpy 2) and its utils3d, without their dependencies.
pip install -q --no-deps "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
  "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
pip install -q click trimesh -c keep.txt
# The container's OpenCV and the headless one installed over it mix into a broken cv2: only the headless one.
pip uninstall -y -q opencv opencv-python opencv-contrib-python opencv-python-headless
rm -rf /usr/local/lib/python3.10/dist-packages/cv2
pip install -q --no-deps opencv-python-headless==4.10.0.84
python -c "import cv2, huggingface_hub, moge, megatron.core, transformer_engine, decord, hydra; from moge.model.v1 import MoGeModel; print('imports ok')" || exit 1
python - <<'PY'
p = "cosmos_predict1/diffusion/inference/gen3c_single_image.py"
s = open(p).read()
old = '    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full)'
assert old in s
s = s.replace(old, old + """
    import os as _os
    if _os.environ.get("GEN3C_METRIC_DEPTH"):
        _ref = torch.from_numpy(np.load(_os.environ["GEN3C_METRIC_DEPTH"])).float().to(moge_depth_hw_full.device)
        _ref = F.interpolate(_ref[None, None], size=moge_depth_hw_full.shape, mode="nearest")[0, 0]
        _ok = (moge_mask_hw_full > 0) & (_ref > 0) & (_ref < 40)
        _k = torch.median(_ref[_ok] / moge_depth_hw_full[_ok])
        print("metric scale", float(_k), flush=True)
        moge_depth_hw_full = moge_depth_hw_full * _k""")
# GEN3C_HFOV: the view's known field of view instead of MoGe's guess, so the cache is built with our intrinsics.
old = "    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)"
assert old in s
s = s.replace(old, """    if _os.environ.get("GEN3C_HFOV"):
        _f = target_w / 2 / np.tan(np.radians(float(_os.environ["GEN3C_HFOV"])) / 2)
        moge_intrinsics_33 = torch.tensor([[_f, 0, target_w / 2], [0, _f, target_h / 2], [0, 0, 1]],
                                          dtype=moge_intrinsics_33.dtype, device=moge_intrinsics_33.device)
""" + old)
# GEN3C_TRAJ: our own path (T x 4 x 4 world to camera, the input camera's frame, metres) instead of a preset.
old = '        log.info(f"Generating 0 - {sample_n_frames} frames")'
assert old in s
s = s.replace(old, """        if os.environ.get("GEN3C_TRAJ"):
            generated_w2cs = torch.from_numpy(np.load(os.environ["GEN3C_TRAJ"])).float().to(device)[None]
            generated_intrinsics = initial_cam_intrinsics_for_traj[None, None].repeat(1, generated_w2cs.shape[1], 1, 1)
""" + old)
open(p, "w").write(s)
PY
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/GEN3C-Cosmos-7B", local_dir="checkpoints/Gen3C-Cosmos-7B",
                  allow_patterns=["README.md", "model.pt", "mean_std.pt", "image_mean_std.pt", "config.json", "*.jit", "guardrail/*"])
snapshot_download("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", local_dir="checkpoints/Cosmos-Tokenize1-CV8x8x8-720p")
snapshot_download("google-t5/t5-11b", local_dir="checkpoints/google-t5/t5-11b",
                  allow_patterns=["config.json", "pytorch_model.bin", "spiece.model", "tokenizer.json"])
snapshot_download("Ruicheng/moge-vitl")
PY
nvidia-smi
mkdir -p /tmp/in /tmp/out
__CASES__
'''

CASE = r'''
curl -s -o /tmp/in/{name}.png "{img}"
curl -s -o /tmp/in/{name}.npy "{dep}"
{fetch}T0=$(date +%s)
{env}GEN3C_METRIC_DEPTH=/tmp/in/{name}.npy CUDA_HOME=/usr/local/cuda PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_single_image.py \
  --checkpoint_dir checkpoints --input_image_path /tmp/in/{name}.png --video_save_name {name} --video_save_folder /tmp/out \
  --guidance 1 --foreground_masking --trajectory zoom_in --camera_rotation no_rotation --movement_distance {far} \
  --num_video_frames {frames} --prompt "{prompt}" --offload_text_encoder_model
echo "case {name} seconds $(( $(date +%s) - T0 ))"
curl -s -X PUT --upload-file /tmp/out/{name}.mp4 "{out}" > /dev/null
up
'''


def view(heading: float) -> dict:
    return {"yaw_deg": float(heading), "pitch_deg": 0.0, "fov_deg": HFOV, "size": [W, H]}


def inputs(place: str, heading: int) -> tuple[np.ndarray, np.ndarray]:
    """The view of the erased panorama (never-seen ground painted, B_layers' front texture) and its metric z-depth
    from our proxy distance."""
    d = ROOT / "data" / "sphere_test" / place
    R = np.array(json.loads((d / "geometry.json").read_text())["levelling"])
    v = view(heading)
    rays_l = capture.rays(*v["size"], v["fov_deg"], v["yaw_deg"], v["pitch_deg"])
    rays_p = render_plan.pano_rays(rays_l, R)
    tex = np.asarray(Image.open(d / "walk" / "BL_scene" / "tex_front.png").convert("RGB"))
    img = np.clip(capture.sample(tex, rays_p) + 0.5, 0, 255).astype(np.uint8)
    pd = np.load(d / "proxy_distance.npz")
    dist = np.where(pd["valid"], pd["distance"], 0).astype(np.float32)
    x, y = capture.to_pixel(rays_p, dist.shape[1], dist.shape[0])
    dd = dist[np.clip(np.round(y).astype(int), 0, dist.shape[0] - 1), np.round(x).astype(int) % dist.shape[1]]
    cam_z = rays_l @ capture.rotation(heading, 0.0)[:, 2]  # cosine to the optical axis
    return img, (dd * cam_z).astype(np.float32)


def launch(cases: list[str], max_hours: float):
    store = S3Store()
    parts = []
    for c in cases:
        place, heading = c.split(":")
        name = f"{place}_h{int(heading):03d}"
        img, dep = inputs(place, int(heading))
        tmp = ROOT / "data" / "sphere_test" / place / "walk" / "E_gen3c"
        tmp.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(tmp / f"{name}_input.png")
        np.save(tmp / f"{name}_depth.npy", dep)
        store.put(BUCKET, f"{PREFIX}/in/{name}.png", (tmp / f"{name}_input.png").read_bytes())
        store.put(BUCKET, f"{PREFIX}/in/{name}.npy", (tmp / f"{name}_depth.npy").read_bytes())
        put = store.s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/out/{name}.mp4"},
                                              ExpiresIn=int((max_hours + 2) * 3600))
        parts.append(CASE.format(name=name, img=store.url(BUCKET, f"{PREFIX}/in/{name}.png", max_hours + 2),
                                 dep=store.url(BUCKET, f"{PREFIX}/in/{name}.npy", max_hours + 2), out=put,
                                 far=FAR_M, frames=FRAMES, prompt=PROMPT, fetch="", env=""))
    log_url = store.s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/out/job.log"},
                                              ExpiresIn=int((max_hours + 2) * 3600))
    script = JOB.replace("__CASES__", "\n".join(parts))
    path = ROOT / "data" / "sphere_test" / "gen3c_job.sh"
    path.write_text(script)
    assert path.stat().st_size < 60_000, "the script must fit Nebius' 64 KiB inject limit"
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    project = os.environ["SIM2CHIP_NEBIUS_PROJECT"]
    args = [cli, "ai", "job", "create", "--name", f"eyeforge-gen3c-{int(time.time())}", "--image", IMAGE,
            "--platform", WHERE[0], "--preset", WHERE[1], "--on-demand", "--timeout", f"{int(max_hours * 60)}m",
            "--parent-id", project, "--inject-file", f"{path}:/opt/gen3c_job.sh", "--container-command", "bash",
            "--args", "/opt/gen3c_job.sh", "--env", f"LOG_URL={log_url}", "--async"]
    out = subprocess.run(args, capture_output=True, text=True)
    print(out.stdout[-500:], out.stderr[-800:])


def fetch():
    store = S3Store()
    keys = store.listing(BUCKET, f"{PREFIX}/out/")
    for k in keys:
        name = k.rsplit("/", 1)[-1]
        if name == "job.log":
            dst = ROOT / "data" / "sphere_test" / "gen3c_job.log"
        else:
            place = name.split("_h")[0]
            dst = ROOT / "data" / "sphere_test" / place / "walk" / "E_gen3c" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(store.s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        print(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["launch", "fetch"])
    ap.add_argument("--case", action="append", default=[])
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()
    launch(args.case, args.max_hours) if args.step == "launch" else fetch()


if __name__ == "__main__":
    main()
