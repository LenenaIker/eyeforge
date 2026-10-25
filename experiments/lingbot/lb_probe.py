"""Probe of LingBot-World v1 (github.com/robbyant/lingbot-world, Apache 2.0, Wan2.2 A14B with camera-pose control)
on photo21: does one long generation keep a place the same when the camera comes back to it?

WorldStereo invents each leg on its own and the legs contradict each other (docs/pruebas_reconstruccion_worldstereo.md).
LingBot generates the whole walk as one video, so the question is whether its implicit memory holds over a loop:
out 8 m along walk 0 (past the silver SUV), turn round, back to the start, turn to face the start view again. The
last frame should match the first; the way back sees the way out from the other side.

    <gsplat env>/bin/python experiments/lingbot/lb_probe.py stage    # start image, poses, prompt, job script
    python experiments/lingbot/lb_probe.py launch [--dry]            # edge-sdg, after `set -a; . ./.env; set +a`
    python experiments/lingbot/lb_probe.py fetch                     # video and log
    <gsplat env>/bin/python experiments/lingbot/lb_probe.py legs     # split into legs for ws_depth.py

Upstream conventions (read from its code): poses.npy are camera-to-world in OpenCV axes, one per frame, and only
their direction counts (translations are normalised per clip); intrinsics.npy are [fx, fy, cx, cy] in pixels of an
832x480 frame. Its examples use fx 415 to 503, so the probe uses 415 (a 90 degree view), not WorldStereo's 240.
Single GPU on a Nebius Serverless RTX PRO 6000 (96 GB): T5 on the CPU, both experts in bf16 on the GPU, PyTorch's
SDPA instead of flash-attn (no wheel for Blackwell in the image)."""
import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "worldstereo"))
import ws_test as ws  # noqa: E402

CASE = "photo21"
OUT = ws.DATA / CASE / "lingbot"
COMMIT = "a43bec7f8091c83e9b30b16b912f6fc906236fa6"
FX = 415.5
FRAMES = 121           # 4n + 1
OUT_M = 8.0            # metres along walk 0, as reconstruct_0's two legs
STEPS = 20             # upstream's default is 70; its action example uses 20
SEED = 1024
MAX_MIN = 60
PREFIX = "lingbot_probe"


def trajectory() -> np.ndarray:
    """c2w (FRAMES, 4, 4), world z up as ws_test: out 40 frames, turn 20, back 40, turn 20."""
    yaw0 = json.loads((ws.DATA / CASE / "prep.json").read_text())["walks"][0][0]
    d = np.array([math.cos(math.radians(yaw0)), math.sin(math.radians(yaw0)), 0.0])
    poses = []
    for f in range(FRAMES):
        if f <= 40:
            pos, yaw = d * OUT_M * f / 40, yaw0
        elif f <= 60:
            pos, yaw = d * OUT_M, yaw0 + 180 * (f - 40) / 20
        elif f <= 100:
            pos, yaw = d * OUT_M * (100 - f) / 40, yaw0 + 180
        else:
            pos, yaw = d * 0, yaw0 + 180 + 180 * (f - 100) / 20
        poses.append(np.linalg.inv(ws.w2c(pos, yaw, ws.PITCH)))
    return np.stack(poses).astype(np.float32)


def stage():
    import cv2
    src = OUT / "in" / "probe"
    src.mkdir(parents=True, exist_ok=True)
    c2w = trajectory()
    meta = json.loads((ws.DATA / CASE / "prep.json").read_text())
    full = np.asarray(Image.open(ws.PANOS / CASE / meta["pano"]).convert("RGB"))
    ws.F = FX  # ws_test.crop reads the module's focal length
    Image.fromarray(ws.crop(full, np.linalg.inv(c2w[0]))).save(src / "image.jpg", quality=95)
    np.save(src / "poses.npy", c2w)
    np.save(src / "intrinsics.npy", np.tile(np.array([FX, FX, ws.W / 2, ws.H / 2], np.float32), (FRAMES, 1)))
    cap = json.loads((ws.DATA / CASE / "scene" / "reconstruct_0" / "traj0" / "traj_caption.json").read_text())
    (src / "prompt.txt").write_text(cap["prompt"])
    print(f"{src}: {FRAMES} frames, start yaw {meta['walks'][0][0]}, image {cv2.imread(str(src / 'image.jpg')).shape}")


JOB = r'''#!/bin/bash
exec > /tmp/job.log 2>&1
set -xeo pipefail
up() { curl -s -X PUT --upload-file /tmp/job.log "$LOG_URL" > /dev/null || true; }
(while true; do sleep 300; up; done) &
trap up EXIT
nvidia-smi; free -g; df -h /tmp
mkdir -p /tmp/in /tmp/out
curl -sf "__IN_URL__" | tar -x -C /tmp/in
pip install -q easydict ftfy
git clone -q https://github.com/robbyant/lingbot-world.git /opt/lb && git -C /opt/lb checkout -q __COMMIT__
# flash-attn is not in the image: route the model's attention through upstream's SDPA fallback
sed -i 's/from .attention import flash_attention$/from .attention import attention as flash_attention/' /opt/lb/wan/modules/model.py
grep -n "import attention as flash_attention" /opt/lb/wan/modules/model.py
export HF_HUB_ENABLE_HF_TRANSFER=1
t0=$(date +%s)
python -c "from huggingface_hub import snapshot_download; snapshot_download('robbyant/lingbot-world-base-cam', local_dir='/tmp/lingbot-world-base-cam', ignore_patterns=['assets/*', 'examples/*'])"
echo "download $(( $(date +%s) - t0 )) s"; up
cd /opt/lb
t0=$(date +%s)
python generate.py --task i2v-A14B --size '480*832' --ckpt_dir /tmp/lingbot-world-base-cam \
  --image /tmp/in/probe/image.jpg --action_path /tmp/in/probe --frame_num __FRAMES__ --sample_steps __STEPS__ \
  --offload_model False --t5_cpu --convert_model_dtype --base_seed __SEED__ \
  --prompt "$(cat /tmp/in/probe/prompt.txt)" --save_file /tmp/out/probe.mp4
echo "generate $(( $(date +%s) - t0 )) s"
cp /tmp/job.log /tmp/out/
tar -C /tmp/out -cf /tmp/out.tar .
curl -sf -X PUT --upload-file /tmp/out.tar "__OUT_URL__"
'''


def launch(dry: bool):
    sys.path.insert(0, str(ROOT))
    from sim2chip import config  # noqa: F401  (loads .env)
    from sim2chip.cloud_sync import S3Store
    store = S3Store()
    exp = MAX_MIN / 60 + 2
    tar = OUT / "in.tar"
    with tarfile.open(tar, "w") as t:
        t.add(OUT / "in" / "probe", arcname="probe")
    store.put(ws.BUCKET, f"{PREFIX}/in.tar", tar.read_bytes())

    def put_url(key):
        return store.s3.generate_presigned_url("put_object", Params={"Bucket": ws.BUCKET, "Key": f"{PREFIX}/{key}"},
                                               ExpiresIn=int(exp * 3600))
    script = OUT / "job.sh"
    script.write_text(JOB.replace("__IN_URL__", store.url(ws.BUCKET, f"{PREFIX}/in.tar", exp))
                      .replace("__OUT_URL__", put_url("out.tar")).replace("__COMMIT__", COMMIT)
                      .replace("__FRAMES__", str(FRAMES)).replace("__STEPS__", str(STEPS)).replace("__SEED__", str(SEED)))
    image = f"{os.environ['SIM2CHIP_NEBIUS_REGISTRY']}/eyeforge-worldstereo:{ws.image_tag()}"
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    platform, preset, project = ws.GPUS["rtx6000"]
    args = [cli, "ai", "job", "create", "--name", f"eyeforge-lingbot-{int(time.time())}", "--image", image,
            "--platform", platform, "--preset", preset, "--on-demand", "--timeout", f"{MAX_MIN}m",
            "--disk-size", "250Gi", "--shm-size", "32Gi", "--parent-id", project,
            "--inject-file", f"{script}:/opt/lb_job.sh", "--container-command", "bash", "--args", "/opt/lb_job.sh",
            "--env", f"LOG_URL={put_url('job.log')}", "--async"]
    if dry:
        print(" ".join(args[:args.index("--env") + 1]), "LOG_URL=<presigned>", f"\njob script: {script}")
        return
    token = subprocess.run([cli, "iam", "get-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    args += ["--registry-username", "iam", "--registry-password", token]
    r = subprocess.run(args, capture_output=True, text=True)
    print(f"lingbot probe ({image})", r.stdout[-300:], r.stderr[-500:])


def fetch():
    sys.path.insert(0, str(ROOT))
    from sim2chip import config  # noqa: F401
    from sim2chip.cloud_sync import S3Store
    store = S3Store()
    for key in ("job.log", "out.tar"):
        try:
            data = store.s3.get_object(Bucket=ws.BUCKET, Key=f"{PREFIX}/{key}")["Body"].read()
        except Exception as e:  # noqa: BLE001
            print(key, "not there yet:", type(e).__name__)
            continue
        (OUT / key).write_bytes(data)
        print(key, len(data), "bytes")
    if (OUT / "out.tar").exists():
        with tarfile.open(OUT / "out.tar") as t:
            t.extractall(OUT / "out")
        print(sorted(p.name for p in (OUT / "out").iterdir()))


def legs():
    """The video as two legs for ws_depth.py (out: frames 0-40, back: 60-100) next to WorldStereo's, each with a
    camera.json in WorldStereo's format (w2c, K) under legsets/lingbot."""
    import imageio.v3 as iio
    frames = iio.imread(OUT / "out" / "probe.mp4", plugin="pyav")
    c2w = trajectory()
    print(f"{len(frames)} frames {frames.shape[1:]}")
    K = [[FX, 0.0, ws.W / 2], [0.0, FX, ws.H / 2], [0.0, 0.0, 1.0]]
    for name, a, b in (("lb_out", 0, 41), ("lb_back", 60, 101)):
        res = ws.DATA / CASE / "out" / "out" / CASE / name / "traj0"
        cam = ws.DATA / CASE / "legsets" / "lingbot" / name / "traj0"
        res.mkdir(parents=True, exist_ok=True)
        cam.mkdir(parents=True, exist_ok=True)
        iio.imwrite(res / f"{ws.MODEL}_result.mp4", frames[a:b], fps=16, codec="libx264")
        json.dump({"extrinsic": np.linalg.inv(c2w[a:b]).tolist(), "intrinsic": [K] * (b - a),
                   "width": ws.W, "height": ws.H}, open(cam / "camera.json", "w"))
        print(name, b - a, "frames ->", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "launch", "fetch", "legs"])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"stage": stage, "launch": lambda: launch(a.dry), "fetch": fetch, "legs": legs}[a.cmd]()
