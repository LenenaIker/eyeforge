"""Lyra 1 against photo2scene (plan: docs/plan_lyra1_2026-10-21.md). Lyra 1 turns one photo into 3D Gaussians: GEN3C
generates six short camera moves from the photo (left, right, up, back, forward, orbit) and a decoder turns their
latents into one 3DGS. The test feeds it the user's phone photos of the 2026-10-15 photo2scene baseline (full
resolution, landscape 16:9, as Lyra wants) and puts its renders next to that baseline's scene and new views
(data/lyra1_test/baseline/, saved from the review page). Run in edge-sdg (boto3 for the bucket):

    python experiments/lyra1/lyra1_test.py prep
    python experiments/lyra1/lyra1_test.py launch [--case photo21 ...] [--gpu l40s|h100] [--max-hours 4]
    python experiments/lyra1/lyra1_test.py fetch
    python experiments/lyra1/lyra1_test.py sheet

The job runs in our own image (experiments/lyra1/build.sh pushes it); its log, GEN3C's videos and Lyra's
renders and Gaussians go to s3://<cache bucket>/lyra1_test/out/, fetched into data/lyra1_test/out/."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip import config  # noqa: E402,F401  (loads .env)
from sim2chip.cloud_sync import S3Store  # noqa: E402

BUCKET = os.environ.get("SIM2CHIP_CACHE_BUCKET", "sim2chip-cache")
PREFIX = "lyra1_test"
DATA = ROOT / "data" / "lyra1_test"
W, H = 1280, 720  # GEN3C's depth step resizes to 1280x720 without cropping: the photo is cut to 16:9 first
# The 2026-10-15 baseline's photos with their full-resolution phone original (user's verdicts in the plan doc).
CASES = {
    "photo3": "data/real/roads/PXL_20261002_165349016.jpg",
    "photo12": "data/real/roads/PXL_20261002_165709768.jpg",
    "photo21": "data/real/roads/PXL_20261002_170012932.jpg",
    "photo24": "data/real/roads/PXL_20261002_170053761.jpg",
    "photo27": "data/real/roads/PXL_20261002_170243978.jpg",
    "photo31": "data/real/roads/PXL_20261002_170339779.jpg",
    "photo68": "data/real/roads/PXL_20261002_171601397.jpg",
    "bottles13": "data/real/count/PXL_20261003_112340460.jpg",
}
GPUS = {"l40s": ("gpu-l40s-d", "1gpu-16vcpu-96gb"), "h100": ("gpu-h100-sxm", "1gpu-16vcpu-200gb")}
# Lyra's README flags for GPUs under 80 GB (~43 GB peak). The prompt encoder is off (Lyra's own low-memory
# setting, a dummy T5), so no prompt and no 45 GB T5 download.
SDG_FLAGS = ("--offload_diffusion_transformer --offload_tokenizer --offload_text_encoder_model "
             "--offload_prompt_upsampler --offload_guardrail_models --disable_guardrail --disable_prompt_encoder "
             "--disable_prompt_upsampler")

JOB = r'''#!/bin/bash
exec > /tmp/job.log 2>&1
set -x
up() { curl -s -X PUT --upload-file /tmp/job.log "$LOG_URL" > /dev/null; }
trap up EXIT
( while true; do sleep 120; up; done ) &
cd /opt/lyra/Lyra-1
nvidia-smi
export HF_HUB_ENABLE_HF_TRANSFER=1
T0=$(date +%s)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/GEN3C-Cosmos-7B", local_dir="checkpoints/Gen3C-Cosmos-7B",
                  allow_patterns=["README.md", "model.pt", "mean_std.pt", "image_mean_std.pt", "config.json", "*.jit"])
snapshot_download("nvidia/Cosmos-Tokenize1-CV8x8x8-720p", local_dir="checkpoints/Cosmos-Tokenize1-CV8x8x8-720p")
snapshot_download("nvidia/Lyra", local_dir="checkpoints/Lyra", allow_patterns=["lyra_static.pt", "config.json"])
snapshot_download("Ruicheng/moge-vitl")
PY
echo "weights seconds $(( $(date +%s) - T0 ))"
# The decoder (with has_latents=False) loads the tokenizer from cosmos_predict1/.
mkdir -p checkpoints/cosmos_predict1
ln -sfn ../Cosmos-Tokenize1-CV8x8x8-720p checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p
mkdir -p /tmp/in /tmp/sdg /tmp/res
__CASES__
# The decoder: our generated latents as one dataset, Gaussians saved, every case.
cat >> src/models/data/registry.py <<'PY'

dataset_registry['eyeforge'] = dict(dataset_registry['lyra_static_demo_generated'])
dataset_registry['eyeforge']['kwargs'] = dict(dataset_registry['lyra_static_demo_generated']['kwargs'], root_path="/tmp/sdg")
__REGISTRY_EXTRA__
PY
sed -e 's#^out_dir_inference:.*#out_dir_inference: /tmp/lyra_out#' -e 's#^dataset_name:.*#dataset_name: eyeforge#' \
  configs/demo/lyra_static.yaml > /tmp/lyra.yaml
# lyra_static.yaml has no final newline: start on a fresh line.
printf '\nsave_gaussians: true\nnum_test_images: __N__\n' >> /tmp/lyra.yaml
cat /tmp/lyra.yaml
T0=$(date +%s)
accelerate launch sample.py --config /tmp/lyra.yaml
echo "decoder seconds $(( $(date +%s) - T0 ))"
find /tmp/lyra_out -type f | head -100
# Frames for the sheet (no video reader on the workstation): each case's six 3DGS renders at 0, 1/2 and the end.
python - <<'PY'
import json, pathlib, imageio.v3 as iio
from PIL import Image
for meta in pathlib.Path("/tmp/lyra_out").rglob("meta/sample_*.json"):
    i, fn = meta.stem.split("_")[1], json.loads(meta.read_text())["file_name"]
    fn = pathlib.Path(fn[0] if isinstance(fn, list) else fn).stem
    out = pathlib.Path("/tmp/lyra_out/frames") / fn
    out.mkdir(parents=True, exist_ok=True)
    for v in range(6):
        p = meta.parent.parent / "raw" / f"rgb_{i}_view_idx_{v}.mp4"
        if p.exists():
            fr = iio.imread(p, plugin="pyav")
            for tag, k in (("a", 0), ("b", len(fr) // 2), ("c", len(fr) - 1)):
                Image.fromarray(fr[k]).save(out / f"v{v}_{tag}.jpg", quality=92)
PY
tar -C /tmp -cf /tmp/res/lyra_out.tar lyra_out
curl -s -X PUT --upload-file /tmp/res/lyra_out.tar "__OUT_URL__" > /dev/null
up
'''

CASE = r'''
curl -s -o /tmp/in/{name}.png "{img}"
T0=$(date +%s)
CUDA_HOME=/usr/local/cuda PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
  --checkpoint_dir checkpoints --num_gpus 1 --input_image_path /tmp/in/{name}.png --video_save_folder /tmp/sdg \
  --foreground_masking --multi_trajectory --total_movement_distance_factor {factor} --prompt "" {flags}
echo "case {name} seconds $(( $(date +%s) - T0 ))"
tar -C /tmp/sdg -cf /tmp/res/{name}_gen3c.tar $(cd /tmp/sdg && find . -path "*/rgb/{name}.mp4" -o -path "*/pose/{name}.npz" -o -path "*/intrinsics/{name}.npz")
curl -s -X PUT --upload-file /tmp/res/{name}_gen3c.tar "{out}" > /dev/null
up
'''

# Decoder only: a case whose GEN3C videos an earlier job already uploaded comes back from the bucket.
CACHED = r'''
curl -s "{tar}" | tar -C /tmp/sdg -xf -
echo "case {name} restored"
'''


def image_tag() -> str:
    """build.sh's tag: the hash of the Dockerfile without comments."""
    lines = (ROOT / "experiments/lyra1/Dockerfile").read_text().splitlines(keepends=True)
    return hashlib.sha256("".join(ln for ln in lines if not ln.startswith("#")).encode()).hexdigest()[:12]


def prep():
    """Each phone photo cut to the centre 16:9 landscape band and resized to 1280x720."""
    d = DATA / "in"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in CASES.items():
        im = ImageOps.exif_transpose(Image.open(ROOT / src)).convert("RGB")
        w, h = im.size
        cw, ch = (w, round(w * 9 / 16)) if w * 9 / 16 <= h else (round(h * 16 / 9), h)
        box = ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)
        im.crop(box).resize((W, H), Image.LANCZOS).save(d / f"{name}.png")
        print(f"{name}: {w}x{h} -> crop {cw}x{ch} -> {W}x{H}")


def launch(cases: list[str], gpu: str, max_hours: float, factor: float, decode_only: bool = False):
    store = S3Store()
    exp = max_hours + 2
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    image = f"{os.environ['SIM2CHIP_NEBIUS_REGISTRY']}/eyeforge-lyra:{image_tag()}"

    def put_url(key):
        return store.s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/out/{key}"},
                                               ExpiresIn=int(exp * 3600))
    parts = []
    for n in cases:
        if decode_only:
            parts.append(CACHED.format(name=n, tar=store.url(BUCKET, f"{PREFIX}/out/{n}_gen3c.tar", exp)))
            continue
        store.put(BUCKET, f"{PREFIX}/in/{n}.png", (DATA / "in" / f"{n}.png").read_bytes())
        parts.append(CASE.format(name=n, img=store.url(BUCKET, f"{PREFIX}/in/{n}.png", exp), factor=factor,
                                 flags=SDG_FLAGS, out=put_url(f"{n}_gen3c.tar")))
    script = DATA / "job.sh"
    # Restored cases carry the videos but not GEN3C's latents: the decoder encodes the videos with its VAE.
    extra = ("dataset_registry['eyeforge']['kwargs'].update(has_latents=False, is_generated_cosmos_latent=False)"
             if decode_only else "")
    script.write_text(JOB.replace("__CASES__", "\n".join(parts)).replace("__REGISTRY_EXTRA__", extra).replace("__N__", str(len(cases)))
                      .replace("__OUT_URL__", put_url("lyra_out.tar")))
    assert script.stat().st_size < 60_000, "the script must fit Nebius' 64 KiB inject limit"
    platform, preset = GPUS[gpu]
    args = [cli, "ai", "job", "create", "--name", f"eyeforge-lyra1-{int(time.time())}", "--image", image,
            "--platform", platform, "--preset", preset, "--on-demand", "--timeout", f"{int(max_hours * 60)}m",
            "--parent-id", os.environ["SIM2CHIP_NEBIUS_PROJECT"], "--inject-file", f"{script}:/opt/lyra_job.sh",
            "--container-command", "bash", "--args", "/opt/lyra_job.sh",
            "--env", f"LOG_URL={put_url('job.log')}", "--async"]
    r = subprocess.run(args, capture_output=True, text=True)
    print(f"{len(cases)} cases on {gpu} ({image})", r.stdout[-300:], r.stderr[-500:])


def fetch():
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    store = S3Store()
    for k in store.listing(BUCKET, f"{PREFIX}/out/"):
        f = out / k.rsplit("/", 1)[-1]
        f.write_bytes(store.s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        if f.suffix == ".tar":
            with tarfile.open(f) as t:
                t.extractall(out / f.stem, filter="data")
        print(f)


def sheet(cell: int = 360):
    """Per case one JPG: row 1 the photo and photo2scene's scene from the photo's camera; row 2 photo2scene's
    first new views; rows 3-4 Lyra's 3DGS half way and at the end of its six moves (frames the job cut from
    sample.py's raw/rgb_<i>_view_idx_<v>.mp4; view order is lyra_static.yaml's: orbit, left, right, up, back,
    forward). Writes data/lyra1_test/sheet_<case>.jpg."""
    base = DATA / "baseline"
    for name in CASES:
        b = base / name
        fr = next((DATA / "out" / "lyra_out").rglob(f"frames/{name}*"), None)
        if fr is None:
            print(f"{name}: no Lyra renders yet")
            continue
        rows = [[Image.open(b / "photo.jpg")] + ([Image.open(b / "scene.jpg")] if (b / "scene.jpg").exists() else [])]
        rows.append([Image.open(p) for p in sorted(b.glob("view_*.jpg"))[:6]])
        for tag in ("b", "c"):
            rows.append([Image.open(p) for p in sorted(fr.glob(f"v*_{tag}.jpg"))])
        th = int(cell * 9 / 16)
        canvas = Image.new("RGB", (cell * 6, (th + 18) * len(rows)), "white")
        dr = ImageDraw.Draw(canvas)
        labels = ["photo / photo2scene 10-15", "photo2scene new views", "Lyra moves, half way", "Lyra moves, end"]
        for r, row in enumerate(rows):
            dr.text((4, r * (th + 18) + 2), labels[r], fill="black")
            for c, im in enumerate(row[:6]):
                canvas.paste(ImageOps.fit(im.convert("RGB"), (cell, th)), (c * cell, r * (th + 18) + 18))
        p = DATA / f"sheet_{name}.jpg"
        canvas.save(p, quality=88)
        print(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "launch", "fetch", "sheet"])
    ap.add_argument("--case", action="append", choices=list(CASES), help="default: all")
    ap.add_argument("--gpu", choices=list(GPUS), default="l40s")
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--decode-only", action="store_true", help="skip GEN3C, decode the videos a previous job "
                    "uploaded")
    ap.add_argument("--factor", type=float, default=1.0, help="Lyra's total_movement_distance_factor (2.0: more "
                    "camera motion, more artefacts)")
    a = ap.parse_args()
    if a.step == "prep":
        prep()
    elif a.step == "launch":
        launch(a.case or list(CASES), a.gpu, a.max_hours, a.factor, a.decode_only)
    elif a.step == "fetch":
        fetch()
    else:
        sheet()


if __name__ == "__main__":
    main()
