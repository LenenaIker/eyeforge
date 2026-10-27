"""Nebius backend: the pipeline runs in Serverless AI jobs on an L40S, with its data in Object Storage.

Worker mode (default): one job runs every stage (cloud_worker.py). An L40S can take many minutes to get, or
not come at all (measured 2026-09-26: 3 of 4 on-demand jobs failed for lack of capacity), so a run asks for
one GPU once. Stage mode (SIM2CHIP_NEBIUS_MODE=stage): one job per stage, each in its own image.

Data stays in Object Storage between stages: the user uploads the spec and input once, every job pulls
what its stage reads to local disk, runs it and pushes what it wrote (see cloud_sync.py). Only small
results (done markers, reports, the exported model) come back to the local job folder, which keeps
working as the pipeline's view of the job.

    job bucket    s3://<prefix>-<job>   spec.json, input/, <stage>/, logs/, _code/
    cache bucket  s3://<prefix>-cache   objaverse, usd, hdri, shared by jobs

Configuration (environment or the repo's .env):
    SIM2CHIP_NEBIUS_CLI        nebius CLI (default: on PATH or ~/.nebius/bin/nebius)
    SIM2CHIP_NEBIUS_PROJECT    project id (default: the CLI profile's parent-id)
    SIM2CHIP_NEBIUS_PROJECTS   projects in other regions ("region=id,..."), where runs go when REGION has no
                               machine (platforms(): SIM2CHIP_NEBIUS_ELSEWHERE, SIM2CHIP_SM120_KITS)
    SIM2CHIP_BUCKET_PREFIX     bucket name prefix (default sim2chip)
    SIM2CHIP_S3_SECRET         MysteryBox secret with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for the jobs
    SIM2CHIP_NEBIUS_SECRET     MysteryBox secret holding NEBIUS_API_KEY, for stages that call Token Factory
    SIM2CHIP_MAX_USD           spend cap for one run, estimated from GPU time (default 5)
    SIM2CHIP_NEBIUS_REGISTRY   registry of the worker's own images (cr.<region>.nebius.cloud/<id>), and
    SIM2CHIP_IMAGE_TAG         their tag (images/build.sh); without them every run installs at start
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   Object Storage key, to upload input and fetch results
"""
from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path
from secrets import token_hex

from .. import config
from ..cloud_sync import REGION, S3Store
from ..stages import Stage
from ..workspace import DONE, LOG_PREFIX as PREFIX, REPO, Cache, Job

# The GPU and its price go together: set all three to try another platform (for instance when eu-north1 has
# no L40S left). Isaac Sim needs a GPU with RT cores (L40S yes, H100/H200 no).
PLATFORM = config.env("SIM2CHIP_NEBIUS_PLATFORM", "gpu-l40s-a")
PRESET = config.env("SIM2CHIP_NEBIUS_PRESET", "1gpu-8vcpu-32gb")
USD_PER_HOUR = config.env_float("SIM2CHIP_USD_PER_HOUR", 1.55)  # on-demand L40S (Intel), nebius.com/prices, 2026-09
# Where to ask when PLATFORM has no free machine (2026-10-13: none for over 20 min): the same L40S on AMD hosts,
# whose smallest preset is bigger. "" turns it off.
FALLBACK = config.env("SIM2CHIP_NEBIUS_FALLBACK", "gpu-l40s-d/1gpu-16vcpu-96gb")
NO_GPU_S = 1800  # a platform that found no machine is asked last for this long
# On-demand price of the smallest preset per platform (nebius.com/prices, 2026-10-13 and 10-21); others cost
# USD_PER_HOUR.
PRICES = {"gpu-l40s-a": USD_PER_HOUR, "gpu-l40s-d": 1.82, "gpu-rtx6000-a": 1.80, "gpu-rtx6000": 1.80}
# The L40S is only in eu-north1 (2026-10-27, every region listed with the CLI); the other GPUs with RT cores are
# RTX PRO 6000 (Blackwell), asked for when eu-north1 has no machine. "region:platform/preset", comma-separated;
# a region without a project in SIM2CHIP_NEBIUS_PROJECTS is left out.
ELSEWHERE = config.env("SIM2CHIP_NEBIUS_ELSEWHERE", "eu-south1:gpu-rtx6000-a/1gpu-24vcpu-218gb,"
                       "uk-south2:gpu-rtx6000-a/1gpu-24vcpu-218gb,us-central1:gpu-rtx6000/1gpu-24vcpu-218gb")
# Project per region other than REGION: "region=project-id", comma-separated.
PROJECTS = dict(kv.split("=", 1) for kv in config.env("SIM2CHIP_NEBIUS_PROJECTS", "").split(",") if "=" in kv)
ARCH = {"gpu-l40s-a": "sm_89", "gpu-l40s-d": "sm_89", "gpu-rtx6000-a": "sm_120", "gpu-rtx6000": "sm_120"}
# Kits shown to run on Blackwell (sm_120). Isaac Sim 5.1 does; Paddle cu126 does not, and gsplat's wheel has no
# sm_120 code (2026-10-27). Until a kit is listed, runs that need it stay on the L40S.
SM120_KITS = set(filter(None, config.env("SIM2CHIP_SM120_KITS", "").split(",")))


class Where(NamedTuple):
    """Where a job runs. Data stays in REGION's buckets: a job elsewhere reads them across regions (0.015 $/GiB,
    about 0.17 $ a run for the kits) and pulls REGION's images with a token."""
    platform: str
    preset: str
    region: str = REGION

    @property
    def arch(self) -> str:
        return ARCH.get(self.platform, "sm_89")


def usd_per_hour(platform: str | None) -> float:
    return PRICES.get(platform or PLATFORM, USD_PER_HOUR)
WORKER_HOURS = config.env_int("SIM2CHIP_WORKER_HOURS", 3)  # timeout of a whole run in one job
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz"
WORKER_DEPS = ["tensorflow[and-cuda]==2.18.1", "numpy<2.1", "pillow", "objaverse", "boto3", "usd-core==26.8",
               "trimesh==5.1.0",  # these two convert models (glb2usd)
               # PicoDet's ONNX to TFLite (detector.to_tflite): onnx2tf and what it imports
               "onnx==1.17.0", "onnx_graphsurgeon==0.6.1", "sng4onnx==2.0.1", "tf_keras==2.18.0",
               "onnxruntime==1.20.1", "ai-edge-litert==2.2.0", "psutil"]
# Installed without their dependencies: onnx2tf pins its own (TensorFlow 2.19 among them), and works with ours.
WORKER_NODEPS = ["onnx2tf==1.29.24"]
# The teacher (SAM 3, label stage) gets its own environment: torch in the same one as TensorFlow, each with its
# own CUDA libraries, is not worth the risk. Only installed when the run labels photos.
TEACHER_DEPS = ["torch==2.11.0", "torchvision==0.26.0", "transformers==5.17.0", "accelerate", "pillow"]
# PicoDet-S training (detector_worker.py, counting on profiles with room for a detector): PaddlePaddle, its own
# CUDA libraries again, and what PaddleDetection 2.8.1 imports to train and export. Only fetched when the run
# trains such a counter. paddlepaddle-gpu is not on PyPI: the wheel comes from Paddle's server, the rest from PyPI
# (Paddle's index mirrors it too slowly from Europe, 2026-10-04).
PICODET_DEPS = ["paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/"
                "paddlepaddle_gpu-3.0.0-cp311-cp311-linux_x86_64.whl", "paddle2onnx==2.1.0", "numpy==1.26.4",
                "opencv-python-headless==4.5.5.64", "pycocotools", "scipy", "shapely", "terminaltables", "typeguard",
                "Cython", "PyYAML", "tqdm", "lapx", "Pillow", "packaging", "setuptools<70", "requests", "pandas",
                "scikit-learn", "scikit-image", "imageio", "matplotlib", "six",
                # paddle2onnx folds constants with these; without them onnx2tf cannot convert its ONNX
                "onnx==1.17.0", "polygraphy==0.53.6", "onnx_graphsurgeon==0.6.1", "onnxruntime==1.20.1",
                "onnxoptimizer==0.3.13"]
# imgaug wants opencv-python, which needs X libraries the images lack: the headless one above does instead.
PICODET_NODEPS = ["imgaug==0.4.0"]
PADDLEDET_REF = "release/2.8.1"
# The walk scenes' backgrounds (scene_render, sim2chip.walkscene.background): gsplat's only prebuilt wheels are for
# Python 3.10 and torch 2.4 (docs.gsplat.studio/whl, 2026-10-27), with sm_86 code that runs on the L40S (sm_89).
# The kit holds its own Python 3.10. Only fetched when the run renders walk scenes.
GSPLAT_DEPS = ["torch @ https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl",
               "gsplat @ https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
               "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-linux_x86_64.whl",
               "numpy==1.26.4", "pillow", "plyfile==1.1.3",
               # gsplat loads its compiled code through these (torch's cpp_extension) but does not declare them
               "packaging", "setuptools"]
# Brought back for the web (status, report, review) after a worker run; the rest stays in the bucket.
WORKER_RESULTS = ("*.json", "export/*", "label/*", "ingest/test/*", "qa_assets/sheets/*", "preview_assets/*_0.png",
                  "generate_assets/*", "logs/*", "scene_render/*.json")
RENDER_SAMPLE = 48
JOB_DIR, CACHE_DIR = "/tmp/s2c/job", "/tmp/s2c/cache"
CLOUD_JOB = "cloud_job.json"  # in logs/: the Nebius job of a run in progress


@dataclass(frozen=True)
class Runtime:
    image: str
    python: str
    setup: str = ""
    env: dict = field(default_factory=dict)
    kits: bool = False  # the Pythons come as kits from the cache bucket (our own images)


# One L40S preset for every stage: the eu-north1 quota has no CPU-only instances, and Isaac needs RT cores.
RUNTIMES = {
    "isaac": Runtime("nvcr.io/nvidia/isaac-sim:5.1.0", "/isaac-sim/python.sh",
                     env={"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"}),
    "tf": Runtime("tensorflow/tensorflow:2.18.0-gpu", "python3", "pip install -q pillow"),
    "cpu": Runtime("tensorflow/tensorflow:2.18.0-gpu", "python3",
                   "pip install -q objaverse pillow numpy usd-core==26.8 trimesh==5.1.0"),
}
# The worker's own images and Python kits (images/Dockerfile, published by images/build.sh): a run installs
# nothing, it unpacks the kits from the cache bucket (envs/<tag>/), the teachers' only when it labels photos.
# Without them every run uses NVIDIA's public Isaac Sim image and installs at start.
REGISTRY = config.env("SIM2CHIP_NEBIUS_REGISTRY")  # cr.<region>.nebius.cloud/<registry id>
IMAGE_TAG = config.env("SIM2CHIP_IMAGE_TAG")
ENVS_PREFIX, KITS = "envs", ("pythons", "py", "teacher", "picodet", "gsplat")
ENVS = "/tmp/s2c/envs"
KIT_PY, KIT_TEACHER = f"{ENVS}/py/bin/python", f"{ENVS}/teacher/bin/python"
KIT_PICODET, KIT_PADDLEDET = f"{ENVS}/picodet/bin/python", f"{ENVS}/paddledet"  # the picodet kit holds both
KIT_GSPLAT = f"{ENVS}/gsplat/bin/python"  # its kit holds its Python 3.10 too (gsplat-python/)
KIT_TIMES = "/tmp/s2c/kit_times.txt"  # per streamed kit: index, start and end (epoch s), bytes
# A job Nebius cannot give a machine to this long is cancelled and asked for on the next platform (2026-10-13:
# PROVISIONING, its instance FAILED, retried for over 12 min); waiting longer only keeps the user staring at a run
# that never starts.
PROVISION_S = config.env_float("SIM2CHIP_NEBIUS_PROVISION_S", 300)


class NoGPU(RuntimeError):
    """Nebius found no machine for a job in PROVISION_S."""


def _sick_last(home: Path | None, out: list[Where]) -> list[Where]:
    """out with the places that found no machine lately (home/logs/no_gpu.json) moved last, in order otherwise."""
    try:
        sick = json.loads((home / "logs" / "no_gpu.json").read_text()) if home else {}
    except (OSError, ValueError):
        sick = {}
    return sorted(out, key=lambda w: time.time() - sick.get(_sick_key(w), 0) < NO_GPU_S)


def _sick_key(w: Where) -> str:
    return w.platform if w.region == REGION else f"{w.region}:{w.platform}"


def _where(text: str, region: str = REGION) -> Where:
    if ":" in text:
        region, text = text.split(":", 1)
    return Where(*text.split("/", 1), region)


def runs_on(w: Where, kits: set[str]) -> bool:
    """The job can run there: its region has a project, and its GPU runs every kit it needs (the interpreters
    of `pythons` run anywhere)."""
    if w.region != REGION and w.region not in PROJECTS:
        return False
    return w.arch == "sm_89" or not (kits - {"pythons"}) - SM120_KITS


def platforms(home: Path | None, kits: set[str] = frozenset(KITS)) -> list[Where]:
    """Where to ask in turn, the same for the same state: REGION's platforms (no traffic between regions), cheapest
    first, then the other regions' by price; the ones that found no machine lately last. kits: the ones the job
    needs (all of them by default, as a warm GPU)."""
    home_region = [Where(PLATFORM, PRESET)] + ([_where(FALLBACK)] if FALLBACK else [])
    others = sorted((_where(t) for t in ELSEWHERE.split(",") if t.strip()),
                    key=lambda w: (usd_per_hour(w.platform), w.region))
    return _sick_last(home, [w for w in home_region + others if runs_on(w, kits)])


# The scene GPU for several runs at once: three fit in its 96 GB (2026-10-14, 0.51 $ a run against 1.02 to 1.15 $
# with one run per 32 GB machine). For one run alone it costs 17 % more than PLATFORM's 32 GB, which holds one.
SCENE_BIG = config.env("SIM2CHIP_SCENE_BIG", "gpu-l40s-d/1gpu-16vcpu-96gb")


def scene_platforms(home: Path | None, runs: int = 1) -> list[Where]:
    """Where the scene GPU (warm.py) is asked, in turn: SIM2CHIP_SCENE_GPU ("[region:]platform/preset") when set;
    else, for two runs or more, SCENE_BIG first; else as any run. Places that found no machine lately go last."""
    if (where := _env("SIM2CHIP_SCENE_GPU")):
        return [_where(where)]
    if runs < 2 or not SCENE_BIG:
        return platforms(home)
    big = _where(SCENE_BIG)
    return _sick_last(home, [big] + [w for w in platforms(home) if w != big])


def note_no_gpu(home: Path | None, where: Where | str) -> None:
    if home is None:
        return
    f = home / "logs" / "no_gpu.json"
    try:
        sick = json.loads(f.read_text())
    except (OSError, ValueError):
        sick = {}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({**sick, where if isinstance(where, str) else _sick_key(where): time.time()}))
TERMINAL = {"COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED", "DELETED"}


_env = config.env


def code_archive() -> bytes:
    """The sim2chip package as a tar.gz, unpacked by every stage job (no custom image needed)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted((REPO / "sim2chip").rglob("*.py")):
            tar.add(f, arcname=str(f.relative_to(REPO)))
    return buf.getvalue()


def bootstrap(stage: Stage, fp: str) -> str:
    """Shell script that runs one stage inside its container: fetch the code, pull, run, push.

    The code comes from a presigned URL in S2C_CODE_URL; boto3 (for the sync only) goes to its own folder
    so it cannot clash with the runtime's packages.
    """
    rt = RUNTIMES[stage.runtime]
    dirs = f"--job {JOB_DIR} --cache {CACHE_DIR}"
    sync = f"PYTHONPATH=/tmp/s2c/code:/tmp/s2c/deps {rt.python} -m sim2chip.cloud_sync"
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        f"mkdir -p /tmp/s2c/code /tmp/s2c/deps {JOB_DIR}/logs {CACHE_DIR}",
        f"{rt.python} -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "
        '"$S2C_CODE_URL" /tmp/s2c/code.tgz',
        "tar xzf /tmp/s2c/code.tgz -C /tmp/s2c/code",
        f"{rt.python} -m pip install -q --target /tmp/s2c/deps boto3",
        rt.setup,
        "export PYTHONUNBUFFERED=1",
        f"{sync} pull {stage.name} {dirs}",
        "set +e",
        f"PYTHONPATH=/tmp/s2c/code {rt.python} -m sim2chip.run_stage {stage.name} {dirs} --fingerprint {fp} 2>&1 "
        f"| tee {JOB_DIR}/logs/{stage.name}.log",
        "set -e",
        f"{sync} push {stage.name} {dirs}",
    ]
    return "\n".join(l for l in lines if l) + "\n"


def print_timeline(job: Job, log=print, since: float | None = None) -> None:
    """Where the time of the last run went, longest first."""
    f = job.logs / "timeline.jsonl"
    if not f.exists():
        return
    recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if since is not None:
        recs = [r for r in recs if r["t"] >= since]
    if not recs:
        return
    # Steps nest (the job's running state holds every stage): shares are of the wall time, not of the sum.
    total = max(r["t"] for r in recs) - min(r["t"] - r["seconds"] for r in recs) or 1
    try:  # the GPU's use per step, from the cloud job (gpu_usage.py)
        gpu = json.loads((job.logs / "gpu_usage.json").read_text())
        if gpu.get("end", 0) < min(r["t"] for r in recs):  # an earlier run's
            gpu = {}
    except (OSError, ValueError):
        gpu = {}
    log(f"  where the time went ({total / 60:.1f} min in all):")
    for r in sorted(recs, key=lambda r: -r["seconds"]):
        g = gpu.get("steps", {}).get(r["step"])
        use = (f", GPU {g['util']}%" + (f", CPU {g['cpu']}%" if "cpu" in g else "")) if g else ""
        log(f"    {r['seconds'] / 60:5.1f} min  {100 * r['seconds'] / total:3.0f}%  {r['step']} ({r['where']}{use})")
    if gpu.get("all"):
        a = gpu["all"]
        log(f"  GPU in the cloud job: {a['util']}% on average, busy {100 * a['busy']:.0f}% of the time, "
            f"{a['mem_mb'] / 1024:.1f} GB at most")
        if "cpu" in a:
            log(f"  CPU in the cloud job ({gpu.get('cpus') or '?'} cores): {a['cpu']}% on average, full "
                f"{100 * a['cpu_full']:.0f}% of the time, iowait {a['iowait']}%, RAM {a['ram_mb'] / 1024:.1f} GB at most")


def cloud_spec(spec, log=print):
    """The spec without what the cloud cannot run yet, decided before asking for a GPU: 3D models from the
    user's photos need the TRELLIS runtime, which only exists locally (2026-09-27: a run found out inside the
    job, with the GPU already paid). Those classes use the 3D model search instead, and the log says so."""
    from .. import spec as spec_mod

    data = json.loads(spec.to_json())
    for c in data.get("classes") or []:
        a = c.get("assets")
        if not a or not a.get("photos"):
            continue
        a["photos"] = None
        if a.get("source") == "photos":
            a["source"] = "search"
            a["queries"] = a.get("queries") or [c["id"]]
        log(f"WARNING: 3D models from photos are not available on Nebius yet: class {c['id']} uses the "
            f"3D model search ({', '.join(a.get('queries') or a.get('categories') or [])})")
    return spec_mod.from_dict(data)


def check_photos(spec, job: Job) -> None:
    """The user's photo folders must hold photos here, before a GPU is paid for to find out they don't
    (a spec that names a folder no --input brought in)."""
    from ..workspace import IMAGE_EXT

    for k in ("raw", "labeled", "test"):
        rel = getattr(spec.user_data, k)
        if not rel:
            continue
        src = job.resolve_input(rel)
        if not (src.is_dir() and any(f.suffix.lower() in IMAGE_EXT for f in src.rglob("*"))):
            raise FileNotFoundError(f"user_data.{k}: no photos in {src} (bring them in with --input {rel}=PATH)")


def portable_spec(spec, job: Job):
    """The spec with every local file or folder it names (user photos, QA lists) linked into the job's input/
    and named relative to it: input/ is uploaded, and absolute paths of this machine mean nothing on Nebius.
    Paths are rewritten before the pipeline fingerprints its stages, so both sides plan the same run."""
    import hashlib

    from .. import spec as spec_mod
    from ..workspace import link_input

    def portable(path: str | None) -> str | None:
        if not path or not Path(path).is_absolute() or not Path(path).exists():
            return path
        name = f"{Path(path).name}_{hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:8]}"
        link_input(job, name, Path(path))
        return name

    data = json.loads(spec.to_json())
    ud = data.get("user_data") or {}
    for k in ("raw", "labeled", "test"):
        if k in ud:
            ud[k] = portable(ud[k])
    for c in data.get("classes") or []:
        if c.get("assets"):
            c["assets"]["photos"] = portable(c["assets"].get("photos"))
    qa = data.get("qa") or {}
    for k in ("exclude", "class_override", "no_color"):
        if k in qa:
            qa[k] = portable(qa[k])
    return spec_mod.from_dict(data)


def worker_runtime(isaac: bool) -> Runtime:
    """The image of a whole-run job: Isaac Sim's only when a stage renders (it is the big one), else the light
    one (user's choice, 2026-10-04: one job per run, the image picked by what the run uses)."""
    if not (REGISTRY and IMAGE_TAG):
        return RUNTIMES["isaac"]
    if isaac:
        return Runtime(f"{REGISTRY}/eyeforge-isaac:{IMAGE_TAG}", RUNTIMES["isaac"].python, env=RUNTIMES["isaac"].env,
                       kits=True)
    return Runtime(f"{REGISTRY}/eyeforge-light:{IMAGE_TAG}", KIT_PY, kits=True)


def kit_keys(teacher: bool, picodet: bool = False, gsplat: bool = False) -> list[str]:
    """Cache bucket keys of the kits a run unpacks: the Python itself, the worker's, the teachers' if it labels,
    PaddlePaddle's if it trains a PicoDet counter, gsplat's if it renders walk scenes."""
    used = needed_kits(teacher, picodet, gsplat)
    return [f"{ENVS_PREFIX}/{IMAGE_TAG}/{k}.tar.zst" for k in KITS if k in used]


def trains_picodet(spec, todo) -> bool:
    """The run trains a counter with PicoDet (hardware profile counter), not the grid head."""
    from .. import hardware
    from ..tasks import task_of
    return any(st.name == "train" for st in todo) and task_of(spec).detector(hardware.get(spec.hardware)) == "picodet_s"


def needed_kits(teacher: bool, picodet: bool = False, gsplat: bool = False) -> set[str]:
    """The kits a job unpacks (kit_keys), which decide the GPUs it can run on (runs_on)."""
    return {"pythons", "py"} | {k for k, on in (("teacher", teacher), ("picodet", picodet), ("gsplat", gsplat)) if on}


def renders_walks(todo) -> bool:
    """The run renders the walk scenes (scene_render), whose backgrounds need the gsplat kit."""
    return any(st.name == "scene_render" for st in todo)


# Streams one kit from its presigned URL to stdout (into zstd and tar) and writes how many bytes came.
KIT_STREAM = ("import sys, urllib.request\n"
              "n = 0\n"
              "with urllib.request.urlopen(sys.argv[1], timeout=60) as r:\n"
              "    while b := r.read(1 << 20):\n"
              "        sys.stdout.buffer.write(b); n += len(b)\n"
              "open(sys.argv[2], \"w\").write(str(n))")


def kit_lines(var: str = "S2C_ENVS", late_marker: str | None = None) -> list[str]:
    """Shell lines that unpack the kits whose presigned URLs are in $var. The first kit is the bare Python (31 MB),
    fetched first since it streams the others. Those stream from the bucket straight into zstd and tar, all at
    once: download and unpack overlap (on the node each takes about as long as the other, 2026-10-04).
    late_marker: the kits in $var come after start-up, in the background (the bare Python is already there), and
    this file is written when they are unpacked."""
    head = [] if late_marker else [
        f"mkdir -p {ENVS}",
        f'read -ra urls <<< "${var}"',
        "aria2c -q --max-tries 5 --retry-wait 5 -d /tmp/s2c -o kit0.tar.zst \"${urls[0]}\"",
        f"zstd -dcq /tmp/s2c/kit0.tar.zst | tar xf - -C {ENVS}",
        f"KIT_PYTHON=$(ls -d {ENVS}/pythons/cpython-3.11*/bin/python3 | head -1)",
    ]
    first = -1 if late_marker else 0
    tag = "late" if late_marker else "kit"
    loop = [
        "pids=()",
        'for i in "${!urls[@]}"; do',
        f'  [ "$i" = {first} ] && continue',
        "  ( start=$(date +%s.%N)",
        "    for try in 1 2 3; do  # a broken stream starts over: tar overwrites what it had unpacked",
        f'      "$KIT_PYTHON" -c {shlex.quote(KIT_STREAM)} "${{urls[$i]}}" /tmp/s2c/{tag}$i.bytes'
        f" | zstd -T0 -dcq | tar xf - -C {ENVS} && break",
        '      [ "$try" = 3 ] && exit 1',
        f'      echo "{tag} $i: download failed, retrying"; sleep 5',
        "    done",
        (f'    echo "$i $start $(date +%s.%N) $(cat /tmp/s2c/kit$i.bytes)" >> {KIT_TIMES} ) &' if not late_marker
         else "  ) &"),
        "  pids+=($!)",
        "done",
        'for p in "${pids[@]}"; do wait $p; done',
    ]
    if not late_marker:
        return head + loop
    # Written even if a kit failed: a run waiting for it then goes on without it (PicoDet falls back to the grid).
    body = "\n".join(["set +e", f'read -ra urls <<< "${var}"', *loop, f"touch {late_marker}"])
    return [f"( {body}\n) > /tmp/s2c/late_kits.log 2>&1 &"]


def worker_script(job_name: str, plan_args: list[str], teacher: bool = False,
                  runtime: Runtime = RUNTIMES["isaac"], gsplat: bool = False) -> str:
    """Shell script for the worker job: the Pythons (TensorFlow, torch for the teachers only when the run
    labels photos, gsplat's only when it renders walk scenes), the code from a presigned URL, then the whole
    pipeline. With our own images the Pythons are kits unpacked from the presigned URLs in S2C_ENVS, all at
    once; in the stock Isaac Sim image it makes them with uv (a couple of minutes)."""
    from ..gpu_usage import SAMPLER
    isaac = runtime.python == RUNTIMES["isaac"].python
    # A download can meet a passing 503 (2026-10-03, after the GPU was paid): retry before giving up.
    fetch = (f"{KIT_PY if runtime.kits else runtime.python} -c 'import sys, time, urllib.request\n"
             "for i in range(5):\n"
             "    try: urllib.request.urlretrieve(sys.argv[1], sys.argv[2]); break\n"
             "    except Exception as e:\n"
             "        if i == 4: raise\n"
             "        print(\"download failed, retrying:\", e, flush=True); time.sleep(5 * 2 ** i)'")
    py = "/tmp/s2c/py/bin/python"
    install = [
        f"{fetch} {UV_URL} /tmp/s2c/uv.tgz",
        "tar xzf /tmp/s2c/uv.tgz -C /tmp/s2c/bin --strip-components 1",
        "export UV_CACHE_DIR=/tmp/s2c/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/s2c/pythons",
        "/tmp/s2c/bin/uv venv -q --python 3.11 /tmp/s2c/py",
        f"/tmp/s2c/bin/uv pip install -q --python {py} " + " ".join(shlex.quote(d) for d in WORKER_DEPS),
        f"/tmp/s2c/bin/uv pip install -q --no-deps --python {py} " + " ".join(shlex.quote(d) for d in WORKER_NODEPS),
        f"PY={py} TEACHER_PY=",
    ]
    if teacher:
        install += ["/tmp/s2c/bin/uv venv -q --python 3.11 /tmp/s2c/sam3",
                    "/tmp/s2c/bin/uv pip install -q --python /tmp/s2c/sam3/bin/python "
                    + " ".join(shlex.quote(d) for d in TEACHER_DEPS),
                    "TEACHER_PY=/tmp/s2c/sam3/bin/python"]
    if gsplat:
        install += ["/tmp/s2c/bin/uv venv -q --python 3.10 /tmp/s2c/gsplat",
                    "/tmp/s2c/bin/uv pip install -q --python /tmp/s2c/gsplat/bin/python "
                    + " ".join(shlex.quote(d) for d in GSPLAT_DEPS),
                    "export SIM2CHIP_GSPLAT_PYTHON=/tmp/s2c/gsplat/bin/python"]
    unpack = kit_lines() + [
        f"PY={KIT_PY} TEACHER_PY={KIT_TEACHER}",
        # Without the picodet kit these paths do not exist and the train stage counts with the grid head.
        f"export SIM2CHIP_PICODET_PYTHON={KIT_PICODET} SIM2CHIP_PADDLEDET={KIT_PADDLEDET}",
        f"export SIM2CHIP_GSPLAT_PYTHON={KIT_GSPLAT}",  # scene_render says so if the kit did not come
    ]
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "export S2C_T0=$(date +%s)",
        "mkdir -p /tmp/s2c/code /tmp/s2c/bin /tmp/s2c/home",
        SAMPLER,
        *(unpack if runtime.kits else install),
        f'{fetch} "$S2C_CODE_URL" /tmp/s2c/code.tgz',
        "tar xzf /tmp/s2c/code.tgz -C /tmp/s2c/code",
        # The teachers' weights come from the cache bucket (teacher.weights), with no Hugging Face token.
        "export SIM2CHIP_SAM3_PYTHON=$TEACHER_PY HF_HUB_CACHE=/tmp/s2c/hf SIM2CHIP_WEIGHTS_DIR=/tmp/s2c/weights",
        "export PYTHONUNBUFFERED=1 SIM2CHIP_PYTHON=$PY PYTHONPATH=/tmp/s2c/code SIM2CHIP_RUNS_ON=nebius "
        "S2C_SETUP_S=$(( $(date +%s) - S2C_T0 ))" + (f" SIM2CHIP_ISAAC_PYTHON={runtime.python}" if isaac else ""),
        f"$PY -m sim2chip.cloud_worker --home /tmp/s2c/home --job {shlex.quote(job_name)} "
        + " ".join(shlex.quote(a) for a in plan_args),
    ]
    return "\n".join(lines) + "\n"


def warm_script(isaac: bool = True) -> str:
    """Shell script of the warm GPU (warm.py): the kits (the worker's Python and the teachers'), the PicoDet kit (and
    on the scene GPU gsplat's) in the background, its own code, then its loop. Tasks bring their run's code.
    isaac: it runs in Isaac Sim's image, so whole runs that render go to it too (user, 2026-10-09: one GPU from the first labels to the model)."""
    fetch = (f"{KIT_PY} -c 'import sys, time, urllib.request\n"
             "for i in range(5):\n"
             "    try: urllib.request.urlretrieve(sys.argv[1], sys.argv[2]); break\n"
             "    except Exception as e:\n"
             "        if i == 4: raise\n"
             "        print(\"download failed, retrying:\", e, flush=True); time.sleep(5 * 2 ** i)'")
    from ..gpu_usage import SAMPLER
    from ..warm import LATE_READY
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "S2C_T0=$(date +%s)",
        "mkdir -p /tmp/s2c/code /tmp/s2c/bin /tmp/s2c/home",
        SAMPLER,
        *kit_lines(),
        *kit_lines("S2C_ENVS_LATE", LATE_READY),
        f'{fetch} "$S2C_CODE_URL" /tmp/s2c/code.tgz',
        "tar xzf /tmp/s2c/code.tgz -C /tmp/s2c/code",
        f"export SIM2CHIP_PICODET_PYTHON={KIT_PICODET} SIM2CHIP_PADDLEDET={KIT_PADDLEDET}",
        *([f"export SIM2CHIP_GSPLAT_PYTHON={KIT_GSPLAT}"] if isaac else []),
        f"export SIM2CHIP_SAM3_PYTHON={KIT_TEACHER} HF_HUB_CACHE=/tmp/s2c/hf SIM2CHIP_WEIGHTS_DIR=/tmp/s2c/weights",
        # Tasks run side by side: TensorFlow must not take the whole GPU at its first use.
        "export TF_FORCE_GPU_ALLOW_GROWTH=true",
        f"export PYTHONUNBUFFERED=1 SIM2CHIP_PYTHON={KIT_PY} PYTHONPATH=/tmp/s2c/code SIM2CHIP_RUNS_ON=nebius"
        + (f" SIM2CHIP_ISAAC_PYTHON={RUNTIMES['isaac'].python}" if isaac else ""),
        f'{KIT_PY} -m sim2chip.warm_worker --id "$S2C_WARM_ID"',
    ]
    return "\n".join(lines) + "\n"


class NebiusCLI:
    """Adapter: buckets and AI jobs through the `nebius` command line (patterns: docs/architecture.md)."""
    def __init__(self, exe: str | None = None, project: str | None = None, dry_run: bool = False):
        self.exe = exe or _env("SIM2CHIP_NEBIUS_CLI") or shutil.which("nebius") or str(
            Path.home() / ".nebius/bin/nebius")
        self.project = project or _env("SIM2CHIP_NEBIUS_PROJECT")
        self.dry_run = dry_run
        self._secret_ids: dict[str, str] = {}

    def __call__(self, *args: str, check: bool = True) -> str:
        cmd = [self.exe, *args]
        if self.dry_run:
            print("  $ " + shlex.join(cmd), flush=True)
            return ""
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"{shlex.join(cmd[:4])}...: {r.stderr.strip()[-500:]}")
        return r.stdout

    def _parent(self, region: str = REGION) -> list[str]:
        project = self.project if region == REGION else PROJECTS[region]
        return ["--parent-id", project] if project else []

    def _registry(self, region: str) -> list[str]:
        """REGION's registry asks a job elsewhere for credentials (2026-10-21): a short-lived IAM token."""
        if region == REGION:
            return []
        token = "<token>" if self.dry_run else self("iam", "get-access-token").strip()
        return ["--registry-username", "iam", "--registry-password", token]

    def ensure_bucket(self, name: str) -> None:
        if self.dry_run:
            self("storage", "bucket", "create", "--name", name, *self._parent())
            return
        if not self("storage", "bucket", "get-by-name", "--name", name, *self._parent(), check=False).strip():
            try:
                self("storage", "bucket", "create", "--name", name, *self._parent())
            except RuntimeError as e:  # the lookup can fail for a moment (seen 2026-09-27) on a bucket we own
                if "BucketAlreadyOwnedByYou" not in str(e):
                    raise

    def bucket_id(self, name: str) -> str | None:
        out = self("storage", "bucket", "get-by-name", "--name", name, *self._parent(), "--format", "json",
                   check=False).strip()
        return json.loads(out)["metadata"]["id"] if out else None

    def delete_bucket(self, bucket_id: str) -> None:
        """Delete an empty bucket."""
        self("storage", "bucket", "delete", "--id", bucket_id)

    def create_job(self, name: str, runtime: Runtime, env: dict[str, str], script: Path, timeout_h: int,
                   env_secrets: list[str], where: Where = Where(PLATFORM, PRESET)) -> str:
        where = Where(*where)
        args = ["ai", "job", "create", "--name", name, "--image", runtime.image, "--platform", where.platform,
                "--preset", where.preset, "--on-demand", "--timeout", f"{timeout_h}h", *self._parent(where.region),
                *self._registry(where.region),
                "--inject-file", f"{script}:/opt/s2c_stage.sh", "--container-command", "bash",
                "--args", "/opt/s2c_stage.sh", "--async"]
        for k, v in {**runtime.env, **env}.items():
            args += ["--env", f"{k}={v}"]
        for s in env_secrets:
            key, sel = s.split("=", 1)
            args += ["--env-secret", s if where.region == REGION else f"{key}={self._secret_ids.setdefault(sel, self.secret_id(sel))}"]
        out = self(*args)
        if self.dry_run:
            return "<job id>"
        for line in out.splitlines():
            if "Job ID:" in line:
                return line.split("Job ID:", 1)[1].strip()
        raise RuntimeError(f"no job id in: {out[-300:]}")

    def secret_id(self, name: str) -> str:
        """A MysteryBox secret's id, which a job in another region's project names it by (its name is looked up in
        the job's own project)."""
        if self.dry_run or name.startswith("mbsec-"):
            return name
        for line in self("mysterybox", "secret", "get-by-name", "--name", name, *self._parent()).splitlines():
            if line.strip().startswith("id:"):
                return line.split(":", 1)[1].strip()
        raise RuntimeError(f"no MysteryBox secret {name}")

    def job_state(self, job_id: str) -> str:
        for line in self("ai", "job", "get", job_id).splitlines():
            if line.strip().startswith("State:"):
                return line.split(":", 1)[1].strip()
        return "UNKNOWN"

    def cancel(self, job_id: str) -> None:
        self("ai", "job", "cancel", job_id, check=False)

    def job_logs(self, job_id: str) -> str:
        return self("ai", "job", "logs", job_id, check=False)


class NebiusBackend:
    name = "nebius"

    def __init__(self, verbose: bool = False, dry_run: bool = False, max_usd: float | None = None,
                 cli: NebiusCLI | None = None, store=None, poll_s: float = 20.0):
        self.verbose, self.dry_run, self.poll_s = verbose, dry_run, poll_s
        self.max_usd = max_usd if max_usd is not None else config.env_float("SIM2CHIP_MAX_USD", 5)
        self.cli = cli or NebiusCLI(dry_run=dry_run)
        self._store = store
        self.prefix = _env("SIM2CHIP_BUCKET_PREFIX", "sim2chip")
        self.spent_usd = 0.0
        self._ready: set[str] = set()

    @property
    def store(self):
        if self._store is None:
            self._store = S3Store(_env("AWS_ACCESS_KEY_ID"), _env("AWS_SECRET_ACCESS_KEY"))
        return self._store

    def buckets(self, job: Job) -> tuple[str, str]:
        """The job's own bucket and the shared cache. A job name can come back (another server, a wiped
        home), so a new job adds a random id kept in its folder; jobs that already ran in the cloud keep
        their old name, and their files in it."""
        id_file = job.root / "cloud_id"
        if not id_file.exists() and not (job.logs / "cloud_usage.jsonl").exists():
            id_file.parent.mkdir(parents=True, exist_ok=True)
            id_file.write_text(token_hex(4))
        cloud_id = id_file.read_text().strip() if id_file.exists() else ""
        suffix = f"-{cloud_id}" if cloud_id else ""
        base = f"{self.prefix}-{job.name}".lower().replace("_", "-")[:63 - len(suffix)]
        return base + suffix, f"{self.prefix}-cache"

    def delete_storage(self, job: Job) -> bool:
        """Delete the job's bucket and everything in it (the user's photos travel there). False when the job
        never ran in the cloud."""
        if not (job.root / "cloud_id").exists() and not (job.logs / "cloud_usage.jsonl").exists():
            return False
        bucket, _ = self.buckets(job)
        bucket_id = self.cli.bucket_id(bucket)
        if bucket_id is None:
            return False
        for key in self.store.listing(bucket):
            self.store.delete(bucket, key)
        self.cli.delete_bucket(bucket_id)
        return True

    def _kit_urls(self, cache_bucket: str, teacher: bool, picodet: bool = False, gsplat: bool = False) -> str:
        """Presigned URLs of the run's kits, checked before a GPU is paid for."""
        keys = kit_keys(teacher, picodet, gsplat)
        if self.dry_run:
            return " ".join(f"<presigned {k}>" for k in keys)
        have = self.store.listing(cache_bucket, f"{ENVS_PREFIX}/{IMAGE_TAG}/")
        if missing := [k for k in keys if k not in have]:
            raise RuntimeError(f"kits missing in s3://{cache_bucket}: {missing}; run images/build.sh kits")
        return " ".join(self.store.url(cache_bucket, k) for k in keys)

    def _upload_job(self, job: Job, cache: Cache, bucket: str, cache_bucket: str, spec_json: str | None = None) -> str:
        """Buckets, code, spec and input (once per run). Returns the code's presigned URL. spec_json: the spec to
        send when the job has no spec.json of its own yet (a prefetch during calibration)."""
        if self.dry_run:
            print(f"  upload code, spec.json and input/ to s3://{bucket}", flush=True)
            return "<presigned url>"
        if bucket not in self._ready:
            for b in (bucket, cache_bucket):
                self.cli.ensure_bucket(b)
            self.store.put(bucket, "_code/sim2chip.tar.gz", code_archive())
            if job.input.exists():
                self._scenes_through_cache(job, bucket, cache_bucket)
                n = self.store.upload_dir(bucket, job.input, "input")
                print(f"  uploaded {n} input files to s3://{bucket}", flush=True)
            # Input the job no longer has must not reach the worker (it would label and train on it).
            # Walked as upload_dir walks it: input folders are symlinks, which rglob does not enter (it found
            # nothing there and deleted every photo just uploaded, 2026-09-27).
            local = {(Path(d) / n).relative_to(job.root).as_posix()
                     for d, _, names in os.walk(job.input, followlinks=True) for n in names}
            for key in self.store.listing(bucket, "input/"):
                if key not in local:
                    self.store.delete(bucket, key)
            self.store.upload_dir(cache_bucket, cache.db, "db")  # share what this machine learned about assets
            self._ready.add(bucket)
        # The user's corrections change between runs of one job (calibration rounds, fixes after training),
        # often at the same size (a mark moved): sent every run, or the worker labels with old ones.
        for f in sorted(job.input.glob("*.json")) if job.input.exists() else []:
            self.store.put(bucket, f"input/{f.name}", f.read_bytes())
        self.store.put(bucket, "spec.json", spec_json.encode() if spec_json is not None else job.spec_path.read_bytes())
        return self.store.url(bucket, "_code/sim2chip.tar.gz")

    def _scenes_through_cache(self, job: Job, bucket: str, cache_bucket: str) -> None:
        """The job's walk scenes (input/scenes/<id>/, library.export_scenes) go up once, to the cache bucket's
        scenes/<id>/, and are copied from there into the job's bucket inside Object Storage: a splat is about 1 GB
        and the same public scene goes with many runs. A scene's files never change once it is ready. The job's
        own upload then finds them there at their size and skips them."""
        root = job.input / "scenes"
        if not root.is_dir():
            return
        have = self.store.listing(bucket, "input/scenes/")
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            n = self.store.upload_dir(cache_bucket, d, f"scenes/{d.name}")
            for key, size in self.store.listing(cache_bucket, f"scenes/{d.name}/").items():
                dst = f"input/{key}"
                if have.get(dst) != size:
                    self.store.copy(cache_bucket, key, bucket, dst)
            print(f"  scene {d.name}: {n} files uploaded to s3://{cache_bucket}/scenes/", flush=True)

    def prepare(self, pipe):
        """Local folders named in the spec travel as job input, and what the cloud cannot do yet is swapped
        for what it can (3D from photos). The stages are built anew from the new spec."""
        from ..pipeline import Pipeline
        return Pipeline(portable_spec(cloud_spec(pipe.spec), pipe.job), pipe.job, pipe.cache)

    def run(self, pipe, start: str | None = None, only: str | None = None, force: bool = False) -> None:
        if self.worker_mode:  # the whole run in one cloud job
            self.run_pipeline(pipe, start=start, only=only, force=force)
        elif self.dry_run:  # stage mode: print each stage's job
            pipe.job.spec_path.write_text(pipe.spec.to_json())
            for p in pipe.plan(start, only, force):
                if p.action == "run":
                    print(f"> {p.stage.name} ({p.stage.powered_by(pipe.spec, self.name)})")
                    self.run_stage(p.stage, pipe.job, pipe.cache, p.fingerprint)
        else:
            pipe.run(self, start=start, only=only, force=force)

    @property
    def worker_mode(self) -> bool:
        return _env("SIM2CHIP_NEBIUS_MODE", "worker") != "stage"

    def run_pipeline(self, pipe, start: str | None = None, only: str | None = None, force: bool = False,
                     log=print) -> None:
        """Worker mode: the whole run in one job. Progress lines and done markers come back while it runs."""

        plans = pipe.plan(start, only, force, retry=True)
        todo = [p.stage for p in plans if p.action == "run"]
        for p in plans:
            if p.action == "skip" and only is None:
                log(f"= {p.stage.name}: up to date")
        if not todo:
            return
        job, cache = pipe.job, pipe.cache
        # The teachers' Python also runs the likeness filter of the synthetic frames (SigLIP 2).
        teacher = any(st.name in ("label", "render") for st in todo)
        runtime = worker_runtime(any(st.runtime == "isaac" or st.gpu == "rtx" for st in todo))
        if self.spent_usd + WORKER_HOURS * USD_PER_HOUR > self.max_usd:
            raise RuntimeError(f"a cloud run can cost up to ${WORKER_HOURS * USD_PER_HOUR:.2f} (timeout "
                               f"{WORKER_HOURS} h): over the cap of ${self.max_usd:.2f} (SIM2CHIP_MAX_USD)")
        if any(st.name in ("label", "ingest") for st in todo):
            check_photos(pipe.spec, job)
        job.spec_path.write_text(pipe.spec.to_json())
        job_bucket, cache_bucket = self.buckets(job)
        t_up = time.time()
        code_url = self._upload_job(job, cache, job_bucket, cache_bucket)
        if not self.dry_run:
            self.store.upload_dir(cache_bucket, cache.root / "labels", "labels")  # teacher answers known here
            for st in todo:  # stale markers must not pass as success
                self.store.delete(job_bucket, f"{st.name}/{DONE}")
                job.clear_done(st.name)
            for old in ("logs/cloud_pipeline.log", "logs/timeline.jsonl"):  # nor an old run's log as progress
                self.store.delete(job_bucket, old)
            (job.logs / "cloud_pipeline.log").unlink(missing_ok=True)
        plan_args = [a for flag, v in (("--from", start), ("--only", only)) if v for a in (flag, v)]
        plan_args += ["--force"] if force else []
        script = job.logs / "worker.nebius.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        gsplat = renders_walks(todo)
        script.write_text(worker_script(job.name, plan_args, teacher, runtime, gsplat))
        secrets = self._secrets(any(st.calls_llm(pipe.spec) for st in todo))
        env = {"S2C_JOB_BUCKET": job_bucket, "S2C_CACHE_BUCKET": cache_bucket, "S2C_CODE_URL": code_url,
               "SIM2CHIP_REGION": REGION}
        if runtime.kits:
            env["S2C_ENVS"] = self._kit_urls(cache_bucket, teacher, trains_picodet(pipe.spec, todo), gsplat)
        if os.environ.get("SIM2CHIP_LABEL_SAMPLE"):  # calibration: the teacher marks only a few photos
            env["SIM2CHIP_LABEL_SAMPLE"] = os.environ["SIM2CHIP_LABEL_SAMPLE"]
        name = f"s2c-{job.name}-run".lower().replace("_", "-")[:63]
        job.timeline("upload job", time.time() - t_up, "local")
        picodet = trains_picodet(pipe.spec, todo)
        if not self.dry_run:
            w = self.warm_for(cache, runtime)
            if w is not None:
                extra = {k: env[k] for k in ("SIM2CHIP_LABEL_SAMPLE",) if k in env}
                state, hours = self._run_warm(w, job, job_bucket, cache_bucket, code_url, plan_args, extra, picodet,
                                              [st.name for st in todo], log, gsplat)
                if state is not None:
                    return self._finish(job, cache, job_bucket, cache_bucket, state, hours, todo, log, t_up)
        home = job.root.parents[1]
        options = platforms(home, needed_kits(teacher, picodet, gsplat))
        if not options:
            raise NoGPU("no GPU place runs this job's kits (SIM2CHIP_NEBIUS_*, SIM2CHIP_SM120_KITS)")
        for i, where in enumerate(options):
            t0 = time.time()
            job_id = self.cli.create_job(name, runtime, env, script, WORKER_HOURS, secrets, where)
            if self.dry_run:
                return
            log(f"  nebius job {job_id} ({where.region}, {where.platform}, {where.preset}, "
                f"{runtime.image.rsplit('/', 1)[-1]}), whole pipeline in one job")
            # While this file exists a cloud job may be running and billing: the web reads it (and can cancel the
            # job) even when the process that launched it is gone.
            marker = job.logs / CLOUD_JOB
            marker.write_text(json.dumps({"id": job_id, "started": time.time(), "hours": WORKER_HOURS + 1}))
            try:  # if this fails the marker stays, to find the job
                state = self._follow(job_id, job, job_bucket, log, [st.name for st in todo])
                break
            except NoGPU as e:
                marker.unlink(missing_ok=True)
                note_no_gpu(home, where)
                if i == len(options) - 1:
                    raise
                log(f"  {e}: trying {options[i + 1].platform} in {options[i + 1].region}")
        marker.unlink(missing_ok=True)
        hours = (time.time() - t0) / 3600
        self.spent_usd += hours * usd_per_hour(where.platform)
        self._record(job, "run", job_id, state, hours, where=where)
        (job.logs / "worker.job.log").write_text(self.cli.job_logs(job_id))
        self._finish(job, cache, job_bucket, cache_bucket, state, hours, todo, log, t_up)

    def _finish(self, job: Job, cache: Cache, job_bucket: str, cache_bucket: str, state: str, hours: float,
                todo: list, log, t_up: float) -> None:
        """After the run's job or warm task: results back, where the time went, and stages that did not finish."""
        from ..pipeline import StageFailed
        t1 = time.time()
        self._fetch_results(job, cache, job_bucket, cache_bucket)
        job.timeline("download results", time.time() - t1, "local")
        print_timeline(job, log, since=t_up)
        log(f"  {state} after {hours * 60:.1f} min, about ${hours * USD_PER_HOUR:.2f}")
        missing = [st.name for st in todo if not job.done(st.name)]
        if missing:
            raise StageFailed(f"cloud run {state}: stages not done {missing}, see {job.logs / 'worker.job.log'} "
                              f"and {job.logs / 'cloud_pipeline.log'}")

    # Warm GPU (warm.py): one job kept up between runs; a run goes to it as a task when it is up.

    def start_warm(self, home: Path, kind: str = "light", runs: int = 1) -> dict:
        """Asks Nebius for the warm GPU of that kind (warm.KINDS) and notes it in its worker file. Its kits are
        checked first."""
        from .. import warm
        cache_bucket = f"{self.prefix}-cache"
        self.cli.ensure_bucket(cache_bucket)
        # pythons, py, teacher; late: picodet, and gsplat's on the scene GPU
        kits = self._kit_urls(cache_bucket, teacher=True, picodet=True, gsplat=kind == "isaac").split()
        wid = token_hex(6)
        self.store.put(cache_bucket, warm.key(wid, "code.tgz"), code_archive())
        script = home / "warm" / ("warm.nebius.sh" if kind == "light" else f"{kind}.nebius.sh")
        script.parent.mkdir(parents=True, exist_ok=True)
        # The light one in the light image: Isaac Sim's starts 2.2 min slower at Nebius (226 s against 92 s,
        # 2026-10-10). The scene GPU in Isaac Sim's, for the runs that render.
        runtime = worker_runtime(kind == "isaac")
        isaac = runtime.python == RUNTIMES["isaac"].python
        script.write_text(warm_script(isaac))
        where = scene_platforms(home, runs)[0] if kind == "isaac" else \
            platforms(home, needed_kits(True, True, False))[0]
        hours = warm.HOURS if kind == "light" else WORKER_HOURS + 1  # a run that renders takes up to WORKER_HOURS
        parallel = warm.parallel_for(kind, where[1])
        env = {"S2C_CACHE_BUCKET": cache_bucket, "SIM2CHIP_REGION": REGION, "S2C_WARM_ID": wid,
               "S2C_CODE_URL": self.store.url(cache_bucket, warm.key(wid, "code.tgz"), hours=warm.HOURS + 1),
               "S2C_ENVS": " ".join(kits[:3]), "S2C_ENVS_LATE": " ".join(kits[3:]), "S2C_WARM_PARALLEL": str(parallel),
               "S2C_WARM_HOURS": str(hours)}
        # Rounds may put phrases in English; runs on the scene GPU call the vision models.
        secrets = self._secrets(kind == "isaac" or bool(_env("SIM2CHIP_NEBIUS_SECRET")))
        job_id = self.cli.create_job(f"s2c-warm-{wid}", runtime, env, script, hours, secrets, where)
        w = {"wid": wid, "job_id": job_id, "started": time.time(), "image": runtime.image, "platform": where[0],
             "preset": where[1], "region": Where(*where).region, "kind": kind, "parallel": parallel}
        warm.worker_file(home, kind).write_text(json.dumps(w))
        return w

    def warm_for(self, cache: Cache, runtime: Runtime) -> dict | None:
        """A warm GPU of the server whose home holds this cache that can take this run: our own images, Isaac
        Sim's only if the GPU runs in it (older ones ran in the light image, noted without "image"), its job not
        ended, not stopping. A run that renders looks at the scene GPU first, others at the light one."""
        from .. import warm
        if not runtime.kits:
            return None
        renders = runtime.image != worker_runtime(False).image
        for kind in ("isaac", "light") if renders else ("light", "isaac"):
            w = warm.current(cache.root.parent, kind)
            if w is None or runtime.image not in (worker_runtime(False).image, w.get("image")):
                continue
            try:
                bucket = f"{self.prefix}-cache"
                if self.cli.job_state(w["job_id"]) not in TERMINAL and \
                        warm.usable(self.store, bucket, w["wid"], w.get("started")) and \
                        (kind != "isaac" or warm.room(self.store, bucket, w) > 0):
                    return w
            except Exception as e:  # the run starts its own GPU instead
                print(f"  warm GPU not checked ({type(e).__name__}: {e})", flush=True)
        return None

    def _warm_task(self, w: dict, job: Job, task: dict, follow) -> tuple[str, float | None]:
        """Sends a task to the warm GPU and waits for it with `follow(state_of)`. Returns its final state and when
        it started (None: the GPU never took it, and it was withdrawn)."""
        from .. import warm
        cache_bucket = f"{self.prefix}-cache"
        submitted, started, checked = time.time(), [None], [0.0, False]

        def state_of() -> str:
            if time.time() - checked[0] > 30:  # the job's own state, now and then: a CLI call takes a second
                checked[0] = time.time()
                try:
                    checked[1] = self.cli.job_state(w["job_id"]) in TERMINAL
                except RuntimeError:
                    pass
            st = warm.task_state(self.store, cache_bucket, w["wid"], task["id"], checked[1], submitted)
            if st != "QUEUED" and started[0] is None and st != "LOST":
                started[0] = time.time()
            return st
        warm.note(job.root.parents[1], "task", job.name, kind=task.get("kind"))
        warm.submit(self.store, cache_bucket, w["wid"], task)
        state = follow(state_of)
        if state == "LOST" and started[0] is None:
            warm.withdraw(self.store, cache_bucket, w["wid"], task["id"])
        return state, started[0]

    def _run_warm(self, w: dict, job: Job, job_bucket: str, cache_bucket: str, code_url: str, plan_args: list[str],
                  env: dict, picodet: bool, stages: list[str], log, gsplat: bool = False) -> tuple[str | None, float]:
        """The run as a task of the warm GPU. (None, 0) when the GPU did not take it: the run starts its own job."""
        from .. import warm
        tid = token_hex(6)
        task = {"id": tid, "kind": "run", "job": job.name, "bucket": job_bucket, "code_url": code_url,
                "args": plan_args, "env": env, "picodet": picodet, "gsplat": gsplat, "log": "worker.job.log"}
        log(f"  warm GPU {w['job_id']} ({w.get('platform', PLATFORM)}): no start-up, task {tid}")
        marker = job.logs / CLOUD_JOB
        marker.write_text(json.dumps({"id": w["job_id"], "warm": w["wid"], "task": tid, "started": time.time(),
                                      "hours": WORKER_HOURS + 1}))
        state, started = self._warm_task(w, job, task, lambda state_of: self._follow(
            w["job_id"], job, job_bucket, log, stages, state_of=state_of, poll_s=min(self.poll_s, warm.POLL_S),
            label="warm GPU"))
        marker.unlink(missing_ok=True)
        if started is None:
            log("  the warm GPU did not take the run: it gets a GPU of its own")
            if w.get("kind") == "isaac":
                self._drop_idle_scene_gpu(w, job.root.parents[1], log)
            return None, 0.0
        hours = (time.time() - started) / 3600
        self.spent_usd += hours * USD_PER_HOUR
        self._record(job, "run", w["job_id"], state, hours, warm=True)
        return state, hours

    def _drop_idle_scene_gpu(self, w: dict, home: Path, log) -> None:
        """The scene GPU was asked for this run (job_service.start); with the run gone elsewhere and no other task
        on it, it would only bill its start-up and keep-alive (2026-10-27): cancel it."""
        from .. import warm
        cache_bucket = f"{self.prefix}-cache"
        try:
            if any(k.startswith(warm.key(w["wid"], "tasks") + "/")
                   for k in self.store.listing(cache_bucket, warm.key(w["wid"]) + "/")):
                return
            if (warm.heartbeat(self.store, cache_bucket, w["wid"]) or {}).get("running"):
                return
            self.cli.cancel(w["job_id"])
            log(f"  scene GPU {w['job_id']}: no other task, cancelled")
            warm.WarmGPU(home, backend=self, log=log, kind="isaac").record_end(w, "CANCELLED")
        except Exception as e:  # it stops by itself after its keep-alive
            log(f"  scene GPU {w['job_id']}: not cancelled ({type(e).__name__}: {e})")

    def prefetch_warm(self, job: Job, cache: Cache, spec, sample: int, log=print) -> bool:
        """Count calibration: the warm GPU fills the teacher's answers for every photo of the job (Label.prefetch)
        and they come back to this machine's cache, so the next rounds need no GPU. False without a warm GPU or
        if it failed."""
        from .. import warm
        w = self.warm_for(cache, worker_runtime(False))
        if w is None:
            return False
        cloud = portable_spec(cloud_spec(spec), job)
        job_bucket, cache_bucket = self.buckets(job)
        code_url = self._upload_job(job, cache, job_bucket, cache_bucket, spec_json=cloud.to_json())
        verdicts = job.stage_dir("label") / "verdicts.json"
        if verdicts.exists():  # which photos the user saw: the prefetch starts with the next rounds' ones
            self.store.put(job_bucket, "label/verdicts.json", verdicts.read_bytes())
        task = {"id": token_hex(6), "kind": "prefetch", "job": job.name, "bucket": job_bucket, "code_url": code_url,
                "args": ["--prefetch"], "env": {"SIM2CHIP_LABEL_SAMPLE": str(sample)}, "picodet": False,
                "log": "prefetch.log"}

        def follow(state_of) -> str:
            deadline = time.time() + 3600
            while time.time() < deadline:
                st = state_of()
                if st in ("COMPLETED", "FAILED", "LOST"):
                    return st
                time.sleep(warm.POLL_S if self.poll_s else 0)
            return "LOST"
        state, _ = self._warm_task(w, job, task, follow)
        log(f"prefetch {job.name}: {state}")
        if state != "COMPLETED":
            return False
        self._fetch_labels(job, cache, cache_bucket)
        return True

    def _follow(self, job_id: str, job: Job, bucket: str, log, stages: list[str] | None = None,
                state_of=None, poll_s: float | None = None, label: str = "nebius job") -> str:
        """Waits for the job, echoing new lines of the cloud pipeline log (the web reads them from the local
        log) and bringing done markers back as stages finish. state_of: the state of what runs (a warm GPU's
        task) instead of the job's; it may also end LOST."""
        seen, last, since = 0, None, time.time()
        # The job times out after WORKER_HOURS of running; the extra hour covers the queue and the image pull.
        deadline = since + (WORKER_HOURS + 1) * 3600
        misses, store_error = 0, None
        while True:
            try:
                state = state_of() if state_of else self.cli.job_state(job_id)
            except RuntimeError as e:  # a transient CLI failure must not orphan a job that is still billing
                log(f"  could not read the job state: {e}")
                state = "UNKNOWN"
            misses = misses + 1 if state == "UNKNOWN" else 0
            if misses >= 10 or time.time() > deadline:
                raise RuntimeError(f"lost track of Nebius job {job_id} (state {state}): check it with "
                                   f"'nebius ai job get {job_id}' and cancel it if it still runs")
            if state_of is None and state == "PROVISIONING" and last == "PROVISIONING" \
                    and time.time() - since > PROVISION_S:
                self.cli.cancel(job_id)
                raise NoGPU(f"Nebius found no free GPU in {PROVISION_S / 60:.0f} min (job {job_id}, cancelled): "
                            "try again in a while")
            if state != last and state != "UNKNOWN":
                now = time.time()
                if last is not None:  # how long the job stayed in each state: queue, image pull, run
                    job.timeline(f"{label}: {last.lower()}", now - since, "nebius")
                log(f"  state {state}")
                last, since = state, now
            try:
                self.store.download(bucket, "logs", job.logs, ("cloud_pipeline.log",))
                # Only the markers of the stages that run: listing the bucket means thousands of render frames.
                keys = [f"{st}/{DONE}" for st in stages] if stages is not None else \
                    [k for k in self.store.listing(bucket) if k.endswith("/" + DONE)]
                for key in keys:
                    try:
                        (job.root / key).parent.mkdir(parents=True, exist_ok=True)  # the stage's folder is new here
                        self.store.s3.download_file(bucket, key, str(job.root / key))
                    except Exception as e:  # not written yet
                        if "404" not in str(e) and "NoSuchKey" not in str(e) and "Not Found" not in str(e):
                            raise
            except Exception as e:  # nothing uploaded yet, or a transient error; say it once
                if store_error is None and "NoSuchKey" not in str(e) and "404" not in str(e):
                    log(f"  could not read the job bucket yet: {e}")
                store_error = e
            cloud_log = job.logs / "cloud_pipeline.log"
            lines = cloud_log.read_text(errors="replace").splitlines() if cloud_log.exists() else []
            if len(lines) < seen:  # a new log replaced an older one
                seen = 0
            for line in lines[seen:]:
                if line.startswith(("> ", "= ", "  ", "FAILED", "model:", "real test")):
                    log(line)
            seen = max(seen, len(lines))
            if state in TERMINAL or state == "LOST":
                return state
            time.sleep(self.poll_s if poll_s is None else poll_s)

    def _fetch_results(self, job: Job, cache: Cache, bucket: str, cache_bucket: str) -> None:
        import fnmatch
        import random

        keys = list(self.store.listing(bucket))
        want = [k for k in keys if any(fnmatch.fnmatch(k, p) for p in WORKER_RESULTS) and k != "logs/timeline.jsonl"
                and (not k.startswith("render/") or k in ("render/labels.json", f"render/{DONE}"))]  # not 2 json per frame
        frames = sorted(k for k in keys if k.startswith("render/rgb_"))
        want += random.Random(0).sample(frames, min(RENDER_SAMPLE, len(frames)))  # for review
        for key in want:
            out = job.root / key
            out.parent.mkdir(parents=True, exist_ok=True)
            self.store.s3.download_file(bucket, key, str(out))
        if "logs/timeline.jsonl" in keys:  # the cloud's steps go after this machine's
            cloud = self.store.s3.get_object(Bucket=bucket, Key="logs/timeline.jsonl")["Body"].read().decode()
            with open(job.logs / "timeline.jsonl", "a") as f:
                f.write(cloud)
        self.store.download(cache_bucket, "db", cache.root / "db", ("*",), overwrite=False)  # for later runs here
        self._fetch_labels(job, cache, cache_bucket)

    def _fetch_labels(self, job: Job, cache: Cache, cache_bucket: str) -> int:
        """The teacher's answers for this job's photos (and its cached phrases and prompts), for later rounds and
        runs here. Not the whole shared folder: every job's answers, thousands of files, took minutes (2026-10-04)."""
        from ..workspace import IMAGE_EXT, content_key
        photos = {content_key(Path(d) / n) for d, _, names in os.walk(job.input, followlinks=True) for n in names
                  if Path(n).suffix.lower() in IMAGE_EXT}
        keys = [k for k in self.store.listing(cache_bucket, "labels/")
                if Path(k).stem in photos or k.split("/")[1] in ("phrases", "prompts", "siglip_text")]
        return self.store.pull(cache_bucket, keys, cache.root)

    def _secrets(self, llm: bool) -> list[str]:
        s3_secret = _env("SIM2CHIP_S3_SECRET", "sim2chip-s3")
        secrets = [f"AWS_ACCESS_KEY_ID={s3_secret}", f"AWS_SECRET_ACCESS_KEY={s3_secret}"]
        if llm:
            secret = _env("SIM2CHIP_NEBIUS_SECRET")
            if not secret and not self.dry_run:
                raise RuntimeError("the run calls Token Factory: set SIM2CHIP_NEBIUS_SECRET")
            secrets.append(f"NEBIUS_API_KEY={secret or '<secret>'}")
        return secrets

    def run_stage(self, stage: Stage, job: Job, cache: Cache, fp: str) -> None:
        if stage.runtime not in RUNTIMES:
            raise NotImplementedError(f"stage {stage.name} (runtime {stage.runtime}) has no cloud image yet: "
                                      "run it with the local backend")
        timeout_h = stage.max_hours
        if self.spent_usd + timeout_h * USD_PER_HOUR > self.max_usd:
            raise RuntimeError(f"stage {stage.name} could cost up to ${timeout_h * USD_PER_HOUR:.2f} "
                               f"(timeout {timeout_h} h) and ${self.spent_usd:.2f} is already spent: over the "
                               f"cap of ${self.max_usd:.2f} (SIM2CHIP_MAX_USD)")
        job_bucket, cache_bucket = self.buckets(job)
        code_url = self._upload_job(job, cache, job_bucket, cache_bucket)
        if not self.dry_run:
            self.store.delete(job_bucket, f"{stage.name}/{DONE}")  # a stale marker must not pass as success

        script = job.logs / f"{stage.name}.nebius.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(bootstrap(stage, fp))
        secrets = self._secrets(stage.needs_llm)
        name = f"s2c-{job.name}-{stage.name}".lower().replace("_", "-")[:63]
        t0 = time.time()
        env = {"S2C_JOB_BUCKET": job_bucket, "S2C_CACHE_BUCKET": cache_bucket, "S2C_CODE_URL": code_url,
               "SIM2CHIP_REGION": REGION}
        job_id = self.cli.create_job(name, RUNTIMES[stage.runtime], env, script, timeout_h, secrets)
        if self.dry_run:
            return
        print(f"  nebius job {job_id} ({PLATFORM}, {PRESET})", flush=True)
        state = self._wait(job_id)
        hours = (time.time() - t0) / 3600
        self.spent_usd += hours * USD_PER_HOUR
        self._record(job, stage.name, job_id, state, hours)

        # The job log (nebius ai job logs) is only its last lines: the full stage log comes from the bucket.
        tail = self.cli.job_logs(job_id)
        (job.logs / f"{stage.name}.job.log").write_text(tail)
        stage_log = job.logs / f"{stage.name}.log"
        stage_log.unlink(missing_ok=True)
        self.store.download(job_bucket, "logs", job.logs, (stage_log.name, "*.jsonl"))  # + Token Factory usage
        lines = stage_log.read_text(errors="replace").splitlines() if stage_log.exists() else []
        for line in lines + [l for l in tail.splitlines() if l not in lines]:
            if self.verbose or line.startswith(PREFIX):
                print("  " + line.removeprefix(PREFIX).strip(), flush=True)
        self.store.download(job_bucket, stage.name, job.stage_dir(stage.name), (DONE, *stage.results))
        if stage.uses_db:  # new asset reviews made in the cloud
            self.store.download(cache_bucket, "db", cache.db, ("*.jsonl",), overwrite=False)
        print(f"  {state} after {hours * 60:.1f} min, about ${hours * USD_PER_HOUR:.2f} "
              f"(run total ${self.spent_usd:.2f})", flush=True)

    def _wait(self, job_id: str) -> str:
        last = None
        while True:
            state = self.cli.job_state(job_id)
            if state != last:
                print(f"  state {state}", flush=True)
                last = state
            if state in TERMINAL:
                return state
            time.sleep(self.poll_s)

    @staticmethod
    def _record(job: Job, stage: str, job_id: str, state: str, hours: float, warm: bool = False,
                where: Where = Where(PLATFORM, PRESET)) -> None:
        """warm: a task of the warm GPU, whose whole life the server notes apart (warm.py): the daily budget skips
        these records, the usage panel shows them."""
        rec = {"t": round(time.time()), "stage": stage, "job_id": job_id, "state": state, "platform": where[0],
               "preset": where[1], "region": Where(*where).region, "hours": round(hours, 4),
               "usd_estimate": round(hours * usd_per_hour(where[0]), 4)}
        if warm:
            rec["warm"] = True
        with open(job.logs / "cloud_usage.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
