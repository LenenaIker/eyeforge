"""The whole scene route on the Nebius GPU VM (vm.py's RTX PRO 6000), so making a scene needs no GPU where it is
asked for: the web server or the CLI starts the VM, sends the code and the photo, runs the route there step by step
(printing its lines, which scene_job.py follows), brings the bundle back and always stops the VM.

On the VM's weights disk (vm.MNT), next to WorldStereo's weights and image:
    envs/sam3, envs/erase, envs/driver   the SAM 3 and Qwen Pythons (envs/*.txt, pinned) and the route's driver
    tools/                               MoGe's code and checkpoint, the 4-bit Qwen pipeline, WorldStereo's code
    hf/                                  the Hugging Face cache (WorldStereo's, SAM 3, the LoRAs, MapAnything)
    eyeforge/                            the code of each run and data/ (the cases, Fixer's code and weights)
The steps of SIM2CHIP_SCENE_PYTHON, SIM2CHIP_MAPANYTHING_PYTHON and SIM2CHIP_GSPLAT_PYTHON run in the route image
(route.Dockerfile) through route_python.sh, Fixer in its own image, WorldStereo's legs in its image on the same
GPU (vm.py, SIM2CHIP_WS_HERE).

    ws_route.py vm setup                                 fill the disk (once; idempotent, ~1 GPU hour)
    ws_route.py remote --photo-file F --photo P --case C the route for one photo, through bundle"""
from __future__ import annotations

import io
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import cloud
from .paths import DATA, HERE, ROOT, env, required
from .vm import GPU_VM, MNT, MOUNT, SSH_OPTS, USER, docker_login, get, neb, note, stop, wait_ssh, wait_stopped

REPO = f"{MNT}/eyeforge"
DRIVER = f"{MNT}/envs/driver/bin/python"
TORCH_CU128 = "https://download.pytorch.org/whl/cu128"
FIXER_COMMIT = "b39dfca"  # github.com/nv-tlabs/Fixer
MOGE_COMMIT = "74fbce0"   # github.com/microsoft/MoGe
NF4 = "weights/scene/qwen-nf4"  # the 4-bit Qwen-Image-Edit pipeline in the cache bucket (erase_qwen --save-nf4)
# What the run sends: the package the steps import and the route's code.
CODE = ["sim2chip", "experiments/worldstereo/ws_route.py", "experiments/worldstereo/wsroute",
        "experiments/worldstereo/route_python.sh", "experiments/worldstereo/Dockerfile",
        "experiments/sphere_walk/gs_usd_light.py"]
# The settings the route reads from .env on the VM, besides the paths below.
SECRETS = ("NEBIUS_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "SIM2CHIP_CACHE_BUCKET",
           "SIM2CHIP_NEBIUS_REGISTRY", "SIM2CHIP_LLM_DAILY_USD")


def images() -> dict[str, str]:
    reg = required("SIM2CHIP_NEBIUS_REGISTRY")
    return {"worldstereo": cloud.image(), "route": f"{reg}/eyeforge-route:{route_tag()}",
            "fixer": env("SIM2CHIP_FIXER_IMAGE") or f"{reg}/eyeforge-fixer:5ae3be1c0479"}


def route_tag() -> str:
    """build.sh's tag of the route image: the hash of route.Dockerfile and of the WorldStereo tag it starts from."""
    import hashlib
    text = (HERE.parent / "route.Dockerfile").read_text() + cloud.image()
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def vm_env() -> str:
    """The VM's .env: its own paths and the few secrets the steps need (never traced or printed)."""
    route = f"{REPO}/experiments/worldstereo/route_python.sh"
    img = images()
    lines = {
        "SIM2CHIP_TOOLS": f"{MNT}/tools", "SIM2CHIP_SAM3_PYTHON": f"{MNT}/envs/sam3/bin/python",
        "SIM2CHIP_ERASE_PYTHON": f"{MNT}/envs/erase/bin/python", "SIM2CHIP_SCENE_PYTHON": route,
        "SIM2CHIP_MAPANYTHING_PYTHON": route, "SIM2CHIP_GSPLAT_PYTHON": route, "SIM2CHIP_GRUT_DIR": "",
        "SIM2CHIP_ROUTE_IMAGE": img["route"], "SIM2CHIP_FIXER_IMAGE": img["fixer"], "SIM2CHIP_ROUTE_MOUNTS": MNT,
        "SIM2CHIP_WS_HERE": "1", "SIM2CHIP_QWEN_NF4": f"{MNT}/tools/qwen_edit_nf4", "HF_HOME": f"{MNT}/hf",
        "TORCH_HOME": f"{MNT}/torch", "MOGE_CKPT": f"{MNT}/tools/moge_ckpt/moge-2-vitl/model.pt",
    }
    lines.update({k: env(k) for k in SECRETS if env(k)})
    return "".join(f"{k}={shlex.quote(v)}\n" for k, v in lines.items())


def ssh(host: str, cmd: str, check=True, **kw) -> subprocess.CompletedProcess:
    r = subprocess.run(["ssh", *SSH_OPTS, "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
                        f"{USER}@{host}", cmd], **kw)
    if check and r.returncode:
        raise RuntimeError(f"on the scene VM: `{cmd[:120]}` failed ({r.returncode})")
    return r


def code_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for rel in CODE:
            t.add(ROOT / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def start() -> str:
    """The GPU VM running and answering; its address."""
    vm = wait_stopped(GPU_VM)
    if vm["status"].get("state") != "RUNNING":
        neb("compute", "instance", "start", "--id", vm["metadata"]["id"])
    return wait_ssh(GPU_VM)


def prepare(host: str):
    """Disk mounted, the three images pulled (a no-op once there), this run's code and .env in place."""
    ssh(host, MOUNT, capture_output=True)
    docker_login(host)
    for img in images().values():
        ssh(host, f"sudo docker image inspect {img} >/dev/null 2>&1 || sudo docker pull -q {img}")
    ssh(host, f"sudo mkdir -p {REPO} && sudo chown {USER}: {REPO} && cd {REPO} && "
              "rm -rf sim2chip experiments && tar -xzf -", input=code_tar())
    ssh(host, f"umask 077 && cat > {REPO}/.env", input=vm_env().encode())


def remote(photo_file: Path, photo: str, case: str, out: Path | None = None, hours: float = 4.0):
    """The route for one photo on the VM through bundle, its lines printed as they come; the bundle lands in
    <out>/bundle (default this machine's data/worldstereo_test/<case>). One run at a time on the VM (the
    WorldStereo lock)."""
    from sim2chip.locks import WS_VM, hold
    with hold(WS_VM, f"scene route of {case}"):
        t0 = time.time()
        host = start()
        note(step="scene_boot", case=case, s=round(time.time() - t0))
        try:
            prepare(host)
            ssh(host, f"mkdir -p {REPO}/in && cat > {REPO}/in/{shlex.quote(photo_file.name)}",
                input=photo_file.read_bytes())
            route = f"{DRIVER} experiments/worldstereo/ws_route.py"
            script = (f"cd {REPO} && set -e && {route} ingest in/{shlex.quote(photo_file.name)} --photo {photo} && "
                      f"{route} init --photo {photo} --case {case} >/dev/null && "
                      f"{route} status --case {case} | sed 's/^/[status] /' && "
                      f"timeout {int(hours * 3600)} {route} run --case {case} --until bundle")
            r = ssh(host, f"bash -c {shlex.quote(script)} 2>&1", check=False)
            note(step="scene_route", case=case, s=round(time.time() - t0), rc=r.returncode)
            if r.returncode:
                raise RuntimeError(f"the scene route stopped on the VM (exit {r.returncode})")
            out = Path(out or DATA / case)
            out.mkdir(parents=True, exist_ok=True)
            got = subprocess.run(["ssh", *SSH_OPTS, f"{USER}@{host}",
                                  f"tar -C {REPO}/data/worldstereo_test/{case} -cf - bundle"],
                                 stdout=subprocess.PIPE, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(got)) as t:
                t.extractall(out, filter="data")
            print(f"[{case}] bundle fetched -> {out / 'bundle'}", flush=True)
        finally:
            ssh(host, f"rm -f {REPO}/.env", check=False)
            stop()
            note(step="scene_total", case=case, s=round(time.time() - t0))


SETUP = r"""set -e
W=__MNT__
sudo usermod -aG docker __USER__
sudo mkdir -p $W/envs $W/tools $W/uv $W/torch __REPO__/data/fixer && sudo chown __USER__: $W/envs $W/tools $W/uv $W/torch __REPO__ __REPO__/data __REPO__/data/fixer
export UV_INSTALL_DIR=$W/uv/bin UV_PYTHON_INSTALL_DIR=$W/uv/python UV_CACHE_DIR=$W/uv/cache UV_NO_MODIFY_PATH=1 PATH=$W/uv/bin:$PATH
[ -x $W/uv/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
for e in sam3 erase; do
  [ -x $W/envs/$e/bin/python ] || { uv venv -q -p 3.11 $W/envs/$e && uv pip install -q -p $W/envs/$e/bin/python \
    -r /tmp/envs/$e.txt --extra-index-url __TORCH__ --index-strategy unsafe-best-match; }
done
[ -x $W/envs/driver/bin/python ] || { uv venv -q -p 3.11 $W/envs/driver && uv pip install -q -p $W/envs/driver/bin/python numpy==1.26.4 pillow boto3; }
tar -C $W/tools -xzf /tmp/tools.tgz
ln -sfn /opt/WorldStereo $W/tools/WorldStereo
[ -d __REPO__/data/fixer/src/.git ] || git clone -q https://github.com/nv-tlabs/Fixer __REPO__/data/fixer/src
git -C __REPO__/data/fixer/src checkout -q __FIXER__
export HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1
sudo mkdir -p $W/hf && sudo chown -R __USER__: $W/hf
$W/envs/erase/bin/python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("facebook/sam3")
snapshot_download("facebook/map-anything-apache")
snapshot_download("prithivMLmods/Qwen-Image-Edit-2511-Object-Remover")
hf_hub_download("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
hf_hub_download("mickmumpitz/QWEN-EDIT_360", "251018_MICKMUMPITZ_QWEN-EDIT_360_03.safetensors")
snapshot_download("Ruicheng/moge-2-vitl", local_dir="__MNT__/tools/moge_ckpt/moge-2-vitl")
snapshot_download("nvidia/Fixer", local_dir="__REPO__/data/fixer/models")
print("weights ok")
PY
mkdir -p $W/tools/qwen_edit_nf4
cd $W/tools/qwen_edit_nf4 && while read -r rel url; do
  [ -s "$rel" ] || { mkdir -p "$(dirname "$rel")"; curl -sfL -o "$rel" "$url"; }
done < /tmp/nf4_urls.txt
du -sh $W/envs/* $W/tools/* $W/hf; df -h $W
"""


def setup():
    """Fill the VM's disk for the route (idempotent): the SAM 3, Qwen and driver Pythons, the images, the weights,
    MoGe and Fixer. Needs HF_TOKEN (SAM 3 is gated) and the cache bucket's keys here."""
    token = env("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    tools = io.BytesIO()
    with tarfile.open(fileobj=tools, mode="w:gz") as t:
        base = Path(required("SIM2CHIP_TOOLS"))
        for rel in ("MoGe", "moge_deps"):  # MoGe's code and the two utils3d versions it and the route import
            t.add(base / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name or "/.git" in ti.name
                  else ti)
    s3 = cloud.store()
    keys = [o for o in s3.listing(cloud.BUCKET, NF4 + "/")]
    urls = "".join(f"{k[len(NF4) + 1:]} {s3.url(cloud.BUCKET, k, 3)}\n" for k in keys)
    t0 = time.time()
    host = start()
    try:
        ssh(host, MOUNT, capture_output=True)
        ssh(host, "mkdir -p /tmp/envs && tar -C /tmp/envs -xf -", input=_tar_dir(HERE.parent / "envs"))
        ssh(host, "cat > /tmp/tools.tgz", input=tools.getvalue())
        ssh(host, "cat > /tmp/nf4_urls.txt", input=urls.encode())
        docker_login(host)
        for img in images().values():
            ssh(host, f"sudo docker image inspect {img} >/dev/null 2>&1 || sudo docker pull -q {img}")
        script = (SETUP.replace("__MNT__", MNT).replace("__USER__", USER).replace("__REPO__", REPO)
                  .replace("__TORCH__", TORCH_CU128).replace("__FIXER__", FIXER_COMMIT))
        ssh(host, "cat > /tmp/setup.sh", input=script.encode())
        ssh(host, "read -r HF_TOKEN; export HF_TOKEN; bash /tmp/setup.sh", input=(token + "\n").encode())
        note(step="scene_setup", s=round(time.time() - t0))
    finally:
        stop()


def _tar_dir(d: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for f in sorted(d.iterdir()):
            t.add(f, arcname=f.name)
    return buf.getvalue()
