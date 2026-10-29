"""What the WorldStereo job reads and writes: the scene goes up to the cache bucket as a tar, job.sh (with run_ws.py
inside) is filled with presigned URLs, and the generated legs come back as a tar. The job runs on the VM (vm.py) in
the image build.sh pushes (experiments/worldstereo/Dockerfile).

With resume, the legs already generated (out/out/<case>) go along: upstream's runner skips them and loads them into
its memory, so only the new ones (next-best-view legs) are generated."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile

from .paths import HERE, MODEL, Case, env, required

BUCKET = env("SIM2CHIP_CACHE_BUCKET") or "sim2chip-cache"
PREFIX = "worldstereo_vm"  # in/ and out/ under it; the Serverless test used worldstereo_test


def image() -> str:
    """The job image; its tag is build.sh's, the hash of the Dockerfile without comments."""
    lines = (HERE.parent / "Dockerfile").read_text().splitlines(keepends=True)
    tag = hashlib.sha256("".join(ln for ln in lines if not ln.startswith("#")).encode()).hexdigest()[:12]
    return f"{required('SIM2CHIP_NEBIUS_REGISTRY')}/eyeforge-worldstereo:{tag}"


def preflight(case: Case, img: str, here: bool = False):
    """Free checks before paying for a GPU: the legs the job will read and the pushed image (here: pulled on this
    machine, the scene VM)."""
    legs = [t for t in case.scene.glob("*/traj*") if t.is_dir()]
    need = ("render.mp4", "render_mask.mp4", "camera.json", "traj_caption.json")
    missing = [f"{t.relative_to(case.scene)}/{n}" for t in legs for n in need if not (t / n).exists()]
    missing += [f"{t.parent.name}/start_frame.png" for t in legs if not (t.parent / "start_frame.png").exists()
                and not (t.parent / "chain_from.txt").exists()]
    missing += [] if (case.scene / "pano_bank/cameras.json").exists() else ["pano_bank/cameras.json"]
    if not legs or missing:
        sys.exit(f"scene not ready ({len(legs)} legs), missing: {missing}")
    if subprocess.run(["docker", "image" if here else "manifest", "inspect", img], capture_output=True).returncode:
        sys.exit(f"{img} is not {'on this machine' if here else 'in the registry: run experiments/worldstereo/build.sh'}")
    print(f"preflight ok: {len(legs)} legs, image {'here' if here else 'pushed'}")


def store():
    from sim2chip.cloud_sync import S3Store
    return S3Store()


def put_url(s3, key: str, hours: float) -> str:
    return s3.s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/out/{key}"},
                                        ExpiresIn=int(hours * 3600))


def job_script(case: str, scene_url: str, out_url: str, resume_url: str = "") -> str:
    """job.sh with its placeholders filled; LOG_URL comes as an environment variable."""
    text = (HERE / "job.sh").read_text().replace("__RUN_WS__", (HERE / "run_ws.py").read_text().rstrip("\n"))
    for k, v in {"__MODEL__": MODEL, "__CASE__": case, "__SCENE_URL__": scene_url, "__OUT_URL__": out_url,
                 "__RESUME_URL__": resume_url}.items():
        text = text.replace(k, v)
    return text


def upload(case: Case, hours: float, resume: bool = False):
    """The scene (and with resume the generated legs) up, job.sh written next to the case; returns (job.sh, the
    log's upload URL). The URLs last the job's timeout plus 2 h."""
    s3 = store()
    exp = hours + 2
    tar = case.dir / "scene.tar"
    with tarfile.open(tar, "w") as t:
        t.add(case.scene, arcname=case.name, filter=lambda ti: None if ti.name.endswith("_render.jpg") else ti)
    s3.put(BUCKET, f"{PREFIX}/in/{case.name}.tar", tar.read_bytes())
    resume_url = ""
    if resume:
        done = case.dir / "resume.tar"
        with tarfile.open(done, "w") as t:
            t.add(case.results, arcname=case.name)
        s3.put(BUCKET, f"{PREFIX}/in/{case.name}_resume.tar", done.read_bytes())
        resume_url = s3.url(BUCKET, f"{PREFIX}/in/{case.name}_resume.tar", exp)
    script = case.dir / "job.sh"
    script.write_text(job_script(case.name, s3.url(BUCKET, f"{PREFIX}/in/{case.name}.tar", exp),
                                 put_url(s3, f"{case.name}_out.tar", exp), resume_url))
    return script, put_url(s3, f"{case.name}_job.log", exp)


def fetch(case: Case):
    """This case's tar of generated legs (unpacked into out/) and its job log."""
    out = case.dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    s3 = store()
    for name in (f"{case.name}_out.tar", f"{case.name}_job.log"):
        key = f"{PREFIX}/out/{name}"
        if key not in s3.listing(BUCKET, key):
            print(f"not in the bucket: {key}")
            continue
        f = out / name
        f.write_bytes(s3.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        if f.suffix == ".tar":
            with tarfile.open(f) as t:
                t.extractall(out, filter="data")
        print(f)
