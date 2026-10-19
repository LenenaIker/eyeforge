"""GEN3C in batch for one place (the production idea, 2026-10-19): paths that cover the movement area, several
paths per Nebius job (the model loads once per job) and several jobs side by side. Each path starts at the capture
point looking along a heading and walks straight ahead, as far as the free-space rule allows (prep.reachable: 1 m
from anything standing) and at most RADIUS_M. Our intrinsics and our own path go into GEN3C through the job's
GEN3C_HFOV and GEN3C_TRAJ patches (gen3c_walk.JOB). Run in edge-sdg:

    python experiments/sphere_walk/gen3c_batch.py plan   --place <place>
    python experiments/sphere_walk/gen3c_batch.py launch --place <place> --jobs 2 [--max-hours 3] [--gpu auto]
    python experiments/sphere_walk/gen3c_batch.py fetch  --place <place>

Writes data/sphere_test/<place>/walk/G_gen3c/: paths.json (per path: heading, length, frames' camera-to-world in
Isaac's world), <name>_input.png, <name>_depth.npy, <name>_traj.npy and, after fetch, <name>.mp4."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen3c_walk as g  # noqa: E402
from prep import BODY_M, reachable  # noqa: E402
from sim2chip.cloud_sync import S3Store  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

PREFIX = "gen3c_batch"
HEADINGS = tuple(range(0, 360, 45))
RADIUS_M = 5.0
MIN_M = 1.5    # a path shorter than this shows nothing new
STEP_M = 0.25
# GPU by the work a job gets (user, 2026-10-19): a long batch on an H100 (GEN3C needs no RT cores, unlike Isaac),
# a short one on the default L40S, where the H100's higher price per hour does not pay back. The threshold is a
# guess until an H100 clip is timed; an H100 that cannot be had falls back to the L40S.
GPUS = {"l40s": ("gpu-l40s-d", "1gpu-16vcpu-96gb"), "h100": ("gpu-h100-sxm", "1gpu-16vcpu-200gb")}
H100_FROM_CLIPS = 8
PROMPT = "A walk through a sunny town square, realistic, sharp, natural light, static scene."


def place_dir(place: str) -> Path:
    return g.ROOT / "data" / "sphere_test" / place


def obstacles(d: Path) -> np.ndarray:
    geo = json.loads((d / "geometry.json").read_text())
    proxy = np.load(d / "proxy.npz")
    pw = render_plan.to_world(proxy["vertices"].astype(np.float64), geo["camera_height_m"])
    stand = ~proxy["ground"] & (pw[:, 2] > BODY_M[0]) & (pw[:, 2] < geo["camera_height_m"] + BODY_M[1])
    return pw[stand, :2]


def plan(place: str) -> list[dict]:
    d = place_dir(place)
    out = d / "walk" / "G_gen3c"
    out.mkdir(parents=True, exist_ok=True)
    h = json.loads((d / "geometry.json").read_text())["camera_height_m"]
    obs = obstacles(d)
    paths = []
    for hd in HEADINGS:
        fwd = render_plan.to_world(capture.rotation(hd, 0) @ np.array([0, 0, 1.0]), 0)
        fwd[2] = 0
        length = 0.0
        for s in np.arange(STEP_M, RADIUS_M + 1e-6, STEP_M):
            if not reachable(s * fwd, obs):
                break
            length = float(s)
        if length < MIN_M:
            print(f"h{hd:03d}: blocked at {length} m, left out")
            continue
        name = f"{place}_p{hd:03d}"
        s = length * np.arange(g.FRAMES) / (g.FRAMES - 1)
        traj = np.repeat(np.eye(4, dtype=np.float32)[None], g.FRAMES, 0)
        traj[:, 2, 3] = -s  # world to camera: the camera goes ahead along its own z, never turning
        np.save(out / f"{name}_traj.npy", traj)
        img, dep = g.inputs(place, hd)
        Image.fromarray(img).save(out / f"{name}_input.png")
        np.save(out / f"{name}_depth.npy", dep)
        v = g.view(hd)
        poses = [render_plan.camera_pose(v, h, si * fwd) for si in s]
        paths.append({"name": name, "heading": hd, "length_m": length, "view": v, "poses": poses})
        print(f"h{hd:03d}: {length} m")
    (out / "paths.json").write_text(json.dumps({"place": place, "paths": paths}))
    return paths


def gpu_for(clips: int, choice: str = "auto") -> list[str]:
    """The GPUs to try, in order, for a job of this many clips."""
    if choice != "auto":
        return [choice]
    return ["h100", "l40s"] if clips >= H100_FROM_CLIPS else ["l40s"]


def launch(place: str, jobs: int, max_hours: float, gpu: str = "auto"):
    out = place_dir(place) / "walk" / "G_gen3c"
    paths = json.loads((out / "paths.json").read_text())["paths"]
    store = S3Store()
    exp = max_hours + 2
    cases = []
    for p in paths:
        n = p["name"]
        for ext in ("_input.png", "_depth.npy", "_traj.npy"):
            store.put(g.BUCKET, f"{PREFIX}/in/{n}{ext}", (out / f"{n}{ext}").read_bytes())
        put = store.s3.generate_presigned_url("put_object", Params={"Bucket": g.BUCKET, "Key": f"{PREFIX}/out/{n}.mp4"},
                                              ExpiresIn=int(exp * 3600))
        cases.append(g.CASE.format(
            name=n, img=store.url(g.BUCKET, f"{PREFIX}/in/{n}_input.png", exp),
            dep=store.url(g.BUCKET, f"{PREFIX}/in/{n}_depth.npy", exp), out=put, far=p["length_m"],
            frames=g.FRAMES, prompt=PROMPT,
            fetch=f'curl -s -o /tmp/in/{n}_traj.npy "{store.url(g.BUCKET, f"{PREFIX}/in/{n}_traj.npy", exp)}"\n',
            env=f"GEN3C_TRAJ=/tmp/in/{n}_traj.npy GEN3C_HFOV={g.HFOV} "))
    cli = os.environ.get("SIM2CHIP_NEBIUS_CLI") or str(Path.home() / ".nebius/bin/nebius")
    for j in range(jobs):
        part = cases[j::jobs]
        if not part:
            continue
        log_url = store.s3.generate_presigned_url("put_object", ExpiresIn=int(exp * 3600),
                                                  Params={"Bucket": g.BUCKET, "Key": f"{PREFIX}/out/job{j}.log"})
        script = out / f"job{j}.sh"
        script.write_text(g.JOB.replace("__CASES__", "\n".join(part)))
        assert script.stat().st_size < 60_000, "the script must fit Nebius' 64 KiB inject limit"
        for kind in gpu_for(len(part), gpu):
            platform, preset = GPUS[kind]
            args = [cli, "ai", "job", "create", "--name", f"eyeforge-gen3c-b{j}-{int(time.time())}", "--image",
                    g.IMAGE, "--platform", platform, "--preset", preset, "--on-demand", "--timeout",
                    f"{int(max_hours * 60)}m", "--parent-id", os.environ["SIM2CHIP_NEBIUS_PROJECT"], "--inject-file",
                    f"{script}:/opt/gen3c_job.sh", "--container-command", "bash", "--args", "/opt/gen3c_job.sh",
                    "--env", f"LOG_URL={log_url}", "--async"]
            r = subprocess.run(args, capture_output=True, text=True)
            print(f"job {j}: {len(part)} paths on {kind}", r.stdout[-300:], r.stderr[-500:])
            if r.returncode == 0:
                break


def fetch(place: str):
    out = place_dir(place) / "walk" / "G_gen3c"
    store = S3Store()
    for k in store.listing(g.BUCKET, f"{PREFIX}/out/"):
        name = k.rsplit("/", 1)[-1]
        (out / name).write_bytes(store.s3.get_object(Bucket=g.BUCKET, Key=k)["Body"].read())
        print(out / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "launch", "fetch"])
    ap.add_argument("--place", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--gpu", choices=["auto", *GPUS], default="auto", help=f"auto: h100 from {H100_FROM_CLIPS} "
                    "clips per job, else l40s")
    a = ap.parse_args()
    if a.step == "plan":
        plan(a.place)
    elif a.step == "launch":
        launch(a.place, a.jobs, a.max_hours, a.gpu)
    else:
        fetch(a.place)


if __name__ == "__main__":
    main()
