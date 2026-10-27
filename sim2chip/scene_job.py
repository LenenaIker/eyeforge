"""The scene job: one of the owner's photos to a walk scene in their library, apart from any training (user,
2026-10-26). It drives the WorldStereo route (experiments/worldstereo/ws_route.py, recipe F3c) through its last
step, bundle, then adds the bundle to the library. The route's two WorldStereo legs run on the Nebius VM and cost
money (about 1 $ a scene): only a user's click (web) or the admin (CLI) starts a job.

Progress lives in <home>/scene_jobs/<scene id>.json: {"status", "step", "done", "total", "why", "pid"}, read by the
web; the route's whole output in <scene id>.log. The route's driver runs in SIM2CHIP_ROUTE_PYTHON (default this
Python): its cloud steps need boto3 (the edge-sdg env), which the route checks before its first step.

    python -m sim2chip.scene_job --home data --owner local --photo 123   (what the server and CLI start)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .config import env
from .workspace import REPO

ROUTE = REPO / "experiments" / "worldstereo" / "ws_route.py"
LAST = "bundle"
COST_USD = 1.0  # what a scene costs us, about: two WorldStereo runs on the VM and the vision model's calls
STEP_LINE = re.compile(r"^\[(?P<case>[^\]]+)\] (?P<step>[\w.]+)(?: \.\.\.|: done|: nothing to erase)")


def progress_file(home: Path, sid: str) -> Path:
    return Path(home) / "scene_jobs" / f"{sid}.json"


def progress(home: Path, sid: str) -> dict | None:
    f = progress_file(home, sid)
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def no_focal(photo: Path) -> str | None:
    """Why the route cannot take the photo, or None: it needs the field of view from EXIF (the 35 mm equivalent
    focal length)."""
    from PIL import Image
    try:
        im = Image.open(photo)
    except Exception as e:
        return f"the photo cannot be opened ({type(e).__name__})"
    try:
        if im.getexif().get_ifd(0x8769).get(0xA405):
            return None
    except Exception as e:  # a broken EXIF block: no focal length either
        return f"the photo's EXIF data cannot be read ({type(e).__name__}), so its focal length is unknown"
    return "the photo has no focal length in its EXIF data (the camera's field of view)"


# What the route needs on this machine (wsroute/paths.py ENVS and SIM2CHIP_GRUT_DIR): a server without them, the
# public one until the route moves to the VM, cannot make scenes.
ROUTE_VARS = ("SIM2CHIP_SAM3_PYTHON", "SIM2CHIP_SCENE_PYTHON", "SIM2CHIP_ERASE_PYTHON",
              "SIM2CHIP_MAPANYTHING_PYTHON", "SIM2CHIP_GSPLAT_PYTHON", "SIM2CHIP_GRUT_DIR")


def unavailable() -> str | None:
    """Why this machine cannot run the scene route, or None."""
    missing = [v for v in ROUTE_VARS if not (env(v) and Path(env(v)).exists())]
    return f"scene extraction is not set up on this server (missing {', '.join(missing)})" if missing else None


def route_python() -> str:
    return env("SIM2CHIP_ROUTE_PYTHON") or sys.executable


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def stale(home: Path, sid: str) -> bool:
    """A building scene whose job is gone (the server restarted, the process was killed): it never ends by itself.
    A job still queued in the server has no progress file yet; the server calls this only at start, when its queue
    is empty."""
    p = progress(home, sid)
    return not (p and p.get("status") == "building" and alive(p.get("pid")))


def photo_of(sid: str) -> str:
    """The route's name for the scene's photo (data/panogen_test/in/<name>.png)."""
    return f"lib_{sid}"


def case_of(sid: str) -> str:
    """The route's case: another name than the photo's, whose erased copy the route publishes under the case's."""
    return f"e2e_{photo_of(sid)}"


def steps(case: str) -> list[str]:
    """The route's steps for the case, in order (ws_route.py status)."""
    out = subprocess.run([route_python(), str(ROUTE), "status", "--case", case], capture_output=True, text=True,
                         check=True, cwd=REPO).stdout
    return [line[6:].split("  (paid)")[0].strip() for line in out.splitlines() if line.strip()]


def run(home: Path, owner: str, photo_id: int) -> str:
    """The whole job, blocking (hours): the library's scene goes from building to ready or failed."""
    from .library import Library
    lib = Library(Path(home), Path(home) / "cache")
    sid = lib.start_scene(owner, photo_id)
    pfile = progress_file(home, sid)
    pfile.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "building", "step": "starting", "done": 0, "total": 0, "why": None, "started": time.time(),
             "pid": os.getpid()}

    def save(**kw):
        state.update(kw)
        pfile.write_text(json.dumps(state))

    save()
    log = Path(home) / "scene_jobs" / f"{sid}.log"

    def short(*args):
        """A quick route command, its output in the log; a failure says the command's own last line."""
        r = subprocess.run([*py, *args], cwd=REPO, capture_output=True, text=True)
        with open(log, "a") as fh:
            fh.write(f"$ ws_route.py {' '.join(args)}\n{r.stdout}{r.stderr}")
        if r.returncode:
            last = (r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}").splitlines()[-1]
            raise RuntimeError(f"ws_route.py {args[0]} failed: {last}")

    py = [route_python(), str(ROUTE)]
    try:
        photo = lib.photo_file(owner, photo_id)
        if photo is None or not photo.exists():
            raise RuntimeError("the photo is no longer in the library")
        if why := no_focal(photo):
            raise RuntimeError(why)
        case = case_of(sid)
        short("ingest", str(photo), "--photo", photo_of(sid))
        short("init", "--photo", photo_of(sid), "--case", case)
        names = steps(case)
        save(total=len(names))
        tail: list[str] = []
        with open(log, "a") as fh, subprocess.Popen([*py, "run", "--case", case, "--until", LAST], cwd=REPO,
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                                    bufsize=1) as proc:
            for line in proc.stdout:
                fh.write(line)
                tail = (tail + [line.rstrip()])[-5:]
                m = STEP_LINE.match(line.strip())
                if m and m["step"] in names:
                    k = names.index(m["step"])
                    save(step=m["step"], done=k + (0 if line.rstrip().endswith("...") else 1))
        if proc.returncode:
            raise RuntimeError(f"the scene route stopped at {state['step']}: " + " | ".join(tail[-2:]))
        bundle = REPO / "data" / "worldstereo_test" / case / "bundle"  # wsroute.paths.DATA
        lib.add_scene(sid, bundle)
        save(status="ready", step=LAST, done=state["total"])
    except Exception as e:  # the library keeps the failure and why; the page shows it
        why = str(e) or type(e).__name__
        lib.fail_scene(sid, why)
        save(status="failed", why=why)
    return sid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--photo", type=int, required=True)
    a = ap.parse_args()
    print(run(Path(a.home).resolve(), a.owner, a.photo))


if __name__ == "__main__":
    main()
