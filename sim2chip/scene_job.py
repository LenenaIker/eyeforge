"""The scene job: one of the owner's photos to a walk scene in their library, apart from any training (user,
2026-10-26). It drives the WorldStereo route (experiments/worldstereo/ws_route.py, recipe F3c) through its last
step, bundle, then adds the bundle to the library. The route's two WorldStereo legs run on the Nebius VM and cost
money (about 1 $ a scene): only a user's click (web) or the admin (CLI) starts a job.

Progress lives in <home>/scene_jobs/<scene id>.json: {"status", "step", "done", "total", "why"}, read by the web.

    python -m sim2chip.scene_job --home data --owner local --photo 123   (what the server and CLI start)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .workspace import REPO

ROUTE = REPO / "experiments" / "worldstereo" / "ws_route.py"
LAST = "bundle"
COST_USD = 1.0  # what a scene costs us, about: two WorldStereo runs on the VM and the vision model's calls
STEP_LINE = re.compile(r"^\[(?P<case>[^\]]+)\] (?P<step>[\w.]+)(?: \.\.\.|: done)")


def progress_file(home: Path, sid: str) -> Path:
    return Path(home) / "scene_jobs" / f"{sid}.json"


def progress(home: Path, sid: str) -> dict | None:
    f = progress_file(home, sid)
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def has_focal(photo: Path) -> bool:
    """Whether the photo's EXIF gives its field of view (35 mm equivalent focal length): the route needs it."""
    from PIL import Image
    try:
        return bool(Image.open(photo).getexif().get_ifd(0x8769).get(0xA405))
    except Exception:
        return False


def photo_of(sid: str) -> str:
    """The route's name for the scene's photo (data/panogen_test/in/<name>.png)."""
    return f"lib_{sid}"


def case_of(sid: str) -> str:
    """The route's case: another name than the photo's, whose erased copy the route publishes under the case's."""
    return f"e2e_{photo_of(sid)}"


def steps(case: str) -> list[str]:
    """The route's steps for the case, in order (ws_route.py status)."""
    out = subprocess.run([sys.executable, str(ROUTE), "status", "--case", case], capture_output=True, text=True,
                         check=True, cwd=REPO).stdout
    return [line[6:].split("  (paid)")[0].strip() for line in out.splitlines() if line.strip()]


def run(home: Path, owner: str, photo_id: int) -> str:
    """The whole job, blocking (hours): the library's scene goes from building to ready or failed."""
    from .library import Library
    lib = Library(Path(home), Path(home) / "cache")
    sid = lib.start_scene(owner, photo_id)
    pfile = progress_file(home, sid)
    pfile.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "building", "step": "starting", "done": 0, "total": 0, "why": None, "started": time.time()}

    def save(**kw):
        state.update(kw)
        pfile.write_text(json.dumps(state))

    save()
    try:
        photo = lib.photo_file(owner, photo_id)
        if photo is None or not photo.exists():
            raise RuntimeError("the photo is no longer in the library")
        if not has_focal(photo):
            raise RuntimeError("the photo has no focal length in its EXIF data (the camera's field of view)")
        case = case_of(sid)
        py = [sys.executable, str(ROUTE)]
        subprocess.run([*py, "ingest", str(photo), "--photo", photo_of(sid)], check=True, cwd=REPO,
                       capture_output=True)
        subprocess.run([*py, "init", "--photo", photo_of(sid), "--case", case], check=True, cwd=REPO,
                       capture_output=True)
        names = steps(case)
        save(total=len(names))
        log = Path(home) / "scene_jobs" / f"{sid}.log"
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
                    save(step=m["step"], done=k + (1 if line.rstrip().endswith("done") else 0))
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
    print(run(Path(a.home), a.owner, a.photo))


if __name__ == "__main__":
    main()
