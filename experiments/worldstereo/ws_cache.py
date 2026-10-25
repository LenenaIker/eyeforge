"""Phase 1 generated again with a 3D cache that grows leg by leg (ws_cache_job.py), one change per variant (C rows of
docs/pruebas_reconstruccion_worldstereo.md). Each variant is a scene of its own under data/worldstereo_test/cache/,
all in one VM session and one model load; every leg gets the same fresh seed (ws_knobs.py's _FreshNoise), so the
variants differ only in what they change:

    c0_control  phase 1 with its staged guides (the panorama's cloud only), as before: what a new seed alone changes
    c1_cache    the guide of each leg rendered from the panorama's cloud plus the hole pixels of every earlier leg
                (MapAnything depth), so an earlier leg's invention is no longer a hole
    c2_order    c1 with the turn back (wonder_0) last, after the second legs: the paper generates its rotations last
                so they draw on the most memory; upstream runs it before the second legs

    <eyeforge-sam3d env>/bin/python experiments/worldstereo/ws_cache.py stage   # scenes and cache_init.npz
    python experiments/worldstereo/ws_cache.py dry                              # local, in the image: guides only
    python experiments/worldstereo/ws_cache.py run                              # VM: start, generate, fetch, stop
    python experiments/worldstereo/ws_cache.py link                             # results as legs of photo21
    then ws_depth.py depth and consistency (sets C0, C1, C2)

dry runs the cache on the 3090 inside the job image with phase 1's old results standing in for the generations:
data/worldstereo_test/cache_dry/sheet.jpg shows each leg's staged guide next to the one the cache gives it.
`set -a; . ./.env; set +a` first, like ws_vm.py."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

SRC = "photo21"
BATCH = "cache"
PHASE1 = ("reconstruct_0", "reconstruct_1", "reconstruct_2", "wonder_0")
VARIANTS = ("c0_control", "c1_cache", "c2_order")
ORDER_C2 = ("reconstruct_0/traj0", "reconstruct_1/traj0", "reconstruct_2/traj0", "reconstruct_0/traj1",
            "reconstruct_1/traj1", "reconstruct_2/traj1", "wonder_0/traj0")
HERE = Path(__file__).resolve().parent


def stage(batch: str = BATCH, variants=VARIANTS):
    """scene/<variant>/: pano_bank and phase 1's staged legs; cache_init.npz (the panorama's guide cloud) for the
    cache variants, order.txt for c2."""
    src, root = ws.DATA / SRC, ws.DATA / batch
    if root.exists():
        shutil.rmtree(root)
    points, colors, _ = ws.guide_cloud(SRC)
    for v in variants:
        scene = root / "scene" / v
        for d in ("pano_bank", *PHASE1):
            shutil.copytree(src / "scene" / d, scene / d, ignore=shutil.ignore_patterns("*_render.jpg"))
        if v != "c0_control":
            np.savez(scene / "cache_init.npz", points=points.numpy().astype(np.float32),
                     colors=colors.numpy().astype(np.float16))
        if v == "c2_order":
            (scene / "order.txt").write_text("\n".join(ORDER_C2) + "\n")
    print(f"staged {variants} under {root} ({len(points)} panorama points)")


# Into the job's run_ws.py, after its own wrappers: the cache and the leg order.
INSTALL = """sys.path.insert(0, "/tmp")
import ws_cache_job, src.retrieval_wm as _rwm
ws_cache_job.install(du, _rwm)
"""


def run(batch: str = BATCH):
    import ws_knobs
    import ws_vm
    ws.PREFIX = f"worldstereo_{batch}"
    marker = "WorldStereo.from_pretrained = classmethod(from_pretrained)\n"
    wrap = "du.sort_trajs, du.load_mutli_traj_dataset = sort_trajs, load_mutli_traj_dataset\n"
    out_arg = "--input_path /tmp/scene/__CASE__ --output_path /tmp/out"
    heredoc = "cat > /tmp/run_ws.py <<'PY'\n"
    src = (HERE / "ws_cache_job.py").read_text()
    assert all(m in ws.JOB for m in (marker, wrap, out_arg, heredoc)) and "\nPYC\n" not in src
    ws.JOB = (ws.JOB.replace(marker, marker + ws_knobs.PATCH).replace(wrap, wrap + INSTALL)
              .replace(out_arg, out_arg + "/__CASE__")
              .replace(heredoc, "cat > /tmp/ws_cache_job.py <<'PYC'\n" + src + "PYC\n" + heredoc))
    ws.preflight = ws_knobs.preflight
    ws_vm.cmd_run(batch)


def dry():
    """The cache in the job image on this GPU, phase 1's old results standing in for the generations."""
    stage("cache_dry", ("c1_cache",))
    # The job image's pytorch3d has no kernels for the 3090 (8.6): WS_DRY_IMAGE names a local build of it with one.
    img = os.environ.get("WS_DRY_IMAGE") or f"{os.environ['SIM2CHIP_NEBIUS_REGISTRY']}/eyeforge-worldstereo:{ws.image_tag()}"
    code = f"""
import sys, glob, json, os, torch
sys.path.insert(0, "{HERE}"); sys.path.insert(0, ".")
import ws_cache_job as cj, src.data_utils as du
from src.general_utils import load_video
scene, res = "{ws.DATA}/cache_dry/scene/c1_cache", "{ws.DATA / SRC}/out/out/{SRC}"
out = "{ws.DATA}/cache_dry/out"
for p in du.sort_trajs(scene):
    view, traj = p.split("/")[-3:-1]
    cj.reguide(scene, out, view, traj)
    cam = json.load(open(f"{{scene}}/{{view}}/{{traj}}/camera.json"))
    frames = load_video(f"{{res}}/{{view}}/{{traj}}/{ws.MODEL}_result.mp4")
    cj.grow(scene, frames, torch.tensor(cam["extrinsic"]), torch.tensor(cam["intrinsic"]), view, traj)
"""
    mounts = []
    for d in (ws.ROOT, ws.TOOLS.parent):
        mounts += ["-v", f"{d}:{d}"]
    subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "HOME=/tmp", "-e", f"HF_HOME={ws.TOOLS / 'hf_cache'}", *mounts, "-w", "/opt/WorldStereo",
                    img, "python", "-c", code], check=True)
    dry_sheet()


def frames_of(path: Path) -> list[np.ndarray]:
    import cv2
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def dry_sheet():
    from PIL import ImageFont
    root = ws.DATA / "cache_dry"
    font = ImageFont.load_default(size=20)
    cw, ch = 416, 240
    rows = []
    for g in sorted((root / "out").glob("*/traj*/guide.json"), key=lambda p: p.stat().st_mtime):
        leg = f"{g.parent.parent.name}/{g.parent.name}"
        info = json.loads(g.read_text())
        staged = frames_of(ws.DATA / SRC / "scene" / leg / "render.mp4")
        cache = frames_of(g.parent / "guide_render.mp4")
        made = frames_of(ws.DATA / SRC / "out" / "out" / SRC / leg / f"{ws.MODEL}_result.mp4")
        rows.append((f"{leg}: holes {info['holes_staged']:.2f} staged -> {info['holes_cache']:.2f} with the cache",
                     [staged[10], cache[10], staged[-1], cache[-1], made[-1]]))
    heads = ["staged guide, frame 10", "cache guide, frame 10", "staged guide, last", "cache guide, last",
             "old generation, last"]
    sh = Image.new("RGB", (cw * 5, 40 + len(rows) * (ch + 26)), (0, 0, 0))
    d = ImageDraw.Draw(sh)
    d.text((8, 8), "   |   ".join(heads), fill=(255, 255, 255), font=font)
    for r, (label, tiles) in enumerate(rows):
        y = 40 + r * (ch + 26)
        d.text((8, y), label, fill=(255, 255, 0), font=font)
        for k, im in enumerate(tiles):
            sh.paste(Image.fromarray(im).resize((cw, ch)), (k * cw, y + 26))
    sh.save(root / "sheet.jpg", quality=88)
    print(root / "sheet.jpg")


def link(batch: str = BATCH):
    """Each variant's legs as legs of photo21 (reconstruct_0 -> c1_cache_reconstruct_0) in out/out/photo21 and
    legsets/cache, so ws_depth.py's depth and consistency see them next to phase 1."""
    res = ws.DATA / SRC / "out" / "out" / SRC
    sets = ws.DATA / SRC / "legsets" / batch
    for v in VARIANTS:
        got = ws.DATA / batch / "out" / "out" / batch / v
        for leg in PHASE1:
            name = f"{v}_{leg}"
            if not list((got / leg).glob(f"traj*/{ws.MODEL}_result.mp4")):
                print(f"missing {v}/{leg}")
                continue
            if (res / name).exists():
                shutil.rmtree(res / name)
            shutil.copytree(got / leg, res / name)
            for t in (got / leg).glob("traj*"):
                src = ws.DATA / batch / "scene" / v / leg / t.name
                shutil.copytree(src, sets / name / t.name, dirs_exist_ok=True)
    print(f"linked into {res} and {sets}")


# C3: the full model (40 UniPC steps, CFG 5) instead of the DMD student (4 steps), which the paper finds less
# consistent (Tanks and Temples F1 0.578 vs 0.534) and prone to copy the teacher's hallucinations. K0's setup (the
# quarter turns at the SUV, phase 1 in memory, seed 1024), only its first two turns: ~20x the DMD's work per leg.
FULL_MODEL = "worldstereo-memory"
FULL_LEGS = ("spg_0_0", "spg_0_1")
FULL_HOURS = 1.25


def stage_full():
    """data/worldstereo_test/full/: one scene (pano_bank, phase 1, two turns) and phase 1's DMD results renamed to
    the full model's file name, so the runner loads them into memory instead of generating them again."""
    src, root = ws.DATA / SRC, ws.DATA / "full"
    if root.exists():
        shutil.rmtree(root)
    scene = root / "scene" / "c3_full"
    for d in ("pano_bank", *PHASE1, *FULL_LEGS):
        shutil.copytree(src / "scene" / d, scene / d, ignore=shutil.ignore_patterns("*_render.jpg"))
    for d in PHASE1:
        for t in (src / "out" / "out" / SRC / d).glob("traj*"):
            dst = root / "out" / "out" / "full" / "c3_full" / d / t.name
            dst.mkdir(parents=True)
            shutil.copy(t / f"{ws.MODEL}_result.mp4", dst / f"{FULL_MODEL}_result.mp4")
    print(f"staged c3_full under {root}")


def run_full():
    import ws_knobs
    import ws_vm
    ws.PREFIX = "worldstereo_full"
    ws.MODEL = FULL_MODEL  # the job's __MODEL__: weights, runner and result names
    ws_vm.MAX_HOURS = FULL_HOURS
    marker = "WorldStereo.from_pretrained = classmethod(from_pretrained)\n"
    out_arg = "--input_path /tmp/scene/__CASE__ --output_path /tmp/out"
    assert marker in ws.JOB and out_arg in ws.JOB
    ws.JOB = ws.JOB.replace(marker, marker + ws_knobs.PATCH).replace(out_arg, out_arg + "/__CASE__")
    ws.preflight = ws_knobs.preflight
    ws_vm.cmd_run("full", "resume")


def link_full():
    """The two turns as legs of photo21 (c3_full_spg_0_0), the result renamed to the DMD file name that ws_depth
    reads."""
    res = ws.DATA / SRC / "out" / "out" / SRC
    got = ws.DATA / "full" / "out" / "out" / "full" / "c3_full"
    for leg in FULL_LEGS:
        name, f = f"c3_full_{leg}", got / leg / "traj0" / f"{FULL_MODEL}_result.mp4"
        if not f.exists():
            print(f"missing {leg}")
            continue
        if (res / name).exists():
            shutil.rmtree(res / name)
        (res / name / "traj0").mkdir(parents=True)
        shutil.copy(f, res / name / "traj0" / f"{ws.MODEL}_result.mp4")
        sets = ws.DATA / SRC / "legsets" / "full" / name
        shutil.copytree(ws.DATA / SRC / "scene" / leg / "traj0", sets / "traj0", dirs_exist_ok=True)
    print(f"linked into {res}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "dry", "run", "link", "dry_sheet", "stage_full", "run_full", "link_full"])
    a = ap.parse_args()
    {"stage": stage, "dry": dry, "run": run, "link": link, "dry_sheet": dry_sheet, "stage_full": stage_full,
     "run_full": run_full, "link_full": link_full}[a.cmd]()
