"""The finished scene as the library keeps it, <case>/bundle/: what a training job needs to place objects in it and
render them over it, and nothing about any class (a scene is only the scene, user 2026-10-26).

    splat.ply    the final arm's 3DGS (the training render reads it with gsplat)
    splat.usdz   its NuRec USDZ (Isaac Sim, the video); hard links, no copy
    sky.jpg      the panorama: the colour along a ray the splat leaves empty (the sky is at infinity)
    free.npz     ground grid: occupied cells (an obstacle point inside) and free ones (a camera may stand there)
    card.jpg     the photo's direction from the walks' centre, for the library's card
    scene.json   conventions (metres, z up, the photo's camera at the origin looking along x, the ground at
                 -camera_height), the centre, the radius, the grid's layout, the coverage scores and the model
                 that made the panorama
(SIM2CHIP_GSPLAT_PYTHON)"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
from PIL import Image

from . import camera, student
from .camera import CAMERA_HEIGHT, OBSTACLE_Z
from .paths import PANOS, Case

FORMAT = 1
CARD_SIZE = (640, 400)
CARD_PITCH_DEG = -5.0
GRID_CELL_M, GRID_SPAN_M = 0.25, 12.0
OLD_PANO_MODEL = "WorldGen img2scene LoRA on FLUX.1-Fill-dev (non-commercial)"  # every pano before pano.py's record


def pano_model(case: Case, pano_name: str) -> str:
    """The model that made the case's panorama, from pano.py's record next to it (pano_s<seed>.json)."""
    f = PANOS / case.name / pano_name.replace("_paste.jpg", ".json")
    return json.loads(f.read_text()).get("model", OLD_PANO_MODEL) if f.exists() else OLD_PANO_MODEL


def link(src, dst):
    """dst as a hard link to src (a copy across file systems)."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def card(sp, sky: np.ndarray, centre) -> Image.Image:
    """The splat from the walks' centre at eye height, looking the photo's way; the panorama behind it."""
    w, h = CARD_SIZE
    m = camera.w2c([centre[0], centre[1], 0.0], 0.0, CARD_PITCH_DEG)
    K = camera.intrinsic_fov(w, h, student.STUDENT_FOV_DEG)
    rgb, _, alpha = student.render(sp, m[None].astype(np.float32), K, w, h)
    a = alpha[0].cpu().numpy()[..., None]
    out = rgb[0].cpu().numpy() * a + camera.pano_lookup(sky, m, K, w, h) * (1 - a)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def bundle(case_name: str, arm: str, radius: float, photo: str):
    import torch
    case = Case(case_name)
    out = case.dir / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    ply = case.splat_ply(arm)
    link(ply, out / "splat.ply")
    if ply.with_suffix(".usdz").exists():
        link(ply.with_suffix(".usdz"), out / "splat.usdz")
    prep = json.loads(case.prep_json.read_text())
    sky = Image.open(PANOS / case.name / prep["pano"]).convert("RGB")
    sky.save(out / "sky.jpg", quality=92)

    sp = student.load_splat(ply, torch.device("cuda"))
    centre = student.seed(case)
    occ, free = student.free_grid(student.obstacles(case, sp), centre, GRID_CELL_M, GRID_SPAN_M)
    np.savez_compressed(out / "free.npz", occupied=occ, free=free)
    card(sp, np.asarray(sky), centre).save(out / "card.jpg", quality=88)

    grid = case.conflict(arm) / "grid.json"
    scores = json.loads(grid.read_text())["all"] if grid.exists() else {}
    meta = dict(
        format=FORMAT, route="walk", case=case.name, photo=photo, arm=arm, made=time.strftime("%Y-%m-%dT%H:%M:%S"),
        units="m", up="z", camera_height=CAMERA_HEIGHT, ground_z=-CAMERA_HEIGHT, obstacle_z=list(OBSTACLE_Z),
        convention="the photo's camera at the origin looking along +x, y to its left; OpenCV world-to-camera",
        centre=[float(centre[0]), float(centre[1])], radius=radius,
        grid=dict(file="free.npz", cell_m=GRID_CELL_M, span_m=GRID_SPAN_M, clearance_m=student.CLEARANCE_M,
                  index="cell [i, j] covers x from i*cell - span, y from j*cell - span"),
        free_m2=round(float(free.sum()) * GRID_CELL_M ** 2, 1), scores=scores,
        pano_model=pano_model(case, prep["pano"]),
        files=sorted(p.name for p in out.iterdir() if p.name != "scene.json"))
    (out / "scene.json").write_text(json.dumps(meta, indent=1))
    print(f"[{case.name}] bundle: {out} ({meta['free_m2']} m2 free)", flush=True)
