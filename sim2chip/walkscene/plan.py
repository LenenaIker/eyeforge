"""Cameras and objects of every frame in the job's walk scenes. A camera stands on the scene's free ground within
its radius, at the spec's eye height kept near the walks' (the splat only holds up there); the class's models stand
on free ground in front of it, at the distance where they fill the spec's share of the frame (camera_distance),
apart from each other and from obstacles, at a random heading.

Classifying: one class per frame (or none, empty_ratio of the frames, for the background class), every class an
equal share. Counting: 0 to generation.n_targets objects of any class (none in empty_ratio of the frames)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..spec import MEASURES, camera_distance
from . import Scene, w2c

EYE_HEIGHT = (1.3, 1.9)      # metres: the walks are filmed at 1.6; far from it the splat shows holes
PITCH_DEG = (-20.0, -2.0)    # looking slightly down, as the walks do
AIM_JITTER = 0.25            # an object's bearing within this share of the half field of view (classifying)
SPREAD = 0.85                # counting: objects anywhere within this share of the half field of view
EDGE = 0.95                  # an object's side no further out than this share of the half field of view
FOOTPRINT_STEP = 0.25        # metres between the points an object's footprint is checked at
GAP_M = 0.4                  # between two objects' footprints
TRIES = 60


@dataclass
class Model:
    """A 3D model of a class, at the real size drawn for it."""
    key: str
    usd: str
    cls: str
    measure: int        # MEASURES index the size is of
    size_m: float
    radius_m: float     # half the diagonal seen from above, at that size
    long_m: float       # the longest side, at that size


def models(assets: list[dict], spec, cache_root: Path) -> dict[str, list[Model]]:
    """The job's kept models by class (qa_assets/assets.json), each at a size drawn once from its class's range, as
    the template did (stable across runs and jobs)."""
    out: dict[str, list[Model]] = {}
    for a in assets:
        if a.get("role") == "prop" or a["class"] not in {c.id for c in spec.classes}:
            continue
        c = spec.cls(a["class"])
        if c.size_m is None:
            continue
        k = MEASURES.index(c.size_m.measure)
        if a["dims"][k] <= 0:
            continue
        size = random.Random(f"{spec.generation.seed}-{a['uid']}").uniform(*c.size_m.range)
        s = size / a["dims"][k]
        out.setdefault(a["class"], []).append(Model(
            a["uid"], str(Path(cache_root) / a["usd"]), a["class"], k, size,
            math.hypot(a["dims"][0], a["dims"][1]) * s / 2, max(a["dims"]) * s))
    return out


def footprint_free(scene: Scene, xy, radius: float) -> bool:
    """Every point of the disc of that radius around xy on free ground."""
    r = np.arange(-radius, radius + 1e-6, FOOTPRINT_STEP)
    gx, gy = np.meshgrid(r, r)
    disc = np.stack([gx.ravel(), gy.ravel()], 1)
    disc = disc[np.hypot(*disc.T) <= radius + 1e-6]
    return bool(scene.is_free(disc + np.asarray(xy)).all())


def place(scene: Scene, cam_xy, yaw: float, models_: list[Model], spec, rng: random.Random,
          aim: float) -> list[dict] | None:
    """Positions for the models in front of the camera, or None when they do not all fit."""
    cam = spec.camera
    half = cam.hfov_deg / 2
    taken: list[tuple[np.ndarray, float]] = []
    out = []
    for m in models_:
        for _ in range(TRIES):
            d = camera_distance(m.long_m, rng.uniform(*cam.target_fill), cam.hfov_deg, tuple(cam.distance_m))
            d = min(d, scene.meta["radius"] * 1.5)
            off = rng.uniform(-aim, aim) * half
            if d < m.radius_m + 1.0 or abs(off) + math.degrees(math.atan2(m.radius_m, d)) > half * EDGE:
                continue  # the whole object in the frame, not cut by its sides
            b = math.radians(yaw + off)
            xy = np.asarray(cam_xy) + d * np.array([math.cos(b), math.sin(b)])
            if any(np.hypot(*(xy - p)) < r + m.radius_m + GAP_M for p, r in taken):
                continue
            if footprint_free(scene, xy, m.radius_m):
                taken.append((xy, m.radius_m))
                out.append({"key": m.key, "usd": m.usd, "class": m.cls, "measure": m.measure, "size_m": m.size_m,
                            "xy": [round(float(v), 3) for v in xy], "yaw_deg": round(rng.uniform(0, 360), 1)})
                break
        else:
            return None
    return out


def contents(spec, classes: list[str], n: int, rng: random.Random) -> list[tuple[str | None, list[str]]]:
    """(the frame's class, the classes of its objects) for n frames."""
    gen, out = spec.generation, []
    counting = spec.task != "roi_classification"
    for i in range(n):
        empty = rng.random() < gen.empty_ratio and (counting or spec.background_class)
        if counting:
            k = 0 if empty else rng.randint(1, gen.n_targets)
            out.append((None, [rng.choice(classes) for _ in range(k)]))
        elif empty:
            out.append((spec.background_class, []))
        else:
            c = classes[i % len(classes)]
            out.append((c, [c]))
    return out


def plan(scenes: dict[str, Scene], assets: list[dict], spec, cache_root: Path, n: int,
         size: tuple[int, int]) -> dict:
    """Every frame's scene, camera and objects (plan.json): frames whose objects find no room are tried again with
    another camera, then left out."""
    rng = random.Random(spec.generation.seed)
    by_class = models(assets, spec, cache_root)
    classes = sorted(by_class)
    if not classes:
        return {"size": list(size), "fov_deg": spec.camera.hfov_deg, "views": [], "why": "no 3D model of any class"}
    lo, hi = spec.camera.height_m
    eye = (min(max(lo, EYE_HEIGHT[0]), EYE_HEIGHT[1]), min(max(hi, EYE_HEIGHT[0]), EYE_HEIGHT[1]))
    ids = sorted(scenes)
    stands = {s: scenes[s].free_points(within=scenes[s].meta["radius"]) for s in ids}
    views, missed = [], 0
    for i, (cls, want) in enumerate(contents(spec, classes, n, rng)):
        sid = ids[i % len(ids)]
        sc = scenes[sid]
        if not len(stands[sid]):
            missed += 1
            continue
        aim = AIM_JITTER if spec.task == "roi_classification" else SPREAD
        for _ in range(TRIES):
            cam_xy = stands[sid][rng.randrange(len(stands[sid]))]
            yaw = rng.uniform(0, 360)
            objs = place(sc, cam_xy, yaw, [rng.choice(by_class[c]) for c in want], spec, rng, aim)
            if objs is not None:
                break
        else:
            missed += 1
            continue
        z = rng.uniform(*eye) - sc.camera_height
        views.append({"name": f"{len(views):04d}", "scene": sid, "class": cls, "objects": objs,
                      "w2c": w2c([*cam_xy, z], yaw, rng.uniform(*PITCH_DEG)).round(6).tolist()})
    return {"size": list(size), "fov_deg": spec.camera.hfov_deg, "views": views, "missed": missed}
