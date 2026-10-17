"""S.6 of the sphere route, extrapolation (docs/roadmap_photosphere.md): new views turned around the capture point,
each with 1 to 5 targets placed where the sphere could hold them, for steps/render.py to draw. Alhaija et al.
(IJCV 2018): placing on the ground plane with the road's segmentation and a random heading works as well as by
hand; more than 5 cars per image does worse (docs/conocimiento_photosphere.md, placement).

Where: on the kinds of ground the erased targets stood on (their contacts' SAM 3 votes; a sphere without targets
takes the paved kinds), on ground the proxy measured (render_plan.Ground), with the whole footprint free and apart
from the others.
Heading: the erased target nearest the spot, if one stood within NEIGHBOUR_M (rows of parked cars keep their
order; its front from Orient Anything, S.5), else along the nearest wall (the street), else any.
View: a heading drawn at random (around the user's camera heading when given), the spec's pitch with a little
spread, and the camera raised or lowered a little (P.6: at most 0.3 m).
Numpy, run in the main Python.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import capture, render_plan
from .geometry import PAVED

OBJECTS_PER_VIEW = (1, 5)
DISTANCE_M = (4.0, 25.0)      # horizontal distances tried for a target (nearer it fills the view, farther it is
                              # a few pixels on a phone camera)
NEIGHBOUR_M = 6.0             # an erased target this close gives its heading
HEADING_JITTER_DEG = 8.0
FLIP_SHARE = 0.3              # of copied headings turned round (cars park nose in and nose out)
GAP_M = 0.4                   # between two placed footprints
TRIES = 40                    # placements tried per object
PITCH_SPREAD_DEG = 4.0
HEIGHT_SPREAD_M = 0.2         # P.6: the camera stays within 0.3 m of the capture point
HEADING_SPREAD_DEG = 30.0     # around the user's camera heading, when given
KIND_SHARE = 0.2              # of the erased targets' footprints for a kind of ground to count as theirs
FOOTPRINT_ON_KIND = 0.8       # of a new footprint's points on those kinds
MIN_VISIBLE_PX = 150          # a target showing fewer pixels is left unlabelled (it is still in the image)


def allowed_kinds(kind_map: "KindMap", fits: list[dict] | None) -> set[str]:
    """The kinds of ground the erased targets stood on: the erased panorama's kinds under each fit's footprint,
    those covering at least KIND_SHARE of the points. None stood: the paved kinds."""
    if not fits:
        return set(PAVED)
    seen = [kind_map.at(q) for f in fits for q in footprint_points(
        np.array([f["centre_xz"][0], f["ground_y"], f["centre_xz"][1]]), f["size_m"][0], f["size_m"][1],
        f["yaw_deg"])]
    seen = [k for k in seen if k]
    out = {k for k in set(seen) if seen.count(k) >= KIND_SHARE * len(seen)}
    return out or set(PAVED)


def footprint_points(centre: np.ndarray, length_m: float, width_m: float, yaw_deg: float, grid=(5, 3)):
    """A grid over a footprint (levelled points at the centre's height), as render_plan.Ground checks it."""
    a = math.radians(yaw_deg)
    along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, grid[0]) * length_m, np.linspace(-0.5, 0.5, grid[1]) * width_m)
    xy = u.reshape(-1, 1) * along + v.reshape(-1, 1) * across
    return centre + np.stack([xy[:, 0], np.zeros(len(xy)), xy[:, 1]], 1)


def pitch_range(captured: dict, fov_deg: float, size, margin_deg: float = 1.0) -> tuple[float, float]:
    """Pitches (down positive) that keep a view inside the captured band (sphere.json "captured"): a phone's
    photosphere stops about 30 degrees under the horizon, and past it the view shows the uncaptured black."""
    vfov = math.degrees(2 * math.atan(math.tan(math.radians(fov_deg) / 2) * size[1] / size[0]))
    lo = -(captured["up_deg"] - vfov / 2 - margin_deg)
    hi = captured["down_deg"] - vfov / 2 - margin_deg
    return (lo, hi) if lo <= hi else ((lo + hi) / 2, (lo + hi) / 2)


class KindMap:
    """The ground kind under a levelled point, from SAM 3's votes on the erased panorama."""

    def __init__(self, d: Path, R: np.ndarray):
        votes = np.load(d / "stuff_bg.npz") if (d / "stuff_bg.npz").exists() else np.load(d / "stuff.npz")
        self.kinds = [k for k in ("road", "parking", "sidewalk", "grass", "floor") if k in votes.files]
        stack = np.stack([votes[k].astype(np.float32) for k in self.kinds])
        if "floor" in self.kinds:  # "ground, floor" is SAM 3's catch-all: it names a pixel only when nothing else does
            f = self.kinds.index("floor")
            others = np.delete(stack, f, 0).max(0) > 0.3
            stack[f] = np.where(others, 0, stack[f])
        self.best = np.where(stack.max(0) > 0.3, stack.argmax(0), -1)
        self.R = R
        self.H, self.W = self.best.shape

    def at(self, p_level: np.ndarray) -> str | None:
        x, y = capture.to_pixel(render_plan.pano_rays(p_level / np.linalg.norm(p_level), self.R), self.W, self.H)
        k = self.best[min(max(int(round(y)), 0), self.H - 1), int(round(x)) % self.W]
        return self.kinds[k] if k >= 0 else None


def wall_axis(walls: list[dict], p_xz: np.ndarray, max_m: float = 15.0) -> float | None:
    """Yaw (degrees, box_corners' convention) along the wall nearest a point, if one is within max_m."""
    best = None
    for w in walls:
        a, b = (np.asarray(e, float) for e in w["ends_xz"])
        t = np.clip((p_xz - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
        dist = np.linalg.norm(p_xz - (a + t * (b - a)))
        if dist <= max_m and (best is None or dist < best[0]):
            best = (dist, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
    return None if best is None else best[1]


def overlaps(c1, yaw1, s1, c2, yaw2, s2, gap: float = GAP_M) -> bool:
    """Whether two footprints (centre xz, yaw, (length, width)) come closer than gap: separating axes."""
    def corners(c, yaw, s):
        a = math.radians(yaw)
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        L, W = s[0] / 2 + gap / 2, s[1] / 2 + gap / 2
        return np.array([c + su * L * u + sv * W * v for su in (-1, 1) for sv in (-1, 1)])
    p, q = corners(np.asarray(c1, float), yaw1, s1), corners(np.asarray(c2, float), yaw2, s2)
    for yaw in (yaw1, yaw2):
        for axis in (np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))]),
                     np.array([-math.sin(math.radians(yaw)), math.cos(math.radians(yaw))])):
            if (p @ axis).max() < (q @ axis).min() or (q @ axis).max() < (p @ axis).min():
                return False
    return True


def asset_size(asset: dict, rng) -> tuple[float, float, float]:
    """The asset's (length, width, height) in metres at a width drawn from its class's range (size_m), or its own
    converted size when it carries metres already."""
    if "size_m" in asset:
        return tuple(float(v) for v in asset["size_m"])
    L, W, H = (float(v) for v in asset["dims"])
    lo, hi = asset.get("width_range_m", (1.65, 1.95))
    k = float(rng.uniform(lo, hi)) / W
    return L * k, W * k, H * k


def plan_views(d: Path, pool: list[dict], n: int, rng, fov_deg: float, size, label: str,
               pitch_deg: float = 0.0, heading_deg: float | None = None) -> list[dict]:
    """n views of the sphere with their targets (render_plan.write's format)."""
    d = Path(d)
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    walls = geo.get("walls", [])
    captured = json.loads((d / "sphere.json").read_text())["captured"]
    pitches = pitch_range(captured, fov_deg, size)
    fits_path = d / "replica" / "fits.json"
    fits = [f for f in json.loads(fits_path.read_text()) if f.get("kept")] if fits_path.exists() else []
    kind_map = KindMap(d, R)
    kinds = allowed_kinds(kind_map, fits)
    ground = render_plan.Ground(d)
    views = []
    for k in range(n):
        yaw = float(rng.uniform(0, 360)) if heading_deg is None else \
            float(heading_deg + rng.uniform(-HEADING_SPREAD_DEG, HEADING_SPREAD_DEG))
        pitch = float(np.clip(pitch_deg + rng.uniform(-PITCH_SPREAD_DEG, PITCH_SPREAD_DEG), *pitches))
        dz = float(rng.uniform(-HEIGHT_SPREAD_M, HEIGHT_SPREAD_M))
        placed = []
        for _ in range(int(rng.integers(OBJECTS_PER_VIEW[0], OBJECTS_PER_VIEW[1] + 1))):
            o = place_one(ground, kind_map, kinds, fits, walls, pool, placed, rng, yaw, fov_deg, h)
            if o is not None:
                placed.append(o)
        objects = [{"usd": o["usd"], "length_m": o["size"][0], "size_m": [round(v, 3) for v in o["size"]],
                    "box_yaw_deg": round(o["yaw"], 2), "pos": o["pos"], "yaw_deg": o["render_yaw"],
                    "label": label, "uid": o["uid"], "key": f"v{k}_{i}", "distance_m": o["distance_m"],
                    "heading_from": o["heading_from"]} for i, o in enumerate(placed)]
        views.append({"name": f"x{k:03d}", "yaw_deg": round(yaw % 360, 2), "pitch_deg": round(pitch, 2),
                      "fov_deg": fov_deg, "size": list(size), "offset": [0.0, 0.0, round(dz, 3)],
                      "objects": objects})
    return views


def place_one(ground, kind_map, kinds, fits, walls, pool, placed, rng, view_yaw, fov_deg, h) -> dict | None:
    for _ in range(TRIES):
        heading = view_yaw + float(rng.uniform(-0.45, 0.45)) * fov_deg
        D = float(rng.uniform(*DISTANCE_M))
        points = ground.along(heading)
        if not points:
            continue
        horiz = np.array([math.hypot(p[0], p[2]) for p in points])
        i = int(np.argmin(np.abs(horiz - D)))
        if abs(horiz[i] - D) > 0.5:
            continue
        p = points[i]
        if kind_map.at(p) not in kinds:
            continue
        xz = np.array([p[0], p[2]])
        near = min(fits, key=lambda f: np.linalg.norm(np.asarray(f["centre_xz"]) - xz), default=None)
        if near is not None and np.linalg.norm(np.asarray(near["centre_xz"]) - xz) <= NEIGHBOUR_M:
            yaw = near["yaw_deg"] + float(rng.normal(0, HEADING_JITTER_DEG / 2))
            yaw += 180.0 if rng.random() < FLIP_SHARE else 0.0
            source = f"target {near['id']}"
        elif (axis := wall_axis(walls, xz)) is not None:
            yaw, source = axis + float(rng.normal(0, HEADING_JITTER_DEG / 2)) + 180.0 * int(rng.integers(2)), "wall"
        else:
            yaw, source = float(rng.uniform(0, 360)), "random"
        asset = pool[int(rng.integers(len(pool)))]
        size = asset_size(asset, rng)
        if any(overlaps(xz, yaw, size[:2], q["xz"], q["yaw"], q["size"][:2]) for q in placed):
            continue
        if not ground.footprint_free(p, size[0], size[1], yaw):
            continue
        on = [kind_map.at(q) in kinds for q in footprint_points(p, size[0], size[1], yaw)]
        if sum(on) < FOOTPRINT_ON_KIND * len(on):
            continue  # on the targets' kind of ground, not half on the pavement (SAM's votes are patchy)
        return {"usd": asset["usd"], "uid": asset["uid"], "size": size, "xz": xz, "yaw": yaw % 360,
                "render_yaw": (yaw + asset.get("front_flip_deg", 0.0)) % 360,
                "pos": render_plan.to_world(p, h).tolist(), "distance_m": round(float(np.linalg.norm(xz)), 2),
                "heading_from": source}
    return None


def full_area(view: dict, o: dict, height: float) -> float:
    """Pixels the object's box would cover in the view with nothing in front of it (from the capture point)."""
    from .replica import box_corners, silhouette
    centre = (o["pos"][0], o["pos"][1])
    sil = silhouette(view, box_corners(centre, o["box_yaw_deg"], o["size_m"], height - o["pos"][2]))
    return float(sil.sum())


MIN_VISIBLE_SHARE = 0.3   # of its box's silhouette: a car mostly hidden by another is not labelled


def labels(render_json: dict, plan: dict, min_px: int = MIN_VISIBLE_PX) -> dict:
    """Per view: the visible targets' tight boxes (x0, y0, x1, y1, pixels) from the render's instance maps; a
    target showing under MIN_VISIBLE_SHARE of its box's silhouette is left out (it stays in the image)."""
    out = {}
    report = {v["name"]: v for v in render_json["views"]}
    h = plan.get("camera_height_m")
    for v in plan["views"]:
        boxes = report.get(v["name"], {}).get("objects") or []
        keep = []
        for o, b in zip(v["objects"], boxes):
            if not b or b["pixels"] < min_px:
                continue
            share = b["pixels"] / max(full_area(v, o, h), 1.0) if h is not None and "size_m" in o else None
            if share is not None and share < MIN_VISIBLE_SHARE:
                continue
            keep.append({"label": o["label"], "box": b["box"], "pixels": b["pixels"], "uid": o["uid"],
                         "distance_m": o.get("distance_m"),
                         "visible_share": None if share is None else round(min(share, 1.0), 2)})
        out[v["name"]] = keep
    return out
