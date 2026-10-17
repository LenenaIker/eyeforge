"""S.6 of the sphere route, extrapolation (docs/roadmap_photosphere.md): new views turned around the capture point,
each with 1 to 5 targets placed where the sphere could hold them, for steps/render.py to draw. Alhaija et al.
(IJCV 2018): placing on the ground plane with the road's segmentation and a random heading works as well as by
hand; more than 5 cars per image does worse (docs/conocimiento_photosphere.md, placement).

Where: on the kinds of ground the erased targets stood on (their contacts' SAM 3 votes; a sphere without targets
takes the paved kinds), on ground the proxy measured (render_plan.Ground), with the whole footprint free and apart
from the others.
With a context prior from the user's photos (context.py; a sphere holding none of the targets, bins): their
class, height, ground, wall and distance, and their colour as paint (place_by_context).
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
               pitch_deg: float = 0.0, heading_deg: float | None = None, prior: dict | None = None) -> list[dict]:
    """n views of the sphere with their targets (render_plan.write's format). prior (context.prior, from the
    user's photos): targets placed, sized, labelled and painted as those photos show them, for a class the sphere
    holds none of; else as the erased targets stood (label: their class)."""
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
    colours = Colours(d, R) if prior else None
    views = []
    for k in range(n):
        yaw = float(rng.uniform(0, 360)) if heading_deg is None else \
            float(heading_deg + rng.uniform(-HEADING_SPREAD_DEG, HEADING_SPREAD_DEG))
        pitch = float(np.clip(pitch_deg + rng.uniform(-PITCH_SPREAD_DEG, PITCH_SPREAD_DEG), *pitches))
        dz = float(rng.uniform(-HEIGHT_SPREAD_M, HEIGHT_SPREAD_M))
        placed = []
        for _ in range(int(rng.integers(OBJECTS_PER_VIEW[0], OBJECTS_PER_VIEW[1] + 1))):
            o = place_by_context(ground, kind_map, walls, pool, prior, placed, rng, yaw, fov_deg, h, colours) if prior \
                else place_one(ground, kind_map, kinds, fits, walls, pool, placed, rng, yaw, fov_deg, h)
            if o is not None:
                placed.append(o)
        objects = [{"usd": o["usd"], "length_m": max(o["size"][:2]), "size_m": [round(v, 3) for v in o["size"]],
                    "box_yaw_deg": round(o["yaw"], 2), "pos": o["pos"], "yaw_deg": o["render_yaw"],
                    "label": o.get("label", label), "uid": o["uid"], "key": f"v{k}_{i}",
                    "distance_m": o["distance_m"], "heading_from": o["heading_from"],
                    **({"paint": o["paint"]} if o.get("paint") else {})} for i, o in enumerate(placed)]
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


def srgb_to_linear(c) -> list[float]:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).round(4).tolist()


def sized(asset: dict, height_m: float) -> tuple[float, float, float]:
    """The asset's (length, width, height) in metres scaled to a height."""
    L, W, H = (float(v) for v in asset["dims"])
    k = height_m / H
    return L * k, W * k, height_m


def pick_sized(pool: list[dict], height_m: float, rng, among: int = 4) -> dict:
    """Of a few assets drawn at random, the one whose own real height (pool "height_m", the vision model's) is
    nearest the wanted one: a 0.4 m bucket takes a small bin's shape rather than a dumpster's."""
    cands = [pool[int(i)] for i in rng.choice(len(pool), min(among, len(pool)), replace=False)]
    return min(cands, key=lambda a: abs(math.log(a.get("height_m", height_m) / height_m)))


def wall_spot(ground, walls, along_m, depth_m, gap_m, rng, view_yaw, fov_deg, dist_range):
    """A spot against a wall of the proxy for a footprint along_m wide and depth_m deep, in the view and at a wanted
    distance: (levelled ground point, the wall's normal towards the camera as a yaw) or None. gap_m: from the wall
    to the footprint's centre, as measured in the user's photos (never less than half its depth)."""
    w = walls[int(rng.integers(len(walls)))]
    a, b = (np.asarray(e, float) for e in w["ends_xz"])
    run = np.linalg.norm(b - a)
    if run < along_m + 0.4:
        return None
    u = (b - a) / run
    n = np.asarray(w["normal"], float)[[0, 2]]
    n = n / np.linalg.norm(n)
    if n @ (-a) < 0:  # the side the camera is on
        n = -n
    t = float(rng.uniform(along_m / 2 + 0.2, run - along_m / 2 - 0.2))
    xz = a + t * u + n * max(gap_m, depth_m / 2 + 0.03)
    D = float(np.linalg.norm(xz))
    off = (math.degrees(math.atan2(xz[0], xz[1])) - view_yaw + 180) % 360 - 180
    if not dist_range[0] <= D <= dist_range[1] or abs(off) > 0.45 * fov_deg:
        return None
    ray = np.array([xz[0], ground.h, xz[1]])
    ok, dist = ground.look(ray[None])
    if not ok[0] or abs(dist[0] - np.linalg.norm(ray)) > 0.1 * np.linalg.norm(ray):
        return None
    p = ray / np.linalg.norm(ray) * dist[0]
    return p, math.degrees(math.atan2(n[1], n[0]))


def at_edge(ground, kind_map, q, kind, radius_m, n_dirs: int = 16) -> bool:
    """Whether the edge of q's ground (another kind, or no ground: a kerb, a wall, a lawn's border) comes within
    radius_m of it."""
    for a in np.linspace(0, 2 * math.pi, n_dirs, endpoint=False):
        r = q + radius_m * np.array([math.cos(a), 0.0, math.sin(a)])
        ok, dist = ground.look(r[None])
        if not ok[0] or abs(dist[0] - np.linalg.norm(r)) > 0.1 * np.linalg.norm(r) or kind_map.at(r) != kind:
            return True
    return False


class Colours:
    """The albedo the render gives the proxy under a levelled point: the erased panorama's linear colour over the
    light, at render_plan.proxy_colors' scale (its vertices' 99.5th percentile to ALBEDO_MAX)."""

    def __init__(self, d: Path, R: np.ndarray):
        from PIL import Image
        d = Path(d)
        self.background = np.asarray(Image.open(d / "background.jpg").convert("RGB"))
        self.light = np.maximum(np.asarray(render_plan.sky_color(self.background, np.load(d / "stuff.npz")["sky"]),
                                           np.float32), 1e-3)
        self.R = R
        v = np.load(d / "proxy.npz")["vertices"]
        col = self.linear(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)) / self.light
        self.k = render_plan.ALBEDO_MAX / max(float(np.percentile(col, 99.5)), 1e-6)

    def linear(self, dirs: np.ndarray) -> np.ndarray:
        return render_plan.srgb_to_linear(capture.sample(self.background, render_plan.pano_rays(dirs, self.R)))

    def at(self, p_level: np.ndarray) -> np.ndarray:
        d = np.asarray(p_level, float)[None] / np.linalg.norm(p_level)
        return np.clip(self.linear(d)[0] / self.light * self.k, 0, render_plan.ALBEDO_MAX)


def paint_for(p: dict, ground_lin: np.ndarray) -> list[float] | None:
    """A class's paint in this sphere: its chroma at its measured brightness over the ground next to it, times the
    sphere's ground there (photometry by proportions to the floor, as flow_photometry: the user's sunny photos
    and an overcast sphere differ in light, not in the bin's share of the ground's brightness)."""
    if not p.get("chroma") or not p.get("to_ground"):
        return srgb_to_linear(p["colour_srgb"]) if p.get("colour_srgb") else None
    g = float(ground_lin @ np.array([0.2126, 0.7152, 0.0722]))
    return np.clip(np.asarray(p["chroma"]) * p["to_ground"] * g, 0.0, 0.9).round(4).tolist()


def place_by_context(ground, kind_map, walls, pool, prior, placed, rng, view_yaw, fov_deg, h,
                     colours: "Colours | None" = None) -> dict | None:
    """One target placed as the user's photos show them (context.prior): its class drawn by how often it stands in
    them, its height one of theirs, on their kinds of ground, against a wall as often as theirs are (front away
    from it), at their distances from the camera, painted their colour."""
    classes = [c for c, p in prior.items() if p.get("standing") and p.get("heights")]
    if not classes:
        return None
    weights = np.array([prior[c]["standing"] for c in classes], float)
    cls = classes[int(rng.choice(len(classes), p=weights / weights.sum()))]  # once: a class hard to place still
    p = prior[cls]                                                          # keeps its share
    # Distances: how the user frames their photos, not a class's own (a few photos each): all classes together.
    ranges = [prior[c]["distance_m"] for c in classes if prior[c].get("distance_m")]
    lo, hi = (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else DISTANCE_M
    dist_range = (max(lo, 1.5), max(hi, lo + 2.0))
    kinds = {k for k in p.get("ground") or {} if k in kind_map.kinds} or set(PAVED)
    against = bool(walls) and rng.random() < (p.get("wall_share") or 0.0)
    for t in range(TRIES):
        height = float(rng.choice(p["heights"]))
        asset = pick_sized(pool, height, rng)
        size = sized(asset, height)
        front = None
        if against and t < TRIES // 2:  # no wall at the user's distances in this view: it stands free
            turned = round(asset.get("front_deg", 0.0) / 90) % 2  # its front along its width, not its length
            along, depth = (size[0], size[1]) if turned else (size[1], size[0])
            spot = wall_spot(ground, walls, along, depth, p.get("wall_gap_m") or 0.1, rng, view_yaw, fov_deg,
                             dist_range)
            if spot is None:
                continue
            q, front = spot
            source = "wall"
        else:
            heading = view_yaw + float(rng.uniform(-0.45, 0.45)) * fov_deg
            D = float(rng.uniform(*dist_range))
            points = ground.along(heading)
            if not points:
                continue
            horiz = np.array([math.hypot(r[0], r[2]) for r in points])
            i = int(np.argmin(np.abs(horiz - D)))
            if abs(horiz[i] - D) > 0.5:
                continue
            q = points[i]
            front, source = float(rng.uniform(0, 360)), "free"
        kind = kind_map.at(q)
        if source == "free":
            if kind not in kinds:
                continue
            gaps = p.get("edge_gaps") or []
            if gaps and rng.random() < (p.get("edge_share") or 0.0):  # at its ground's edge, as in the photos
                if not at_edge(ground, kind_map, q, kind, max(float(rng.choice(gaps)), min(size[:2]) / 2) + 0.15):
                    continue
        render_yaw = (front - asset.get("front_deg", 0.0)) % 360  # the asset at render_yaw faces front_deg more
        xz = np.array([q[0], q[2]])
        if any(overlaps(xz, render_yaw, size[:2], o["xz"], o["yaw"], o["size"][:2]) for o in placed):
            continue
        if not ground.footprint_free(q, size[0], size[1], render_yaw):
            continue
        if source == "free":
            on = [kind_map.at(r) in kinds for r in footprint_points(q, size[0], size[1], render_yaw)]
            if sum(on) < FOOTPRINT_ON_KIND * len(on):
                continue
        return {"usd": asset["usd"], "uid": asset["uid"], "size": size, "xz": xz, "yaw": render_yaw,
                "render_yaw": render_yaw, "pos": render_plan.to_world(q, h).tolist(),
                "distance_m": round(float(np.linalg.norm(xz)), 2), "heading_from": source, "label": cls,
                "paint": paint_for(p, colours.at(q)) if colours else srgb_to_linear(p["colour_srgb"])}
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
