"""S.6 of the sphere route, context: where the targets stand in the user's own photos, for placing them in a sphere
that shows none (the bins of 2026-10-17: none of the four spheres has one). Placement by context learnt from real
images is what made pasted objects help a detector (Dvornik et al., ECCV 2018: random places hurt, places from a
context model helped); here the context is measured on the user's photos, not written per class.

For every target the teacher marked (label/verdicts.json), from MoGe-2's metric points (steps/photo_depth.py) and
SAM 3's ground kinds and built surfaces (steps/photo_stuff.py): the ground plane, the target's base and top above
it (a base well above the ground hangs from a wall or post), the kind of ground just under it, how far its base is
from the nearest wall, its distance from the camera and its colour (median of the box's centre). The local vision
model was tried first and called every bin on a hanging rack standing (2026-10-17).

Per class (prior): {"n", "mounted", "ground": {kind: share}, "wall_share", "wall_gap_m", "height_m",
"distance_m": [p10, p90], "edge_share", "edge_gaps", "to_ground": its luminance over the ground's next to it,
"chroma": its linear colour over its luminance,
"colour_srgb"}; standing targets only, but for "mounted", "n" and the colours.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import GROUND_KINDS

CENTRE = 0.4          # of a box's width and height around its centre: the colour (the box also holds background)
CORE_W = 0.6          # of a box's width: the target's own points (its sides show what stands behind)
MOUNTED_M = 0.2       # a base this far above the ground hangs (a wheelie bin's wheels touch it)
AGAINST_M = 0.6       # a wall this close to the base: the target stands against it
UNDER = 0.03          # of the photo's height under a box: the ground it stands on
EDGE_M = 1.0          # the edge of its ground (kerb, lawn border, wall) this close: the target stands at it
PLANE_TOL_M = 0.05
PLANE_ITER = 300


def fit_ground(points: np.ndarray, rng) -> tuple[np.ndarray, float] | None:
    """RANSAC plane on ground points: (n, d) with n . p + d the height above it (the camera above: d > 0)."""
    if len(points) < 50:
        return None
    best, best_in = None, 0
    for _ in range(PLANE_ITER):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        inl = int((np.abs(points @ n + d) < PLANE_TOL_M).sum())
        if inl > best_in:
            best, best_in = (n, d), inl
    if best is None:
        return None
    n, d = best
    inl = points[np.abs(points @ n + d) < PLANE_TOL_M]
    centre = inl.mean(0)
    n = np.linalg.svd(inl - centre, full_matrices=False)[2][-1]
    d = -n @ centre
    return (n, d) if d > 0 else (-n, -d)


def lin(c) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luma(c) -> float:
    return float(np.asarray(c) @ (0.2126, 0.7152, 0.0722))


def colour(img: Image.Image, box: list[float]) -> np.ndarray:
    """Median sRGB (0-1) of the box's centre."""
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    H, W = a.shape[:2]
    cx, cy, bw, bh = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H, (box[2] - box[0]) * W, (box[3] - box[1]) * H
    x0, x1 = int(cx - CENTRE * bw / 2), int(cx + CENTRE * bw / 2) + 1
    y0, y1 = int(cy - CENTRE * bh / 2), int(cy + CENTRE * bh / 2) + 1
    return np.median(a[max(y0, 0):y1, max(x0, 0):x1].reshape(-1, 3), axis=0)


def measure_photo(img: Image.Image, depth: dict, stuff: dict, objects: list[dict], rng) -> list[dict]:
    pts, valid = depth["points"], depth["mask"]
    H, W = valid.shape
    kinds = [k for k in GROUND_KINDS if k in stuff]
    ground = np.any([stuff[k] for k in kinds], 0) & valid
    inside = np.zeros((H, W), bool)
    for o in objects:
        b = o["box"]
        inside[int(b[1] * H):int(b[3] * H) + 1, int(b[0] * W):int(b[2] * W) + 1] = True
    plane = fit_ground(pts[ground & ~inside], rng)
    out = []
    for o in objects:
        b = o["box"]
        rec = {"class": o["class"], "box": b, "colour": [round(float(c), 4) for c in colour(img, b)]}
        if plane is None:
            out.append(rec)
            continue
        n, d = plane
        cx = (b[0] + b[2]) / 2
        x0, x1 = int((cx - CORE_W * (b[2] - b[0]) / 2) * W), int((cx + CORE_W * (b[2] - b[0]) / 2) * W) + 1
        y0, y1 = int(b[1] * H), int(b[3] * H) + 1
        core = np.zeros((H, W), bool)
        core[y0:y1, x0:x1] = True
        own = core & valid & ~ground
        if own.sum() < 20:
            out.append(rec)
            continue
        p = pts[own]
        h = p @ n + d
        base_h, top_h = float(np.percentile(h, 3)), float(np.percentile(h, 98))
        low = p[h <= base_h + 0.15]
        foot = low.mean(0) - (low.mean(0) @ n + d) * n  # the base's centre dropped onto the ground
        rec.update(base_m=round(base_h, 3), height_m=round(top_h - base_h, 3), mounted=base_h > MOUNTED_M,
                   distance_m=round(float(np.linalg.norm(foot + d * n)), 2))  # from the camera's foot
        band = np.zeros((H, W), bool)
        band[min(y1, H - 1):min(y1 + int(UNDER * H) + 1, H), x0:x1] = True
        votes = {k: int((stuff[k] & band).sum()) for k in kinds if k != "floor"}
        if not any(votes.values()) and "floor" in kinds:
            votes = {"floor": int((stuff["floor"] & band).sum())}
        rec["ground"] = max(votes, key=votes.get) if any(votes.values()) else None
        a = np.asarray(img, np.float32) / 255.0
        under = band & ground
        if under.sum() >= 10:  # the ground's colour next to it: the target's colour is kept as a ratio to it,
            # which the sphere's own ground turns back into a colour under its light (photometry by proportions)
            rec["ground_colour"] = [round(float(c), 4) for c in np.median(a[under], axis=0)]
        flat = lambda q: q - np.outer(q @ n + d, n)  # noqa: E731 (onto the ground plane)
        if rec["ground"] and rec["ground"] in stuff:  # the edge of its ground: a kerb, a lawn's border, a wall
            whole = np.zeros((H, W), bool)
            whole[y0:y1, int(b[0] * W):int(b[2] * W) + 1] = True
            other = valid & ~stuff[rec["ground"]] & ~whole
            near = pts[other]
            nh = near @ n + d
            near = near[nh < 2.0]
            if len(near):
                rec["edge_gap_m"] = round(float(np.min(np.linalg.norm(flat(near) - foot, axis=1))), 2)
        if "structure" in stuff:
            wall = pts[stuff["structure"] & valid & ~core]
            if len(wall):
                wh = wall @ n + d
                wall = wall[(wh > 0.2) & (wh < 2.0)]
            if len(wall):
                rec["wall_gap_m"] = round(float(np.min(np.linalg.norm(flat(wall) - foot, axis=1))), 2)
        out.append(rec)
    return out


def measure(photos: Path, verdicts: dict, depth_dir: Path, stuff_dir: Path, out: Path, seed: int = 0) -> dict:
    """{photo: [per target: class, box, colour, base_m, height_m, mounted, distance_m, ground, wall_gap_m]}."""
    rng = np.random.default_rng(seed)
    res = {}
    for name, v in sorted(verdicts.items()):
        objs, stem = v.get("objects") or [], Path(name).stem
        if not objs or not (depth_dir / f"{stem}.npz").exists() or not (stuff_dir / f"{stem}.npz").exists():
            continue
        depth, stuff = dict(np.load(depth_dir / f"{stem}.npz")), dict(np.load(stuff_dir / f"{stem}.npz"))
        img = ImageOps.exif_transpose(Image.open(photos / name)).convert("RGB")
        img = img.resize(depth["mask"].shape[::-1])
        res[name] = measure_photo(img, depth, stuff, objs, rng)
    out.write_text(json.dumps(res, indent=1))
    return res


def prior(measured: dict) -> dict:
    by = defaultdict(list)
    for marks in measured.values():
        for m in marks:
            by[m["class"]].append(m)
    out = {}
    for cls, ms in sorted(by.items()):
        known = [m for m in ms if "mounted" in m]
        standing = [m for m in known if not m["mounted"]]
        ground = Counter(m["ground"] for m in standing if m.get("ground"))
        gaps = [m["wall_gap_m"] for m in standing if "wall_gap_m" in m]
        dist = [m["distance_m"] for m in standing]
        edges = [m["edge_gap_m"] for m in standing if "edge_gap_m" in m]
        ratios = [luma(lin(m["colour"])) / max(luma(lin(m["ground_colour"])), 1e-3) for m in ms
                  if m.get("ground_colour")]
        chroma = np.median([lin(m["colour"]) / max(luma(lin(m["colour"])), 1e-3) for m in ms], axis=0)
        out[cls] = {"n": len(ms), "measured": len(known), "standing": len(standing),
                    "heights": [m["height_m"] for m in standing],
                    "mounted": round(1 - len(standing) / len(known), 2) if known else None,
                    "ground": {k: round(c / sum(ground.values()), 2) for k, c in ground.most_common()},
                    "wall_share": round(sum(g < AGAINST_M for g in gaps) / len(gaps), 2) if gaps else None,
                    "wall_gap_m": round(float(np.median([g for g in gaps if g < AGAINST_M])), 2)
                    if any(g < AGAINST_M for g in gaps) else None,
                    "height_m": round(float(np.median([m["height_m"] for m in standing])), 2) if standing else None,
                    "distance_m": [round(float(np.percentile(dist, 10)), 1), round(float(np.percentile(dist, 90)), 1)]
                    if dist else None,
                    "edge_share": round(sum(g < EDGE_M for g in edges) / len(edges), 2) if edges else None,
                    "edge_gaps": sorted(g for g in edges if g < EDGE_M),
                    "to_ground": round(float(np.median(ratios)), 3) if ratios else None,
                    "chroma": [round(float(c), 3) for c in chroma],
                    "colour_srgb": [round(float(c), 3) for c in np.median([m["colour"] for m in ms], axis=0)]}
    return out


# Rigs: what holds the hanging targets (support.py names it, steps/photo_holders.py segments it) and how the
# targets hang on it, measured as one layout to copy into a sphere: the holders and their targets in a frame of
# their own (x along them, y towards the camera, z up from the ground).
LINK_M = 0.35         # a target this close (on the ground plane) to a holder's points hangs on it
BATTERY_M = 1.0       # holders this close to each other are one rig (a row of rails on one wall)
MIN_HOLDER_PX = 40
ERODE_PX = 2          # a holder's mask is shrunk this much before measuring: its rim mixes depth with what is behind


def erode(m: np.ndarray, px: int) -> np.ndarray:
    """Binary erosion by a square of side 2 px + 1 (numpy only)."""
    out = m.copy()
    for _ in range(px):
        e = out.copy()
        e[1:] &= out[:-1]
        e[:-1] &= out[1:]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        out = e
    return out


def _own_points(pts, valid, ground, box):
    """The target's own points: the core of its box, without the ground (as measure_photo)."""
    H, W = valid.shape
    cx = (box[0] + box[2]) / 2
    core = np.zeros((H, W), bool)
    core[int(box[1] * H):int(box[3] * H) + 1,
         int((cx - CORE_W * (box[2] - box[0]) / 2) * W):int((cx + CORE_W * (box[2] - box[0]) / 2) * W) + 1] = True
    return pts[core & valid & ~ground]


WALL_NEAR_M = 1.5     # wall points this close to a rig give its axis


def rig_axes(o, wall, up, d):
    """A rig's horizontal axes: x along the wall behind it (the main direction of the wall's points within
    WALL_NEAR_M), else across the camera's view; y towards the camera's side."""
    cam = -d * up  # the camera's foot on the ground
    toward = cam - o
    toward -= (toward @ up) * up
    toward /= max(np.linalg.norm(toward), 1e-9)
    x = np.cross(up, toward)
    if wall is not None and len(wall):
        near = wall[np.linalg.norm(wall - o, axis=1) < WALL_NEAR_M]
        if len(near) >= 50:
            q = near - near.mean(0)
            q -= np.outer(q @ up, up)
            x = np.linalg.svd(q, full_matrices=False)[2][0]
            x -= (x @ up) * up
            x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(x, up)
    if y @ toward < 0:
        y, x = -y, -x
    return x, y


def rigs_photo(img: Image.Image, depth: dict, stuff: dict, holders: dict, targets: list[dict], rng) -> list[dict]:
    """The rigs of one photo: [{"holders": [{"x", "y", "w", "d", "bottom", "top", "colour", "chroma", "to_ground"}],
    "targets": [{"class",
    "x", "y", "base_m", "height_m"}], "standing": every holder reaches the ground, "wall_gap_m"}], only rigs that
    hold a target."""
    pts, valid = depth["points"], depth["mask"]
    kinds = [k for k in GROUND_KINDS if k in stuff]
    ground = np.any([stuff[k] for k in kinds], 0) & valid
    plane = fit_ground(pts[ground], rng)
    if plane is None or not len(holders.get("masks", [])):
        return []
    n, d = plane
    flat = lambda q: q - np.outer(q @ n + d, n)  # noqa: E731
    a = np.asarray(img, np.float32) / 255.0
    g_lum = luma(lin(np.median(a[ground], axis=0))) if ground.any() else None
    hs = []
    for m in holders["masks"]:
        m = erode(m, ERODE_PX) & valid
        if m.sum() < MIN_HOLDER_PX:
            continue
        p = pts[m]
        h = p @ n + d
        hs.append({"flat": flat(p), "bottom": float(np.percentile(h, 3)), "top": float(np.percentile(h, 97)),
                   "colour": np.median(a[m], axis=0)})
    links = {}
    for ti, t in enumerate(targets):
        if not t.get("mounted"):
            continue
        p = _own_points(pts, valid, ground, t["box"])
        if len(p) < 20:
            continue
        c = flat(p).mean(0)
        # within reach, the biggest holder: the rail, not a bolt on it (a 130-pixel bracket was nearer than the
        # bin's own 8000-pixel rail, 2026-10-17)
        reach = [(len(hh["flat"]), hi) for hi, hh in enumerate(hs)
                 if hh["bottom"] - 0.3 < t["base_m"] < hh["top"] + 0.3
                 and float(np.min(np.linalg.norm(hh["flat"] - c, axis=1))) < LINK_M]
        if reach:
            links[ti] = (max(reach)[1], c)
    used = sorted({hi for hi, _ in links.values()})
    # holders one rig when close: single link on their centres
    centre = {hi: np.median(hs[hi]["flat"], axis=0) for hi in used}
    group = {hi: hi for hi in used}
    for i in used:
        for j in used:
            if i < j and np.linalg.norm(centre[i] - centre[j]) < BATTERY_M:
                gi, gj = group[i], group[j]
                for k in used:
                    if group[k] == gj:
                        group[k] = gi
    up = n
    wall = None
    if "structure" in stuff:
        wall = pts[stuff["structure"] & valid]
        wh = wall @ n + d
        wall = flat(wall[(wh > 0.2) & (wh < 2.0)])
    rigs = []
    for g in sorted(set(group.values())):
        members = [hi for hi in used if group[hi] == g]
        o = np.mean([centre[hi] for hi in members], axis=0)
        x, y = rig_axes(o, wall, up, d)
        loc = lambda q: (float((q - o) @ x), float((q - o) @ y))  # noqa: E731
        rig = {"holders": [], "targets": [], "standing": True}
        for hi in members:
            hh = hs[hi]
            px, py = (hh["flat"] - o) @ x, (hh["flat"] - o) @ y
            rig["holders"].append({
                "x": round(float(np.median(px)), 3), "y": round(float(np.median(py)), 3),
                "w": round(max(float(np.percentile(px, 95) - np.percentile(px, 5)), 0.03), 3),
                # depth from the middle half of its points: across the view MoGe's depth is the noisy axis
                "d": round(max(2 * float(np.percentile(py, 75) - np.percentile(py, 25)), 0.03), 3),
                "bottom": round(max(hh["bottom"], 0.0), 3), "top": round(hh["top"], 3),
                "colour": [round(float(c), 4) for c in hh["colour"]],
                "chroma": [round(float(c), 3) for c in lin(hh["colour"]) / max(luma(lin(hh["colour"])), 1e-3)],
                # its brightness over the photo's ground, for the sphere's paint (photometry by proportions)
                "to_ground": round(luma(lin(hh["colour"])) / max(g_lum, 1e-3), 3) if g_lum else None})
            rig["standing"] &= hh["bottom"] < MOUNTED_M
        for ti, (hi, c) in links.items():
            if hi in members:
                t = targets[ti]
                tx, ty = loc(c)
                rig["targets"].append({"class": t["class"], "x": round(tx, 3), "y": round(ty, 3),
                                       "base_m": t["base_m"], "height_m": t["height_m"]})
        if wall is not None and len(wall):
            rig["wall_gap_m"] = round(float(np.min(np.linalg.norm(wall - o, axis=1))), 2)
        rigs.append(rig)
    return rigs


def measure_rigs(photos: Path, measured: dict, depth_dir: Path, stuff_dir: Path, holder_dir: Path, out: Path,
                 seed: int = 0) -> dict:
    """{photo: [rig]} for the photos with hanging targets and holders found (rigs_photo)."""
    res = {}
    for name, targets in sorted(measured.items()):
        stem = Path(name).stem
        if not any(t.get("mounted") for t in targets) or not (holder_dir / f"{stem}.npz").exists():
            continue
        depth, stuff = dict(np.load(depth_dir / f"{stem}.npz")), dict(np.load(stuff_dir / f"{stem}.npz"))
        holders = dict(np.load(holder_dir / f"{stem}.npz"))
        img = ImageOps.exif_transpose(Image.open(photos / name)).convert("RGB").resize(depth["mask"].shape[::-1])
        rigs = rigs_photo(img, depth, stuff, holders, targets, np.random.default_rng(seed))
        if rigs:
            res[name] = rigs
    out.write_text(json.dumps(res, indent=1))
    return res
