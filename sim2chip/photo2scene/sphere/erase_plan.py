"""Where the sphere route's eraser looks (steps/erase.py): the objects grouped so that each group fits one view,
and that view, centred on the group and just wide enough for it. Pure Python, so it is tested in the main env.
Objects are instances.json's: centre (lon, lat) and angular extent (width, height), degrees.
"""
from __future__ import annotations

import math

MARGIN_DEG = 6.0       # room around a group in its view
MIN_FOV_DEG = 40.0
MAX_FOV_DEG = 100.0    # wider views stretch their edges too much for the eraser; a wider group is split
MAX_SIDE = 1344        # the eraser's working size (about a megapixel at 4:3)
MIN_SIDE = 768         # under this the eraser works badly; a small view is sampled finer than the panorama


def group(objects: list[dict]) -> list[list[dict]]:
    """Greedy groups whose angular boxes, together, fit a view of MAX_FOV_DEG: largest objects first, each joining
    the first group it fits in and overlaps or nearly touches."""
    groups = []
    for o in sorted(objects, key=lambda o: -o["area_share"]):
        for g in groups:
            box = union_box(g + [o])
            near = any(abs((o["lon"] - p["lon"] + 180) % 360 - 180) < (o["width"] + p["width"]) / 2 + MARGIN_DEG
                       for p in g)
            if near and box["width"] + 2 * MARGIN_DEG <= MAX_FOV_DEG and \
                    box["height"] + 2 * MARGIN_DEG <= vfov(MAX_FOV_DEG):
                g.append(o)
                break
        else:
            groups.append([o])
    return groups


def union_box(objs: list[dict]) -> dict:
    c = objs[0]["lon"]
    lo = [((o["lon"] - o["width"] / 2 - c + 180) % 360 - 180) for o in objs]
    hi = [((o["lon"] + o["width"] / 2 - c + 180) % 360 - 180) for o in objs]
    top = max(o["lat"] + o["height"] / 2 for o in objs)
    bottom = min(o["lat"] - o["height"] / 2 for o in objs)
    return {"lon": c + (max(hi) + min(lo)) / 2, "lat": (top + bottom) / 2, "width": max(hi) - min(lo),
            "height": top - bottom}


def vfov(hfov: float, aspect: float = 3 / 4) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * aspect))


def view_for(objs: list[dict], px_per_deg: float) -> dict:
    """A 4:3 view centred on the group, wide enough for it with MARGIN_DEG around (no tilt about its axis), at the
    panorama's density (px_per_deg) so it neither loses nor invents detail, up to MAX_SIDE."""
    box = union_box(objs)
    fov = max(MIN_FOV_DEG, box["width"] + 2 * MARGIN_DEG)
    while vfov(fov) < box["height"] + 2 * MARGIN_DEG and fov < MAX_FOV_DEG:
        fov += 2
    fov = min(fov, MAX_FOV_DEG)
    w = min(MAX_SIDE, max(MIN_SIDE, int(round(fov * px_per_deg / 16)) * 16))
    return {"yaw_deg": round(box["lon"] % 360, 2), "pitch_deg": round(-box["lat"], 2), "fov_deg": round(fov, 1),
            "size": [w, int(round(w * 3 / 4 / 16)) * 16]}
