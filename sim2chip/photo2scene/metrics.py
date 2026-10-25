"""Cheap numbers per built scene, to catch a regression without looking at sheets (2026-10-15: photo68 lost its 21
plants and church_corner's copies floated, both visible only by eye). Read from what the flow already writes; no
model is called. `python -m sim2chip.photo2scene.metrics <before root> <after root> [names]` prints both side by side.

- up_floor_deg: angle between the scene's up and the ground's normal (pieces.json). A real street seldom slopes
  over 5 degrees; more is usually a wrong up, and every building then stands off its floor.
- mean_de: colour difference to the photo from its own camera (photometry_final.json).
- bands_no_species: vegetation bands of the street section with no species from the VLM (or a default one).
- plants: NVIDIA plants in the extended place; plant_cover: the photo's share of trees and shrubs.
- float_max_m: over the buildings of the extended place, the most that one end of a base stands over the ground
  (the lowest point of each eighth of the building along its length, worst eighth minus best).
- views_kept: share of new views the view check kept (judge-dependent: compare only runs with the same judge).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VEG_KINDS = {"tree", "hedge", "shrub"}


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def base_float(pts: np.ndarray, up: np.ndarray, bins: int = 8) -> float:
    """How far one end of a building's base stands over its other end: along its longest horizontal axis, the
    lowest point in each of `bins` slices, the highest of those minus the lowest."""
    h = pts @ up
    flat = pts - np.outer(h, up)
    flat -= flat.mean(0)
    axis = np.linalg.svd(flat[:: max(1, len(flat) // 5000)], full_matrices=False)[2][0]
    s = flat @ axis
    edges = np.linspace(s.min(), s.max(), bins + 1)
    lows = [h[(s >= a) & (s <= b)].min() for a, b in zip(edges[:-1], edges[1:]) if ((s >= a) & (s <= b)).any()]
    return float(max(lows) - min(lows)) if lows else 0.0


def scene(d: Path) -> dict:
    out = {}
    pieces = _json(d / "pieces.json")
    if pieces and isinstance(pieces.get("up"), dict):
        out["up_floor_deg"] = pieces["up"].get("tilt_deg")
    phot = _json(d / "photometry_final.json")
    if phot:
        out["mean_de"] = phot.get("mean_de")
    sec = _json(d / "section.json")
    if sec:
        veg = [b for b in sec.get("bands", []) if b.get("kind") in VEG_KINDS]
        out["bands_no_species"] = f"{sum(1 for b in veg if not b.get('pick') or b.get('pick_default'))}/{len(veg)}"
        cover = sec.get("image_cover", {})
        out["plant_cover"] = round(cover.get("tree", 0) + cover.get("shrub", 0), 3)
    ext = d / "extended" / "isaac"
    sj = _json(ext / "scene.json")
    if sj and (ext / "meshes.npz").exists():
        arr = np.load(ext / "meshes.npz")
        up = np.asarray(sj["up"], float)
        up /= np.linalg.norm(up)
        names = {e["index"]: e["name"] for e in sj.get("env", [])}
        out["plants"] = sum(1 for k in arr.files if k.endswith("_usd"))
        floats = []
        from .steps.standing import is_building
        for i, t in sj["views"][0]["objects"]:
            if f"obj{i}_points" not in arr.files or f"obj{i}_usd" in arr.files:
                continue
            if not is_building(names.get(i, "")) and np.ptp(arr[f"obj{i}_points"] @ up) < 3:
                continue
            t = np.asarray(t)
            floats.append(base_float(arr[f"obj{i}_points"] @ t[:3, :3].T + t[:3, 3], up))
        out["float_max_m"] = round(max(floats), 2) if floats else None
    views = _json(d / "extended" / "views.json")
    if views:
        vs = views.get("views", [])
        out["views_kept"] = f"{sum(v['keep'] for v in vs)}/{len(vs)}"
    return out


def main(argv: list[str]) -> None:
    before, after = Path(argv[0]), Path(argv[1])
    names = argv[2:] or sorted(p.name for p in after.iterdir() if p.is_dir() and (p / "pieces.json").exists())
    keys = ["up_floor_deg", "float_max_m", "mean_de", "bands_no_species", "plants", "plant_cover", "views_kept"]
    print(f"{'scene':20s}" + "".join(f"{k:>22s}" for k in keys))
    for n in names:
        a, b = scene(before / n), scene(after / n)
        print(f"{n:20s}" + "".join(f"{str(a.get(k, '-')) + ' > ' + str(b.get(k, '-')):>22s}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1:])
