"""Shared bits of the teacher benchmark: the photos, the user's marks as truth, where predictions go.

Truth: the points the user placed in review of maceta_counter_8 (input/corrections.json), one per pot.
Predictions: runs/teacher_bench/<name>.json, {photo: [{"box": [x0, y0, x1, y1], "score": s}]} with the box
normalized to 0..1 of the (EXIF-rotated) photo.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "data" / "jobs" / "maceta_counter_8"
PHOTOS = JOB / "input" / "maceta_1d161a2d"
OUT = ROOT / "runs" / "teacher_bench"
SIDE = 1024  # every model sees the same downscaled photo, as the label stage sends it


def spec() -> dict:
    return json.loads((JOB / "spec.json").read_text())


def queries() -> list[str]:
    """Short English names of the object, as the interview wrote them for the asset search (no hand-tuning)."""
    return spec()["classes"][0]["assets"]["queries"]


def description() -> str:
    return spec()["classes"][0]["description"]


def truth() -> dict[str, list[tuple[float, float]]]:
    pts = json.loads((JOB / "input" / "corrections.json").read_text())["points"]
    return {k: [(p[0], p[1]) for p in v] for k, v in sorted(pts.items())}


def load(name: str, side: int = SIDE):
    from PIL import Image, ImageOps
    with Image.open(PHOTOS / name) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side))
        return im


def load_full(name: str):
    from PIL import Image, ImageOps
    with Image.open(PHOTOS / name) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def save(model: str, preds: dict, meta: dict | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}.json"
    p.write_text(json.dumps({"meta": meta or {}, "preds": preds}, indent=1))
    print(f"saved {p}")
    return p


def nms(boxes: list[dict], iou: float = 0.5) -> list[dict]:
    """Drops boxes that mostly overlap a higher-scoring one (the same pot found by two queries or tiles)."""
    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    def overlap(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        inter = max(0.0, w) * max(0.0, h)
        return inter / (area(a) + area(b) - inter + 1e-9)

    keep = []
    for b in sorted(boxes, key=lambda b: -b.get("score", 1.0)):
        if all(overlap(b["box"], k["box"]) < iou for k in keep):
            keep.append(b)
    return keep
