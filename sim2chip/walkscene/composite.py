"""The arithmetic of isaac.py, without Isaac (so the tests run it on the CPU): which prim each object takes, the
differential composite of the two passes over the splat's background, and the boxes of what shows."""
from __future__ import annotations

from collections import Counter

import numpy as np

SHADOW_BLUR_PX = 2.0      # the passes are smoothed this much before their ratio
DARK = 6.0                # B under this (0-255) is too dark for a ratio
ON_GROUND = 0.08          # the splat's surface within this share of the plane's depth is the ground
OCCLUDE = 0.03            # an object pixel this share nearer than the splat shows
LIT_MIN_PX = 100          # fewer lit ground pixels than this keep the last view's gain


def srgb_to_linear(c):
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def smooth(x):
    """Separable Gaussian blur of an (H, W, C) float image, edges repeated."""
    r = int(3 * SHADOW_BLUR_PX)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SHADOW_BLUR_PX) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge")
    rows = sum(k[i] * pad[i:i + x.shape[0]] for i in range(2 * r + 1))
    return sum(k[i] * rows[:, i:i + x.shape[1]] for i in range(2 * r + 1))


def slots(view) -> list[tuple[str, int]]:
    """The prim each object of the view takes: (model, copy). Two objects of one model (counting) are two prims."""
    seen = Counter()
    out = []
    for o in view["objects"]:
        out.append((o["key"], seen[o["key"]]))
        seen[o["key"]] += 1
    return out


def copies(views) -> Counter:
    """Prims per model: as many as the view that holds most objects of it."""
    n = Counter()
    for v in views:
        for key, j in slots(v):
            n[key] = max(n[key], j + 1)
    return n


def object_index(ids, labels: dict, prims: dict[str, int]) -> np.ndarray:
    """Per pixel, 1 + the object it shows (0: none), from Replicator's instance segmentation (ids, and their prim
    paths in labels) and each object's prim path (prims: path -> 1 + its index in the view)."""
    index = np.zeros(np.asarray(ids).shape, np.uint8)
    for i, path in labels.items():
        path = path if isinstance(path, str) else str(path)
        for root, k in prims.items():
            if path == root or path.startswith(root + "/"):
                index[ids == int(i)] = k
    return index


def composite(a, da, b, db, bg, sd, index, last_gain):
    """Pass A (with the objects) and B (without), their depths, the splat's background and depth (0: nothing
    there) -> (out, obj, index, lit, gain). An object pixel shows where it is nearer than the splat; the background
    keeps the splat times the shadow ratio A/B on the ground; the objects take A at the gain that maps B to the
    background on the lit ground (the last view's when too little ground is lit)."""
    obj = (index > 0) & (da > 0) & ((sd == 0) | (da < sd * (1 - OCCLUDE)))
    index = np.where(obj, index, 0).astype(np.uint8)
    ground = (db > 0) & ((sd == 0) | (sd > db * (1 - ON_GROUND)))   # the splat shows the ground (or nothing)
    lit = ground & ~obj & (sd > 0) & (np.abs(sd - db) < db * ON_GROUND) & (b.mean(-1) > 10)
    gain = np.median(bg[lit] / np.maximum(b[lit], 1), axis=0) if lit.sum() > LIT_MIN_PX else last_gain
    sa, sb = smooth(a), smooth(b)
    shadow = np.clip(sa / np.maximum(sb, 1), 0, 1.2)
    shadow[~ground | (sb.mean(-1) < DARK)] = 1.0
    out = np.where(obj[..., None], a * gain, bg * shadow)
    return out, obj, index, lit, gain


def boxes(index, n: int, min_px: int) -> list[dict | None]:
    """Per object of the view: its visible pixels and tight box [x0, y0, x1, y1], or None under min_px."""
    out = []
    for k in range(n):
        ys, xs = np.nonzero(index == k + 1)
        out.append(None if len(xs) < min_px else
                   {"pixels": int(len(xs)), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
    return out
