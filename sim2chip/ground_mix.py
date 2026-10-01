"""Mixed ground: a base texture with patches of others and, in some layouts, a hard-edged strip.

A lawn is not one grass texture repeated: it has bare spots, fallen leaves, a path or a curb crossing it. We
paint that as one big texture per layout (SIZE_M square, repeating) from Poly Haven textures at their real
size: patches follow periodic smooth noise with a ragged border (so the texture tiles without seams), the
strip is a straight band with a hard edge. The label map of every layout (which material is where) is saved
beside it for tasks that classify zones of the ground.

numpy and Pillow only; runs in fetch_assets. Layouts are cached by content in the shared cache.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

SIZE_M = 8.0          # side of one layout, meters
PX = 2048             # its side, pixels (25 px per 10 cm: finer than any camera sees it from 1 m)
LAYOUTS = 12
VERSION = "1"
DEFAULT_COVER = (0.1, 0.4)
DEFAULT_PATCH_M = (0.3, 1.5)
DEFAULT_EDGE_SHARE = 0.3
EDGE_WIDTH_M = (0.8, 2.5)


def noise(rng: random.Random, px: int, feature_px: float):
    """Periodic smooth noise with blobs about feature_px wide, standardized."""
    import numpy as np
    r = np.random.default_rng(rng.getrandbits(32))
    f = np.fft.fftfreq(px)
    k = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    spectrum = np.fft.fft2(r.normal(size=(px, px))) * np.exp(-(k * feature_px) ** 2 * 2)
    n = np.real(np.fft.ifft2(spectrum))
    return (n - n.mean()) / (n.std() + 1e-9)


def tiled(path: Path, texture_m: float, px: int = PX, size_m: float = SIZE_M):
    """The texture at its real size, repeated over the layout (float RGB 0..1)."""
    import numpy as np
    from PIL import Image
    side = max(8, round(texture_m / size_m * px))
    with Image.open(path) as im:
        t = np.asarray(im.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255
    reps = -(-px // side)
    return np.tile(t, (reps, reps, 1))[:px, :px]


def compose(base, patches, edges, rng: random.Random, cover=DEFAULT_COVER, patch_m=DEFAULT_PATCH_M,
            edge_share=DEFAULT_EDGE_SHARE, px: int = PX, size_m: float = SIZE_M):
    """One layout. base, patches, edges: lists of (path, texture size m). Returns (RGB uint8, labels uint8,
    names): label 0 is the base, then one per patch material, then the strip."""
    import numpy as np
    per_m = px / size_m
    img = tiled(*rng.choice(base), px, size_m)
    labels = np.zeros((px, px), np.uint8)
    names = ["base"]
    if patches:
        total = rng.uniform(*cover)
        kinds = rng.sample(patches, rng.randint(1, len(patches)))
        shares = [rng.random() + 0.3 for _ in kinds]
        for tex, share in zip(kinds, shares):
            want = total * share / sum(shares)
            size = rng.uniform(*patch_m) * per_m
            # Big blobs for the patch, finer noise for a ragged border (grass thinning into dirt).
            n = noise(rng, px, size / 2.5) + 0.35 * noise(rng, px, max(2.0, size / 25))
            thr = np.quantile(n, 1 - want)
            soft = 0.12  # width of the transition, in noise units
            a = np.clip((n - thr) / soft + 0.5, 0, 1)[..., None]
            t = tiled(*tex, px, size_m)
            img = img * (1 - a) + t * a
            names.append(Path(tex[0]).stem)
            labels[a[..., 0] >= 0.5] = len(names) - 1
    if edges and rng.random() < edge_share:
        w = rng.uniform(*EDGE_WIDTH_M) * per_m
        y0 = rng.uniform(0, px - w)
        rows = slice(int(y0), int(y0 + w))
        t = tiled(*rng.choice(edges), px, size_m)
        img[rows] = t[rows]
        for y in (int(y0), int(y0 + w) - 1):  # a darker joint where the strip meets the ground
            img[max(0, y - 2):y + 2] *= 0.6
        names.append("strip")
        labels[rows] = len(names) - 1
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out, labels, names


def make(base, patches, edges, dest: Path, seed: int, cover=None, patch_m=None, edge_share=None,
         n: int = LAYOUTS) -> list[Path]:
    """n layouts in dest/ (reused when the same inputs were composed before). Returns the texture files."""
    from PIL import Image
    cover, patch_m = cover or DEFAULT_COVER, patch_m or DEFAULT_PATCH_M
    edge_share = DEFAULT_EDGE_SHARE if edge_share is None else edge_share
    key = hashlib.sha256(json.dumps([VERSION, [[Path(p).name, m] for p, m in base],
                                     [[Path(p).name, m] for p, m in patches], [[Path(p).name, m] for p, m in edges],
                                     list(cover), list(patch_m), edge_share, seed, n, PX]).encode()).hexdigest()[:12]
    files = [dest / f"mix_{key}_{i:02d}.jpg" for i in range(n)]
    if all(f.exists() for f in files):
        return files
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for f in files:
        img, labels, names = compose(base, patches, edges, rng, cover, patch_m, edge_share)
        Image.fromarray(labels).save(f.with_suffix(".labels.png"))
        f.with_suffix(".json").write_text(json.dumps({"labels": names, "size_m": SIZE_M}))
        tmp = f.with_suffix(".part.jpg")
        Image.fromarray(img).save(tmp, quality=90)
        tmp.rename(f)
    return files
