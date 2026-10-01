"""The look of the user's camera: color, contrast, sharpness and noise, measured on their photos and given to
the synthetic images. numpy and Pillow only, no model.

Renders come out grey, flat, sharp and clean; photos have the color cast, contrast, softness and grain of
the camera and the place. We measure both sets at the model's input size and move the renders towards the
photos: a color transfer in Lab (Reinhard) with set statistics, so renders keep their own variety (a sunny
render stays brighter than a cloudy one), then blur and noise when the photos are softer or grainier.
Each image gets a random share of the change (STRENGTH), never a fixed look.

    stats = measure(photos)             # stored in spec.look (the user's side)
    t = fit(spec.look, measure(renders))  # the renders' side comes from the renders being transformed
    img = apply(img, t, rng)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

STRENGTH = (0.6, 1.0)  # share of the change given to each image
MAX_BLUR = 1.5         # px of gaussian sigma at the input size: more would erase the objects
MAX_NOISE = 0.04       # std of the added noise, 0..1 scale
MAX_SCALE = 3.0        # most a channel's spread is stretched: flat renders must not turn into color noise
FLAT_SHARE = 0.2       # noise is read on the flattest blocks, where texture does not pass for grain


def _lin(c):
    import numpy as np
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gamma(c):
    import numpy as np
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M = ((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505))
_WHITE = (0.95047, 1.0, 1.08883)


def to_lab(rgb):
    """rgb: float array (..., 3) in 0..1 (sRGB, D65). Returns L in 0..100, a and b around 0."""
    import numpy as np
    xyz = _lin(rgb) @ np.array(_M).T / np.array(_WHITE)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def from_lab(lab):
    import numpy as np
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.2068966, f ** 3, (f - 16 / 116) / 7.787) * np.array(_WHITE)
    return _gamma(xyz @ np.linalg.inv(np.array(_M)).T)


def _gray(rgb):
    return rgb @ (0.299, 0.587, 0.114)


def sharpness(rgb) -> float:
    """Mean gradient magnitude on the strongest edges (top 10%), over the image contrast: how crisp edges
    are, mostly independent of how much texture there is."""
    import numpy as np
    g = _gray(rgb)
    gx, gy = np.diff(g, axis=1)[:-1], np.diff(g, axis=0)[:, :-1]
    mag = np.hypot(gx, gy)
    top = mag[mag >= np.quantile(mag, 0.9)].mean()
    span = np.quantile(g, 0.98) - np.quantile(g, 0.02)
    return float(top / max(span, 1e-3))


def noise(rgb) -> float:
    """Noise std (0..1) by Immerkaer's operator on the flattest blocks of the image."""
    import numpy as np
    g = _gray(rgb)
    k = g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] \
        - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:]
    b = 8
    h, w = (k.shape[0] // b) * b, (k.shape[1] // b) * b
    blocks = np.abs(k[:h, :w]).reshape(h // b, b, w // b, b).mean(axis=(1, 3)).ravel()
    flat = np.sort(blocks)[:max(1, int(len(blocks) * FLAT_SHARE))]
    return float(math.sqrt(math.pi / 2) * flat.mean() / 6)


def measure(images) -> dict:
    """Set statistics of RGB uint8 arrays (all at the model's input size)."""
    import numpy as np
    labs, sharp, grain = [], [], []
    for im in images:
        rgb = np.asarray(im, dtype=np.float32)[..., :3] / 255
        labs.append(to_lab(rgb).reshape(-1, 3)[::4])
        sharp.append(sharpness(rgb))
        grain.append(noise(rgb))
    if not labs:
        raise ValueError("no images to measure")
    lab = np.concatenate(labs)
    return {"lab_mean": [round(float(v), 3) for v in lab.mean(0)], "lab_std": [round(float(v), 3) for v in lab.std(0)],
            "sharpness": round(float(np.median(sharp)), 4), "noise": round(float(np.median(grain)), 5),
            "n": len(labs)}


@dataclass
class Transform:
    src_mean: tuple
    src_std: tuple
    dst_mean: tuple
    dst_std: tuple
    blur_px: float
    noise: float


def blur_for(src_sharpness: float, dst_sharpness: float) -> float:
    """Gaussian sigma (px) that takes edges from src to dst crispness. An ideal step blurred by sigma has a
    peak slope of 1/(sigma sqrt(2 pi)); combined with the edge's own width, sigmas add in quadrature."""
    if dst_sharpness >= src_sharpness or dst_sharpness <= 0:
        return 0.0
    s0 = 1 / (src_sharpness * math.sqrt(2 * math.pi))
    s1 = 1 / (dst_sharpness * math.sqrt(2 * math.pi))
    return round(min(MAX_BLUR, math.sqrt(max(s1 ** 2 - s0 ** 2, 0.0))), 3)


def fit(user: dict, synth: dict) -> Transform:
    """The change from the renders' statistics to the user's photos'."""
    n = min(MAX_NOISE, math.sqrt(max(user["noise"] ** 2 - synth["noise"] ** 2, 0.0)))
    return Transform(tuple(synth["lab_mean"]), tuple(synth["lab_std"]), tuple(user["lab_mean"]),
                     tuple(user["lab_std"]), blur_for(synth["sharpness"], user["sharpness"]), round(n, 5))


def apply(img, t: Transform, rng: random.Random):
    """A PIL image with a random share of the transform."""
    import numpy as np
    from PIL import Image, ImageFilter

    s = rng.uniform(*STRENGTH)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    lab = to_lab(rgb)
    sm, ss, dm, ds = (np.array(v, dtype=np.float32) for v in (t.src_mean, t.src_std, t.dst_mean, t.dst_std))
    moved = (lab - sm) * np.minimum(ds / np.maximum(ss, 1e-3), MAX_SCALE) + dm
    rgb = from_lab(lab + s * (moved - lab))
    out = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    if t.blur_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(s * t.blur_px))
    if t.noise > 0:
        a = np.asarray(out, dtype=np.float32) / 255
        a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, s * t.noise, a.shape[:2])[..., None]
        out = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return out


def distance(a: dict, b: dict) -> dict:
    """How far two sets' looks are, per part (0: same). Color in Lab units (delta E of the means plus the
    spread gap), sharpness and noise as relative gaps."""
    color = math.dist(a["lab_mean"], b["lab_mean"]) + math.dist(a["lab_std"], b["lab_std"])
    sharp = abs(a["sharpness"] - b["sharpness"]) / max(a["sharpness"], b["sharpness"], 1e-6)
    grain = abs(a["noise"] - b["noise"]) / max(a["noise"], b["noise"], 1e-6)
    return {"color": round(color, 2), "sharpness": round(sharp, 3), "noise": round(grain, 3)}


SAMPLE = 200  # renders measured to fit the transform


def for_renders(user: dict, paths, size: int, seed: int = 0) -> Transform:
    """The transform for a set of renders (files), measured on a sample of them at `size` px (the size the
    stage writes, so blur and noise are in the same pixels)."""
    from PIL import Image
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    imgs = []
    for p in paths[:SAMPLE]:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB").resize((size, size), Image.BILINEAR))
    return fit(user, measure(imgs))
