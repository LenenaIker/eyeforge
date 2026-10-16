"""S.1 of the sphere route: a photosphere (a phone's 360 panorama, equirectangular JPEG) read into the working
panorama and cut into pinhole views, the photos the 2D models (SAM 3, the eraser, MoGe) work on.

A partial sphere (GPano XMP: the cropped area inside the full panorama) is placed in its full canvas with a mask of
what was captured; the cuts keep only the views mostly inside it. Each cut is the place turned, not moved, so it is
exact and maps back to the panorama ray by ray. Pure numpy and PIL.

Writes in the place folder:
    pano.jpg     the working panorama (WORK_WIDTH x WORK_WIDTH/2), black where nothing was captured
    seen.png     the captured mask of the working panorama
    cuts/*.jpg   the views, named by heading and pitch
    sphere.json  what was captured, the capture's heading, time and rough position, and the views (written last:
                 a place with it is done)
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

WORK_WIDTH = 4096       # 11.4 px per degree, about the density of the cuts at their centre
CUT_SIZE = (1024, 768)  # 4:3, what SAM 3 and the eraser work at
CUT_FOV_DEG = 75.0      # horizontal; 11.6 px per degree at the centre, matching the working panorama
CUT_HEADINGS = 12       # every 30 degrees: neighbours overlap by more than half
CUT_PITCHES = (0.0, 35.0)  # the horizon, and the ground and near things (degrees down)
MIN_CAPTURED = 0.5      # share of a cut inside the capture below which it is left out
WRAP_PAD = 64           # source columns copied across the seam so resizing a full turn does not see an edge

_GPANO = ("FullPanoWidthPixels", "FullPanoHeightPixels", "CroppedAreaLeftPixels", "CroppedAreaTopPixels",
          "CroppedAreaImageWidthPixels", "CroppedAreaImageHeightPixels")


def gpano(path: Path) -> dict[str, int]:
    """The GPano crop of a photosphere JPEG (its XMP, as attributes or elements); empty without one."""
    head = Path(path).read_bytes()[:1 << 20].decode("latin-1")
    out = {}
    for key in _GPANO:
        m = re.search(rf'GPano:{key}(?:="|>)\s*(\d+)', head)
        if m:
            out[key] = int(m.group(1))
    return out if len(out) == len(_GPANO) else {}


def _xmp_float(path: Path, key: str) -> float | None:
    m = re.search(rf'GPano:{key}(?:="|>)\s*(-?[\d.]+)', Path(path).read_bytes()[:1 << 20].decode("latin-1"))
    return float(m.group(1)) if m else None


def capture_info(path: Path) -> dict:
    """When and where the sphere was taken, for the sun: the compass heading of the panorama's centre (GPano), the
    local time with its offset and the position rounded to about a kilometre (EXIF). Missing fields are left out."""
    out = {}
    if (heading := _xmp_float(path, "PoseHeadingDegrees")) is not None:
        out["heading_deg"] = heading
    with Image.open(path) as im:
        exif = im.getexif()
    sub, gps = exif.get_ifd(0x8769), exif.get_ifd(0x8825)
    if when := sub.get(36867):  # DateTimeOriginal, with OffsetTimeOriginal when the phone wrote it
        try:
            out["taken_at"] = datetime.strptime(when, "%Y:%m:%d %H:%M:%S").isoformat() + (sub.get(36881) or "")
        except ValueError:
            pass
    if gps.get(2) and gps.get(4):
        deg = lambda v, ref, neg: (-1 if ref == neg else 1) * float(v[0] + v[1] / 60 + v[2] / 3600)  # noqa: E731
        out["lat"] = round(deg(gps[2], gps.get(1), "S"), 2)
        out["lon"] = round(deg(gps[4], gps.get(3), "W"), 2)
    return out


def read(path: Path, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The photosphere in a full 2:1 canvas `width` wide (RGB, uint8) and the mask of what was captured. Without
    GPano only a full 2:1 sphere is accepted: anything else is not a photosphere this route can place."""
    width = width or WORK_WIDTH
    g = gpano(path)
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"))
    ih, iw = img.shape[:2]
    if not g:
        if abs(iw - 2 * ih) > 2:
            raise ValueError(f"{Path(path).name}: not a photosphere (no GPano metadata and not 2:1, {iw}x{ih})")
        g = dict(zip(_GPANO, (iw, ih, 0, 0, iw, ih)))
    if iw != g["CroppedAreaImageWidthPixels"]:  # saved at another size than the metadata says
        s = iw / g["CroppedAreaImageWidthPixels"]
        g = {k: round(v * s) for k, v in g.items()}
    s = width / g["FullPanoWidthPixels"]
    height = width // 2
    full_turn = iw >= g["FullPanoWidthPixels"]
    out_w, out_h = min(round(iw * s), width), min(round(ih * s), height)
    if full_turn:  # resample with the seam's neighbours on both sides, then trim them
        pad = min(WRAP_PAD, iw // 2)
        p = math.ceil(pad * s)
        img = np.concatenate([img[:, -pad:], img, img[:, :pad]], 1)
        small = np.asarray(Image.fromarray(img).resize((out_w + 2 * p, out_h), Image.Resampling.LANCZOS))[:, p:-p]
    else:
        small = np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.Resampling.LANCZOS))
    canvas = np.zeros((height, width, 3), np.uint8)
    seen = np.zeros((height, width), bool)
    x0, y0 = round(g["CroppedAreaLeftPixels"] * s) % width, round(g["CroppedAreaTopPixels"] * s)
    rows = slice(y0, min(y0 + out_h, height))
    cols = (np.arange(out_w) + x0) % width
    canvas[rows, cols] = small[:rows.stop - y0]
    seen[rows, cols] = True
    return canvas, seen


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera to sphere rotation of a view turned to a heading (yaw, positive to the right) and pitched down
    (pitch, positive down): d_sphere = R @ d_camera."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    down = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    turn = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return turn @ down


def focal(w: int, fov_deg: float) -> float:
    return w / 2 / math.tan(math.radians(fov_deg) / 2)


def rays(w: int, h: int, fov_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit rays of a pinhole camera (OpenCV: +x right, +y down, +z forward) turned and pitched as rotation() says,
    in the sphere's frame: (h, w, 3)."""
    f = focal(w, fov_deg)
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, np.arange(h) + 0.5 - h / 2)
    d = np.stack([u / f, v / f, np.ones_like(u)], -1) @ rotation(yaw_deg, pitch_deg).T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(view: dict, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where rays of the sphere land in a view: continuous pixel coordinates (x, y; 0 is the first pixel's centre)
    and whether they fall inside it, in front of the camera."""
    w, h = view["size"]
    c = d @ rotation(view["yaw_deg"], view["pitch_deg"])
    z = np.where(c[..., 2] > 1e-6, c[..., 2], np.nan)
    f = focal(w, view["fov_deg"])
    x, y = f * c[..., 0] / z + w / 2 - 0.5, f * c[..., 1] / z + h / 2 - 0.5
    inside = ~np.isnan(z) & (x > -0.5) & (x < w - 0.5) & (y > -0.5) & (y < h - 0.5)
    return np.nan_to_num(x), np.nan_to_num(y), inside


def pixel_rays(width: int, height: int) -> np.ndarray:
    """The ray through each pixel centre of an equirectangular image (the inverse of to_pixel): (height, width, 3)."""
    x, y = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    lon, lat = (x / width - 0.5) * 2 * math.pi, (0.5 - y / height) * math.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


def to_pixel(d: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous pixel coordinates (x, y; 0 is the first pixel's centre) of rays in an equirectangular image:
    longitude 0 at the centre column growing to the right, latitude up."""
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = -np.arcsin(np.clip(d[..., 1], -1, 1))
    return (lon / (2 * math.pi) + 0.5) * width - 0.5, (0.5 - lat / math.pi) * height - 0.5


def sample(pano: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Bilinear lookup of rays in an equirectangular image (wrapping around), float32. Only the four neighbours of
    each ray are converted, never the whole panorama."""
    H, W = pano.shape[:2]
    x, y = to_pixel(d, W, H)
    y = np.clip(y, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    y1 = np.minimum(y0 + 1, H - 1)
    x0, x1 = x0 % W, (x0 + 1) % W
    if pano.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    at = lambda yy, xx: pano[yy, xx].astype(np.float32)  # noqa: E731
    return (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) + (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy


def view_rays(view: dict) -> np.ndarray:
    return rays(*view["size"], view["fov_deg"], view["yaw_deg"], view["pitch_deg"])


def captured_extent(seen: np.ndarray) -> dict:
    """How far up and down the capture reaches (degrees from the horizon) and the share of headings it covers."""
    H = seen.shape[0]
    rows = np.flatnonzero(seen.any(1))
    if not len(rows):
        return {"up_deg": None, "down_deg": None, "around_share": 0.0}
    lat = lambda r: 90 - (r + 0.5) / H * 180  # noqa: E731
    return {"up_deg": round(float(lat(rows[0])), 1), "down_deg": round(float(-lat(rows[-1])), 1),
            "around_share": round(float(seen.any(0).mean()), 3)}


def plan_cuts(seen: np.ndarray, headings: int | None = None, pitches=None, fov_deg: float | None = None,
              size: tuple[int, int] | None = None) -> list[dict]:
    """The views to cut, every heading at every pitch, without those mostly outside the capture (defaults: the
    CUT_* settings)."""
    headings, pitches = headings or CUT_HEADINGS, CUT_PITCHES if pitches is None else pitches
    fov_deg, size = fov_deg or CUT_FOV_DEG, size or CUT_SIZE
    views = []
    for pitch in pitches:
        for k in range(headings):
            view = {"yaw_deg": round(360.0 * k / headings, 3), "pitch_deg": float(pitch), "fov_deg": fov_deg,
                    "size": list(size)}
            share = float((sample(seen.astype(np.uint8), view_rays(view)) > 0.999).mean())
            if share >= MIN_CAPTURED:
                name = f"y{round(view['yaw_deg']):03d}_p{round(pitch):02d}.jpg"
                views.append({"file": name, **view, "captured": round(share, 3)})
    return views


def prepare(photosphere: Path, d: Path) -> dict:
    """S.1 for one place: the working panorama, its mask and the cuts in d (see the module docstring). Done once:
    a folder with sphere.json is returned as it is."""
    d = Path(d)
    if (d / "sphere.json").exists():
        return json.loads((d / "sphere.json").read_text())
    pano, seen = read(photosphere)
    (d / "cuts").mkdir(parents=True, exist_ok=True)
    Image.fromarray(pano).save(d / "pano.jpg", quality=95)
    Image.fromarray(seen.astype(np.uint8) * 255).save(d / "seen.png")
    views = plan_cuts(seen)
    for view in views:
        img = sample(pano, view_rays(view))
        Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(d / "cuts" / view["file"], quality=95)
    info = {"source": Path(photosphere).name, "work_size": [pano.shape[1], pano.shape[0]],
            "captured": captured_extent(seen), **capture_info(photosphere), "cuts": views}
    (d / "sphere.json").write_text(json.dumps(info, indent=1))
    return info
