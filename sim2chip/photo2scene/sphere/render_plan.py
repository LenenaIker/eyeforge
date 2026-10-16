"""S.4 of the sphere route, what Isaac renders (steps/render.py): the views, each with its background cut from the
erased panorama, the camera pose and the objects placed in it, plus the proxy's colours and the light. Numpy and
PIL, run in the main Python; the Isaac step only renders what this writes.

Frames. The levelled sphere frame (geometry.py: OpenCV axes, +y down, the camera at the origin, the ground plane
at y = camera height) goes to Isaac's world (Z up, metres, the ground at z = 0, the camera at z = height) by
world = (x, z, height - y). A view's heading and pitch are in the levelled frame, so pitch 0 is the true horizon;
its background is cut from the panorama along the same rays taken back to the panorama's frame
(d_pano = R^T d_level, R the levelling).

Writes <dir>/render/plan.json, bg_NN.png (each view's background, exactly what the camera sees of the erased
panorama) and proxy_colors.npy (linear RGB per proxy vertex, for the light it bounces).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import capture

LEVEL_TO_WORLD = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # x, z, -y (plus the height on z)
SENSOR_WIDTH_MM = 20.955  # Isaac's default horizontal aperture; only the ratio to the focal length matters


def to_world(p_level: np.ndarray, height: float) -> np.ndarray:
    return p_level @ LEVEL_TO_WORLD.T + np.array([0.0, 0.0, height])


def camera_pose(view: dict, height: float, offset=(0.0, 0.0, 0.0)) -> list[list[float]]:
    """4x4 camera-to-world matrix of a view (OpenCV camera axes) in Isaac's world; offset moves the camera, in
    world metres, from the capture point."""
    m = np.eye(4)
    m[:3, :3] = LEVEL_TO_WORLD @ capture.rotation(view["yaw_deg"], view["pitch_deg"])
    m[:3, 3] = np.array([0.0, 0.0, height]) + np.asarray(offset, float)
    return m.tolist()


def level_rays(view: dict) -> np.ndarray:
    return capture.view_rays(view)


def pano_rays(rays_level: np.ndarray, levelling: np.ndarray) -> np.ndarray:
    """Rays of the levelled frame in the panorama's own frame (d_pano = R^T d_level)."""
    return rays_level @ np.asarray(levelling)


def background_cut(background: np.ndarray, view: dict, levelling: np.ndarray) -> np.ndarray:
    return np.clip(capture.sample(background, pano_rays(level_rays(view), levelling)) + 0.5, 0, 255).astype(np.uint8)


def captured_share(seen: np.ndarray, view: dict, levelling: np.ndarray) -> float:
    return float((capture.sample(seen.astype(np.uint8), pano_rays(level_rays(view), levelling)) > 0.999).mean())


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


ALBEDO_MAX = 0.9  # a proxy albedo above this would reflect more light than real materials do


def proxy_colors(vertices: np.ndarray, background: np.ndarray, seen: np.ndarray, levelling: np.ndarray,
                 is_ground: np.ndarray, light_rgb) -> np.ndarray:
    """An albedo per proxy vertex: the erased panorama's linear colour with the light's colour divided out (the
    photo is albedo times light; left in, the sky's tint would come back twice, once in the bounce and once in the
    exposure gain), scaled so the brightest is ALBEDO_MAX. Vertices the sphere never saw (the ground under the
    camera) take the median of the ground it did see."""
    d = vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-9)
    rays = pano_rays(d, levelling)
    col = srgb_to_linear(capture.sample(background, rays))
    known = capture.sample(seen.astype(np.uint8), rays) > 0.999
    ground_seen = known & is_ground
    fill = np.median(col[ground_seen], 0) if ground_seen.any() else np.median(col[known], 0)
    col[~known] = fill
    col = col / np.maximum(np.asarray(light_rgb, np.float32), 1e-3)
    return (col * ALBEDO_MAX / max(float(np.percentile(col, 99.5)), 1e-6)).clip(0, ALBEDO_MAX).astype(np.float32)


def sky_color(background: np.ndarray, sky_share: np.ndarray) -> list[float]:
    """Mean linear colour of the sky, scaled so its brightest channel is 1 (the exposure is fitted later)."""
    H, W = sky_share.shape
    small = np.asarray(Image.fromarray(background).resize((W, H), Image.Resampling.BOX))
    sky = sky_share.astype(np.float32) > 0.5
    if sky.sum() < 100:
        return [1.0, 1.0, 1.0]
    c = srgb_to_linear(small[sky]).mean(0)
    return (c / max(float(c.max()), 1e-6)).round(4).tolist()


def resolve_asset(usd: str, out_dir: Path) -> str:
    """The asset as Isaac can open it: the converted wrapper itself when its reference resolves, else a copy in
    out_dir whose reference points at the same file name in the wrapper's own raw/ folder. The
    wrappers converted before the folder rename of 2026-10-01 carry absolute paths to a folder that no longer
    exists (convert_assets.py writes relative ones since)."""
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(usd)
    spec = layer.GetPrimAtPath("/Asset/Geo") if layer else None
    if spec is None:
        return usd
    refs = list(spec.referenceList.prependedItems) or list(spec.referenceList.explicitItems)
    if not refs:
        return usd
    target = Path(layer.ComputeAbsolutePath(refs[0].assetPath))
    if target.exists():
        return usd
    local = Path(usd).parent / "raw" / target.name
    if not local.exists():
        return usd
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(usd).name
    if not out.exists():
        copy = Sdf.Layer.CreateNew(str(out))
        copy.TransferContent(layer)
        geo = copy.GetPrimAtPath("/Asset/Geo")
        geo.referenceList.ClearEditsAndMakeExplicit()
        geo.referenceList.explicitItems = [Sdf.Reference(str(local.resolve()))]
        copy.Save()
    return str(out)


def write(d: Path, views: list[dict], sun: dict | None = None) -> dict:
    """The render folder for these views. Each view: name, yaw_deg, pitch_deg, fov_deg, size, objects (each
    {"usd", "length_m": its real long side (the render scales the measured asset to it), "pos": world xyz of its
    base centre, "yaw_deg" about the vertical, "label"}), optional
    "offset" (camera, world metres)."""
    d = Path(d)
    geo = json.loads((d / "geometry.json").read_text())
    R, height = np.array(geo["levelling"]), geo["camera_height_m"]
    background = np.asarray(Image.open(d / "background.jpg").convert("RGB"))
    seen = np.asarray(Image.open(d / "seen.png")) > 127
    proxy = np.load(d / "proxy.npz")
    out = d / "render"
    out.mkdir(exist_ok=True)
    stuff = np.load(d / "stuff.npz")
    sky = sky_color(background, stuff["sky"])
    np.save(out / "proxy_colors.npy", proxy_colors(proxy["vertices"], background, seen, R, proxy["ground"], sky))
    plan_views = []
    for v in views:
        for o in v.get("objects", []):
            o["usd"] = resolve_asset(o["usd"], out / "assets")
    for k, v in enumerate(views):
        name = v.get("name", f"{k:02d}")
        Image.fromarray(background_cut(background, v, R)).save(out / f"bg_{name}.png")
        plan_views.append({**v, "name": name, "pose": camera_pose(v, height, v.get("offset", (0, 0, 0))),
                           "captured": round(captured_share(seen, v, R), 4)})
    plan = {"camera_height_m": height, "sky_rgb": sky, "sun": sun,
            "views": plan_views}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    return plan


FREE_TOL = 0.1          # a footprint point's measured distance may differ this share from where it should be
GROUND_SEARCH_DEG = (2.0, 45.0)  # depressions below the horizon searched for ground along a heading


class Ground:
    """The erased panorama's measured ground, for putting objects on it: its depth (depth_bg.npz at the proxy's
    scale) and SAM 3's ground votes on it (stuff_bg.npz)."""

    def __init__(self, d: Path):
        d = Path(d)
        geo = json.loads((d / "geometry.json").read_text())
        self.R, self.h = np.array(geo["levelling"]), geo["camera_height_m"]
        bg = np.load(d / "depth_bg.npz")
        self.dist = bg["distance"] * geo["scale"] * geo.get("proxy_depth_ratio", 1.0)
        self.H, self.W = self.dist.shape
        votes = d / "stuff_bg.npz" if (d / "stuff_bg.npz").exists() else d / "stuff.npz"
        self.floor = (np.load(votes)["ground"].astype(np.float32) > 0.5) & bg["valid"]

    def look(self, p_level: np.ndarray):
        """For points of the levelled frame: whether their pixel is ground, and the measured distance there."""
        rays = pano_rays(p_level / np.linalg.norm(p_level, axis=-1, keepdims=True), self.R)
        x, y = capture.to_pixel(rays, self.W, self.H)
        xi, yi = np.round(x).astype(int) % self.W, np.clip(np.round(y).astype(int), 0, self.H - 1)
        return self.floor[yi, xi], self.dist[yi, xi]

    def along(self, heading_deg: float, step_deg: float = 0.25) -> list[np.ndarray]:
        """Measured ground points (levelled frame) straight along a heading, nearest last."""
        out = []
        for dep in np.arange(GROUND_SEARCH_DEG[0], GROUND_SEARCH_DEG[1], step_deg):
            ray = capture.rotation(heading_deg, dep) @ np.array([0.0, 0.0, 1.0])
            ok, dist = self.look(ray[None] * 10)
            if ok[0]:
                out.append(ray * dist[0])
        return out

    def footprint_free(self, centre: np.ndarray, length_m: float, width_m: float, axis_deg: float,
                       grid=(5, 3)) -> bool:
        """Whether a length_m x width_m rectangle (long side at axis_deg, world yaw counterclockwise from +x seen
        from above) centred on a measured ground point lies on ground everywhere, at the centre's height: each grid
        point's pixel is ground and measured where it should be, nothing in front of it and no drop (a kerb, a
        wall). An object over them would sink into the proxy and show the background through itself."""
        a = math.radians(axis_deg)
        along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        u, v = np.meshgrid(np.linspace(-0.5, 0.5, grid[0]) * length_m, np.linspace(-0.5, 0.5, grid[1]) * width_m)
        xy = u.reshape(-1, 1) * along + v.reshape(-1, 1) * across  # world x, y = level x, z
        p = centre + np.stack([xy[:, 0], np.zeros(len(xy)), xy[:, 1]], 1)
        ok, dist = self.look(p)
        expected = np.linalg.norm(p, axis=1)
        return bool(ok.all() and (np.abs(dist - expected) <= FREE_TOL * expected).all())

    def place(self, heading_deg: float, distances_m, length_m: float, width_m: float, axis_deg: float):
        """The measured ground point along a heading nearest each wanted horizontal distance, in order, whose
        footprint is free: (world position of the object's base centre, horizontal distance), or None."""
        points = self.along(heading_deg)
        if not points:
            return None
        horiz = np.array([math.hypot(p[0], p[2]) for p in points])
        for D in distances_m:
            k = int(np.argmin(np.abs(horiz - D)))
            if abs(horiz[k] - D) > 0.5:
                continue
            if self.footprint_free(points[k], length_m, width_m, axis_deg):
                return to_world(points[k], self.h).tolist(), round(float(horiz[k]), 2)
        return None
