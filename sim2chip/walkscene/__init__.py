"""Training frames in a walk scene (the WorldStereo route's bundle, experiments/worldstereo/wsroute/bundle.py): the
class's 3D models placed on the scene's free ground and rendered over its splat (docs/plan_integracion_f3c_2026-10-26.md,
1a to 1d). A scene is only the place: every object in a frame comes from the job's assets.

    plan.py        cameras and objects of every frame (numpy, the stage's own Python)
    background.py  the splat seen by each camera: colour with the sky behind it, z-depth, the ground's height under
                   each object and its colour (SIM2CHIP_GSPLAT_PYTHON)
    isaac.py       the objects rendered in Isaac Sim over a ground plane and composited on the background by depth,
                   with their shadows (SIM2CHIP_ISAAC_PYTHON)

Conventions, the bundle's: metres, z up, the scene photo's camera at the origin looking along +x, the ground
CAMERA_HEIGHT below it; cameras as OpenCV world-to-camera matrices. Isaac's stage puts the ground at z = 0
(isaac_world). The helpers here are numpy only (every step's Python has numpy); they repeat wsroute's camera.py,
which lives in experiments/ until the route moves into the package."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCENE = "scene.json"


def w2c(pos, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenCV world-to-camera for a camera at pos looking at yaw (0 = +x, + = left) and pitch (+ = up)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd], 0)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, -r @ np.asarray(pos, float)
    return m


def intrinsic(w: int, h: int, fov_deg: float) -> np.ndarray:
    """A centred pinhole of horizontal field of view fov_deg."""
    f = w / 2 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def pano_lookup(pano: np.ndarray, m: np.ndarray, K: np.ndarray, w: int, h: int) -> np.ndarray:
    """The equirectangular image's value along each pixel's ray (nearest): the sky, which is at infinity, seen from
    anywhere. The panorama's centre is +x, its left +y."""
    ph, pw = pano.shape[:2]
    j, i = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ray = np.stack([(j - K[0, 2]) / K[0, 0], (i - K[1, 2]) / K[1, 1], np.ones_like(j)], -1) @ np.linalg.inv(m)[:3, :3].T
    yaw = np.arctan2(ray[..., 1], ray[..., 0])
    pitch = np.arcsin(ray[..., 2] / np.linalg.norm(ray, axis=-1))
    return pano[np.clip(((0.5 - pitch / np.pi) * ph).astype(int), 0, ph - 1),
                np.clip(((0.5 - yaw / (2 * np.pi)) * pw).astype(int), 0, pw - 1)]


def isaac_world(m: np.ndarray, camera_height: float) -> np.ndarray:
    """A world-to-camera of the scene as the camera-to-world in Isaac's stage (the ground at z = 0)."""
    c2w = np.linalg.inv(m)
    c2w[2, 3] += camera_height
    return c2w


class Scene:
    """A bundle folder: its scene.json and free-ground grid."""

    def __init__(self, folder: Path):
        self.dir = Path(folder)
        self.meta = json.loads((self.dir / SCENE).read_text())
        g = np.load(self.dir / self.meta["grid"]["file"])
        self.free, self.occupied = g["free"], g["occupied"]
        self.cell, self.span = self.meta["grid"]["cell_m"], self.meta["grid"]["span_m"]

    @property
    def camera_height(self) -> float:
        return float(self.meta["camera_height"])

    def cells(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Grid indices of (x, y) points and whether they fall inside the grid."""
        q = np.floor((np.atleast_2d(np.asarray(xy, float)) + self.span) / self.cell).astype(int)
        return q, (q >= 0).all(1) & (q < len(self.free)).all(1)

    def is_free(self, xy) -> np.ndarray:
        q, inside = self.cells(xy)
        out = np.zeros(len(q), bool)
        out[inside] = self.free[q[inside, 0], q[inside, 1]]
        return out

    def free_points(self, within: float | None = None) -> np.ndarray:
        """Centres of the free cells, within that many metres of the scene's centre when given."""
        ij = np.argwhere(self.free)
        xy = ij * self.cell - self.span + self.cell / 2
        if within is not None:
            xy = xy[np.hypot(*(xy - np.asarray(self.meta["centre"])).T) <= within]
        return xy
