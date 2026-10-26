"""Where a synthetic (student) camera may stand: a grid over the spec's area at the walk's height, every STUDENT
heading, clear of obstacles and reachable from the photo's spot; plus the splat loading and rendering every
measure of the route shares (gsplat, SIM2CHIP_GSPLAT_PYTHON)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import camera
from .camera import CAMERA_HEIGHT, OBSTACLE_Z
from .panorama import Panorama
from .paths import Case

STUDENT_GRID_M = 1.0
STUDENT_HEADINGS = tuple(range(0, 360, 45))
STUDENT_PITCH_DEG = -15.0
STUDENT_FOV_DEG = 68.0
CLEARANCE_M = 0.5      # a camera this near an obstacle point is inside a hedge or a car
ALPHA_MIN = 0.5        # a splat pixel with less opacity shows nothing


def load_splat(path: Path, dev):
    """A 3DGS .ply as gsplat's rasterization arguments."""
    import torch
    import torch.nn.functional as tF
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]

    def g(*k):
        return torch.as_tensor(np.stack([np.asarray(v[x], np.float32) for x in k], -1), device=dev)
    return dict(means=g("x", "y", "z"), quats=tF.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
                scales=torch.exp(g("scale_0", "scale_1", "scale_2")), opacities=torch.sigmoid(g("opacity")[:, 0]),
                colors=g("f_dc_0", "f_dc_1", "f_dc_2")[:, None])


def render(sp, w2cs: np.ndarray, K: np.ndarray, w: int, h: int, batch: int = 16):
    """RGB (uint8), z-depth and alpha of the splat for each world-to-camera."""
    import torch
    from gsplat import rasterization
    dev = sp["means"].device
    rgbs, deps, alphas = [], [], []
    for i in range(0, len(w2cs), batch):
        m = torch.as_tensor(np.asarray(w2cs[i:i + batch]), dtype=torch.float32, device=dev)
        k = torch.as_tensor(K, dtype=torch.float32, device=dev)[None].repeat(len(m), 1, 1)
        with torch.no_grad():
            out, alpha, _ = rasterization(**sp, viewmats=m, Ks=k, width=w, height=h, sh_degree=0,
                                          render_mode="RGB+ED", near_plane=0.05, far_plane=500, packed=False)
        rgbs.append((out[..., :3].clamp(0, 1) * 255).byte())
        deps.append(out[..., 3])
        alphas.append(alpha[..., 0])
    return torch.cat(rgbs), torch.cat(deps), torch.cat(alphas)


def obstacles(case: Case, sp) -> np.ndarray:
    """Points (x, y) standing between OBSTACLE_Z above the ground: the panorama's and the splat's opaque Gaussians
    (what the generated legs added)."""
    pts = Panorama(case).points()[0]
    m = sp["means"].cpu().numpy()
    m = m[sp["opacities"].cpu().numpy() > 0.5]
    every = np.concatenate([pts, m[::5]])
    hgt = every[:, 2] + CAMERA_HEIGHT
    keep = (hgt > OBSTACLE_Z[0]) & (hgt < OBSTACLE_Z[1])
    return every[keep][:, :2]


def seed(case: Case) -> tuple[float, float]:
    """Where free space is grown from: the walks' centre (walks.prep's clearest ground point), else the photo's
    spot. A photo taken against a fence leaves its own spot within CLEARANCE_M of an obstacle (f3c_photo21)."""
    import json
    c = json.loads(case.prep_json.read_text()).get("centre")
    return (float(c[0]), float(c[1])) if c else (0.0, 0.0)


def free_space(obs: np.ndarray, start=(0.0, 0.0), cell: float = 0.25, span: float = 12.0):
    """Where a camera can stand: cells CLEARANCE_M clear of obstacle points and reachable from start (the nearest
    clear cell to it) without crossing them (a hedge's far side is not the street). Returns a test function on
    (x, y)."""
    n = int(2 * span / cell)
    occ = np.zeros((n, n), bool)
    ij = np.floor((obs + span) / cell).astype(int)
    ij = ij[(ij >= 0).all(1) & (ij < n).all(1)]
    occ[ij[:, 0], ij[:, 1]] = True
    r = int(math.ceil(CLEARANCE_M / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    occ = ndimage.binary_dilation(occ, structure=(xx ** 2 + yy ** 2) <= r * r)
    lab, _ = ndimage.label(~occ)
    s = np.floor((np.asarray(start, float) + span) / cell).astype(int)
    if not lab[s[0], s[1]] and lab.any():
        clear = np.argwhere(lab > 0)
        s = clear[np.argmin(np.hypot(*(clear - s).T))]
    free = lab == lab[s[0], s[1]] if lab[s[0], s[1]] else np.zeros_like(occ)

    def ok(xy) -> np.ndarray:
        q = np.floor((np.atleast_2d(xy) + span) / cell).astype(int)
        inside = (q >= 0).all(1) & (q < n).all(1)
        out = np.zeros(len(q), bool)
        out[inside] = free[q[inside, 0], q[inside, 1]]
        return out
    return ok


def poses(obs: np.ndarray, radius: float, start=(0.0, 0.0)) -> tuple[np.ndarray, list]:
    """World-to-cameras of every student camera within radius metres, and their (x, y, heading)."""
    free = free_space(obs, start)
    r = np.arange(-radius, radius + 1e-6, STUDENT_GRID_M)
    where = [(float(x), float(y)) for x in r for y in r if math.hypot(x, y) <= radius + 1e-6 and free([x, y])[0]]
    mats, out = [], []
    for x, y in where:
        for hd in STUDENT_HEADINGS:
            mats.append(camera.w2c([x, y, 0.0], hd, STUDENT_PITCH_DEG).astype(np.float32))
            out.append((x, y, hd))
    return np.stack(mats), out


def at_spots(where: list, spots) -> list[int]:
    """Indices of the student cameras standing on the given grid spots, spot by spot."""
    want = [tuple(float(v) for v in s) for s in spots]
    return [k for s in want for k, (x, y, _) in enumerate(where) if (x, y) == s]
