"""Cameras and pixels in upstream WorldStereo's conventions (read from its example scene): world z up, the photo's
camera at the origin looking along x, OpenCV world-to-camera matrices, 832x480 frames with fx = fy = 240 (120
degree field of view), 21 frames per leg. Metres: the ground lies CAMERA_HEIGHT below the camera."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

W, H, F = 832, 480, 240.0  # upstream's frame and focal length
NFRAME = 21
CAMERA_HEIGHT = 1.6  # metres; the panorama is shot from a hand-held phone
PITCH = -5.0  # degrees; walks look slightly down, like upstream's start elevation
# An obstacle is a point between these heights above the ground.
OBSTACLE_Z = (0.25, 2.2)


def w2c(pos, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenCV world-to-camera matrix for a camera at pos looking at yaw (0 = panorama centre, + = left), pitch."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.cross(fwd, right)
    r = np.stack([right, down, fwd], 0)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, -r @ np.asarray(pos, float)
    return m


def pose_of(m: np.ndarray, from_inverse: bool = False) -> tuple[np.ndarray, float, float]:
    """Position, yaw and pitch (degrees) of a world-to-camera from w2c (no roll). from_inverse reads the forward
    axis from the inverse instead (equal up to rounding): Fixer's rounds were made that way, kept so they repeat
    bit for bit."""
    c2w = np.linalg.inv(m)
    fwd = c2w[:3, 2] if from_inverse else m[2, :3]
    return c2w[:3, 3], math.degrees(math.atan2(fwd[1], fwd[0])), math.degrees(math.asin(np.clip(fwd[2], -1, 1)))


def intrinsic() -> list:
    """The frames' intrinsics (upstream's)."""
    return [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]]


def intrinsic_fov(w: int, h: int, fov_deg: float) -> np.ndarray:
    """A centred pinhole of horizontal field of view fov_deg."""
    f = w / 2 / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def bank_cameras() -> list[np.ndarray]:
    """Upstream's panorama memory: 9 yaws 40 degrees apart (from the back) at pitch 0, +27 and -27."""
    yaws = [180 - 40 * i for i in range(9)]
    return [w2c([0, 0, 0], y, p) for p in (0, 27, -27) for y in yaws]


def pano_dirs(h: int, w: int) -> np.ndarray:
    """Unit direction of each equirectangular pixel: x at the panorama's centre, y to its left, z up."""
    yaw = (0.5 - (np.arange(w) + 0.5) / w) * 2 * np.pi
    pitch = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    yaw, pitch = np.meshgrid(yaw, pitch)
    return np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], -1)


def crop(pano: np.ndarray, m: np.ndarray, nearest: bool = False) -> np.ndarray:
    """The panorama seen by a frame camera at the origin with world-to-camera m (only its rotation counts); nearest
    for maps that must not be interpolated (distances)."""
    import cv2
    ph, pw = pano.shape[:2]
    j, i = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(j - W / 2) / F, (i - H / 2) / F, np.ones_like(j)], -1) @ m[:3, :3]  # camera -> world
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    yaw, pitch = np.arctan2(rays[..., 1], rays[..., 0]), np.arcsin(np.clip(rays[..., 2], -1, 1))
    u = ((0.5 - yaw / (2 * np.pi)) % 1.0) * pw - 0.5
    v = (0.5 - pitch / np.pi) * ph - 0.5
    return cv2.remap(pano, u.astype(np.float32), v.astype(np.float32),
                     cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def edge_mask(dist: np.ndarray, ratio: float = 1.04) -> np.ndarray:
    """True on depth discontinuities (a 3x3 neighbour more than 4 % nearer or farther): flying points."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    hi, lo = cv2.dilate(dist, k), cv2.erode(dist, k)
    return hi > lo * ratio


def zbuffer(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Depth (camera z) of the nearest point per frame pixel, 0 where none lands; a 3x3 minimum closes the gaps
    between points near the camera."""
    from scipy import ndimage
    p = points @ m[:3, :3].T + m[:3, 3]
    p = p[p[:, 2] > 0.1]
    u = np.floor(p[:, 0] / p[:, 2] * F + W / 2).astype(int)
    v = np.floor(p[:, 1] / p[:, 2] * F + H / 2).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    z = np.full(H * W, np.inf, np.float32)
    np.minimum.at(z, v[ok] * W + u[ok], p[ok, 2])
    z = ndimage.minimum_filter(z.reshape(H, W), 3)
    return np.where(np.isfinite(z), z, 0).astype(np.float32)


def lift(dep, K: np.ndarray, w2cs: np.ndarray):
    """World points (N, H, W, 3) of z-depth maps (torch)."""
    import torch
    dev = dep.device
    n, h, w = dep.shape
    j, i = torch.meshgrid(torch.arange(w, device=dev) + 0.5, torch.arange(h, device=dev) + 0.5, indexing="xy")
    cam = torch.stack([(j - K[0, 2]) / K[0, 0] * dep, (i - K[1, 2]) / K[1, 1] * dep, dep], -1)
    c2w = torch.as_tensor(np.linalg.inv(np.asarray(w2cs)), dtype=torch.float32, device=dev)
    return torch.einsum("nij,nhwj->nhwi", c2w[:, :3, :3], cam) + c2w[:, None, None, :3, 3]


def pano_lookup(pano: np.ndarray, m: np.ndarray, K: np.ndarray, w: int, h: int, up_only: bool = False) -> np.ndarray:
    """The equirectangular image's value along each pixel's ray (nearest), for a w x h camera with intrinsics K;
    up_only: False wherever the ray points down."""
    ph, pw = pano.shape[:2]
    j, i = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    ray = np.stack([(j - K[0, 2]) / K[0, 0], (i - K[1, 2]) / K[1, 1], np.ones_like(j)], -1) @ np.linalg.inv(m)[:3, :3].T
    yaw = np.arctan2(ray[..., 1], ray[..., 0])
    pitch = np.arcsin(ray[..., 2] / np.linalg.norm(ray, axis=-1))
    out = pano[np.clip(((0.5 - pitch / np.pi) * ph).astype(int), 0, ph - 1),
               np.clip(((0.5 - yaw / (2 * np.pi)) * pw).astype(int), 0, pw - 1)]
    return out & (ray[..., 2] > 0) if up_only else out


def wrap(a):
    """Angle difference in degrees, into [-180, 180)."""
    return (np.asarray(a) + 180) % 360 - 180


def smooth(t):
    """Smoothstep easing."""
    return t * t * (3 - 2 * t)


def look_features(xy, yaw_deg) -> np.ndarray:
    """(x, y, 2 cos yaw, 2 sin yaw) per camera: 1 m of position weighs like ~30 degrees of heading."""
    return np.array([[x, y, 2 * math.cos(math.radians(h)), 2 * math.sin(math.radians(h))]
                     for (x, y), h in zip(xy, yaw_deg)])


def farthest(feat: np.ndarray, n: int) -> list[int]:
    """Farthest-point pick of n rows of feat, from the first."""
    pick = [0]
    for _ in range(min(n, len(feat)) - 1):
        d = np.min(np.linalg.norm(feat[:, None] - feat[pick][None], axis=-1), 1)
        pick.append(int(d.argmax()))
    return pick


def write_video(path: Path, frames: np.ndarray):
    """A leg's frames (N, H, W[, 3]) as upstream's 16 fps H.264 (PyAV)."""
    import imageio.v3 as iio
    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, -1)
    iio.imwrite(path, frames, plugin="pyav", codec="libx264", fps=16)


def write_video_cv(path: Path, frames: list):
    """The same with imageio's ffmpeg writer, for the envs without PyAV (gsplat)."""
    import imageio.v2 as iio2
    iio2.mimwrite(path, frames, fps=16, codec="libx264", quality=8, macro_block_size=16)


def video_frames(path: Path, idx: list[float]) -> list[np.ndarray]:
    """Frames at fractions idx of a video (RGB), with OpenCV (the gsplat env has no PyAV)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * (n - 1)))
        out.append(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
    return out
