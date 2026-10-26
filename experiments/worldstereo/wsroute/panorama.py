"""The panorama as metric points: MoGe-2's distance map (walks.prep caches it) scaled by prep's ground fit."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import camera
from .paths import PANOS, Case

PANO_W = 2048  # the point cloud's resolution (2M points, about upstream's density per view)


def contract(d: np.ndarray) -> np.ndarray:
    """Upstream's depth contraction for the guide renders: beyond 8x the median distance, squashed to 16x."""
    mid = np.median(d) * 8
    return np.where(d > mid, 2 * mid - mid ** 2 / (d + 1e-6), d)


class Panorama:
    """prep.json's panorama of a case: the full image, the PANO_W copy, the scaled distance and its masks."""

    def __init__(self, case: Case):
        import cv2
        self.meta = json.loads(case.prep_json.read_text())
        self.full = np.asarray(Image.open(PANOS / case.name / self.meta["pano"]).convert("RGB"))
        self.small = cv2.resize(self.full, (PANO_W, PANO_W // 2), interpolation=cv2.INTER_AREA)
        z = np.load(case.distance)
        self.dist, self.valid = z["distance"] * self.meta["scale"], z["mask"]
        self.keep = self.valid & ~camera.edge_mask(self.dist) & np.isfinite(self.dist) & (self.dist > 0)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """Metric points (no far contraction) and their colours (uint8)."""
        return camera.pano_dirs(*self.dist.shape)[self.keep] * self.dist[self.keep][:, None], self.small[self.keep]

    def guide_cloud(self):
        """The guide renders' point cloud (contracted) and colours in [-1, 1], as torch tensors."""
        import torch
        d = contract(self.dist[self.keep])
        return (torch.tensor(camera.pano_dirs(*self.dist.shape)[self.keep] * d[:, None], dtype=torch.float32),
                torch.tensor(self.small[self.keep] / 127.5 - 1, dtype=torch.float32))
