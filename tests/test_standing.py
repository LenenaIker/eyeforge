import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim2chip" / "photo2scene" / "steps"))
import standing  # noqa: E402

UP = np.array([0.0, -1.0, 0.0])  # OpenCV: +Y down


def scene(h=100, w=60):
    """A building whose lowest pixels are row 59, standing 2 m over a floor at y = 0 (its base at y = -2)."""
    mask = np.zeros((h, w), bool)
    mask[10:60] = True
    pts = np.zeros((h, w, 3))
    pts[..., 1] = -2.0
    return mask, pts, (lambda P: np.zeros(len(P)))


def test_a_base_over_the_ground_is_seen():
    mask, pts, floor = scene()
    ground = np.zeros_like(mask)
    ground[60:] = True
    assert abs(standing.photo_gap(mask, pts, UP, floor, ground=ground) - 2.0) < 1e-6


def test_a_base_behind_a_hedge_is_unseen():
    mask, pts, floor = scene()
    ground = np.zeros_like(mask)
    ground[80:] = True  # rows 60-79 are a hedge
    assert standing.photo_gap(mask, pts, UP, floor, ground=ground) is None
    assert abs(standing.photo_gap(mask, pts, UP, floor) - 2.0) < 1e-6  # without the ground masks, as before


def test_building_names():
    assert standing.is_building("002 church") and standing.is_building("townhouses")
    assert not standing.is_building("window shutter") and not standing.is_building("903 hedge")

