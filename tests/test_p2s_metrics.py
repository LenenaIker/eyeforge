import numpy as np

from sim2chip.photo2scene.metrics import base_float

UP = np.array([0.0, 0.0, 1.0])


def box(tilt):
    g = np.linspace(0, 1, 15)
    P = np.array([[a * 10, b * 6, c * 8] for a in g for b in g for c in (0, 1)], float)
    P[:, 2] += tilt * P[:, 0]  # the base rises tilt metres per metre along x
    return P


def test_a_level_base_does_not_float():
    assert base_float(box(0.0), UP) < 1e-6


def test_a_tilted_base_floats_at_one_end():
    assert 0.85 < base_float(box(0.1), UP) < 1.0  # 1 m over 10 m, a little less slice to slice
