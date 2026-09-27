"""Pure helpers of the cloud worker and of convert_assets: which job files a run pulls, and model footprints."""
import math

import numpy as np

from sim2chip.cloud_sync import wanted
from sim2chip.stages import BY_NAME
from sim2chip.stages.convert_assets import min_area_rect


def test_worker_pulls_only_what_the_pending_stages_read():
    train = [BY_NAME["train"]]
    assert wanted("crops/cow/0001.png", train)          # an upstream output
    assert not wanted("render/rgb_0001.png", train)     # not read by train
    assert not wanted("input/photos/a.jpg", train)      # train does not read the input
    label = [BY_NAME["label"]]
    assert wanted("input/photos/a.jpg", label)
    assert wanted("label/verdicts.json", label)          # label reads back its own verdicts
    assert not wanted("label/_small/000001.jpg", label)
    assert wanted("fetch_assets/scene.json", [BY_NAME["render"]])  # the asset stages need its manifest


def test_min_area_rect_of_a_rotated_box():
    a = math.radians(30)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    corners = np.array([[-2, -0.5], [2, -0.5], [2, 0.5], [-2, 0.5]]) @ rot.T + [5, 1]
    inside = np.random.default_rng(0).uniform([-2, -0.5], [2, 0.5], (200, 2)) @ rot.T + [5, 1]
    angle, length, width, center = min_area_rect(np.vstack([corners, inside]))
    assert abs(length - 4) < 1e-6 and abs(width - 1) < 1e-6
    assert np.allclose(center, [5, 1])
    assert abs(math.cos(angle - a)) > 1 - 1e-6  # the long side, in either direction
