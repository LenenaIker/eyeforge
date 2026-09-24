import math
import random

import pytest

from sim2chip.scenes.scatter import PropScatter

EYE, TARGET = (20.0, 0.0, 3.0), (0.0, 0.0, 0.0)


class FakeProps:
    def __init__(self):
        self.poses = {}

    def add(self, path, url):
        return path

    def set_pose(self, prim, pos, yaw=0.0, scale=1.0):
        self.poses[prim] = (pos, scale)


def test_hidden_estimate():
    hid = PropScatter.hidden
    assert hid(EYE, TARGET, 3, 3, (-10, 0), 1, 1) == 0  # behind the target
    assert hid(EYE, TARGET, 3, 3, (10, 8), 1, 1) == 0   # in front but off to the side
    full = hid(EYE, TARGET, 3, 3, (10, 0), 3, 6)         # a tall wide prop right in front
    part = hid(EYE, TARGET, 3, 3, (10, 0), 0.3, 0.7)     # a cone right in front
    assert full == pytest.approx(1.0) and 0 < part < 0.3


def test_place_keeps_target_visible_and_props_apart():
    props = FakeProps()
    sc = PropScatter(props, 60)
    sc.add_kind(6, [("cone.usd", (0.2, 0.7), (0.9, 1.1))], "/P/cone")
    sc.add_kind(2, [("container.usd", (3.0, 2.6), (1.0, 1.0))], "/P/container")
    rng, counts = random.Random(0), []
    for _ in range(200):
        props.poses.clear()
        shown, hidden = sc.place(EYE, TARGET, 3.0, 3.0, rng)
        assert hidden <= PropScatter.MAX_HIDDEN
        counts.append(len(shown))
        placed = [(props.poses[p][0], 3.0 if "container" in p else 0.2) for p in shown]
        for i, (a, ra) in enumerate(placed):
            assert math.hypot(*a[:2]) > 3.0 + ra
            for b, rb in placed[i + 1:]:
                assert math.dist(a[:2], b[:2]) > ra + rb
    assert min(counts) == 0 and max(counts) >= 6  # density varies from empty to cluttered
