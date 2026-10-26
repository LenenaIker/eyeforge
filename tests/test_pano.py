"""The scene route's panorama step (experiments/worldstereo/wsroute/pano.py): made in the eraser's env, pasted in
the scene env, and the numpy placement the same as OpenCV's, bit for bit (the 4-step edit turns any change of its
input into another picture)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "worldstereo"))
from wsroute import pano, steps  # noqa: E402
from wsroute.config import RouteConfig  # noqa: E402
from wsroute.paths import ENVS, PANOS  # noqa: E402


def test_the_pano_is_made_in_the_erasers_env_and_pasted_in_the_scene_env():
    by_name = {s.name: s for s in steps.recipe(RouteConfig(case="t", photo="p", pano_seed=2))}
    assert "pano" not in ENVS  # the FLUX env is gone
    assert by_name["pano"].env == "erase" and by_name["pano_paste"].env == "scene"
    names = list(by_name)
    assert names.index("caption") < names.index("pano") < names.index("pano_paste") < names.index("walks")
    assert by_name["pano"].done.__self__ == PANOS / "t" / "pano_s2.jpg"
    assert by_name["pano_paste"].done.__self__ == PANOS / "t" / "pano_s2_paste.jpg"


def test_the_placed_photo_covers_its_field_of_view_at_the_centre():
    w, h, fx = 720, 360, 100.0  # a 200 px wide photo at fx 100: 90 degrees across
    canvas, inside = pano.place(Image.new("RGB", (200, 100), (200, 50, 10)), fx, w, h)
    cols = inside[h // 2].nonzero()[0]
    assert abs(cols.mean() - (w - 1) / 2) < 1 and abs(len(cols) - w / 4) <= 2
    assert (canvas[inside == 1] == (200, 50, 10)).all() and not canvas[inside == 0].any()


def test_the_numpy_placement_matches_opencv():
    cv2 = pytest.importorskip("cv2")  # the scene env has it; the pano step's env does not
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 256, (90, 160, 3), dtype=np.uint8)
    canvas, inside = pano.place(photo, 120.0, 512, 256, yaw=10, pitch=-5, roll=3, k1=0.1)
    u, v, ins = pano.photo_coords(160, 90, 120.0, 512, 256, 10, -5, 3, 0.1)
    ref = cv2.remap(photo, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    assert (inside == ins).all() and (canvas == ref).all()  # bit for bit: the edit is that sensitive
