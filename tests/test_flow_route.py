import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim2chip" / "photo2scene" / "steps"))
import flow_route  # noqa: E402

STREET = {"outdoor": True, "time_of_day": "dusk_or_dawn", "weather": "overcast", "camera_height": "pedestrian",
          "ground_flat": "flat", "place": "village_street", "layout": "corner", "route": "out_of_scope",
          "template": "none", "ground": [{"phrase": "asphalt"}], "surroundings": [{"phrase": "house"}]}


def photo(tmp_path, rgb):
    p = tmp_path / "photo.png"
    Image.fromarray(np.full((64, 96, 3), rgb, np.uint8)).save(p)
    return p


def test_a_bright_grey_photo_is_day_whatever_the_vlm_said(tmp_path):
    light = flow_route.measure_light(photo(tmp_path, (120, 122, 125)))
    assert light["day"]
    r = flow_route.route({"final": dict(STREET), "unsure": ["time_of_day"]}, light)
    assert r["route"] == "village_street_day" and r["template"] == "crossroads" and not r["why"]
    assert r["light"]["vlm_time_of_day"] == "dusk_or_dawn"


def test_a_dark_orange_photo_is_night(tmp_path):
    light = flow_route.measure_light(photo(tmp_path, (90, 55, 25)))
    assert not light["day"]
    assert flow_route.route({"final": dict(STREET, time_of_day="day", route="village_street_day")}, light)[
        "route"] == "out_of_scope"


def test_without_a_measure_the_vlm_decides():
    assert flow_route.route({"final": dict(STREET)})["why"] == ["fails: daylight"]
