import json
from pathlib import Path

import pytest

from sim2chip import hardware
from sim2chip.spec import SpecError, camera_distance, from_dict, load

REPO = Path(__file__).resolve().parents[1]


def minimal(**over):
    d = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2.0, 2.5]},
                                   "assets": {"categories": ["cow"], "count": 4}}]}
    d.update(over)
    return d


def test_example_specs_load():
    for f in (REPO / "specs").glob("*.json"):
        spec = load(f)
        hardware.get(spec.hardware)
        assert spec.labels[0] == spec.background_class


def test_defaults_and_labels():
    spec = from_dict(minimal())
    assert spec.labels == ["empty", "cow"]
    assert spec.camera.hfov_deg == 60
    assert spec.cls("cow").size_m.range == (2.0, 2.5)


@pytest.mark.parametrize("bad, msg", [
    ({"classes": []}, "classes"),
    ({"colour": 1}, "unknown keys"),
    ({"camera": {"target_fill": [0.5, 2]}}, "target_fill"),
    ({"qa": {"mode": "magic"}}, "qa.mode"),
    ({"user_data": {"labeled": "x", "class_map": {"horse": ["h"]}}}, "class_map"),
    ({"background_class": "cow"}, "background_class"),
])
def test_invalid_specs(bad, msg):
    with pytest.raises(SpecError, match=msg):
        from_dict(minimal(**bad))


def test_assets_need_size():
    d = minimal()
    del d["classes"][0]["size_m"]
    with pytest.raises(SpecError, match="size_m"):
        from_dict(d)


def test_roundtrip_json():
    spec = from_dict(minimal())
    assert from_dict(json.loads(spec.to_json())) == spec


def test_camera_distance():
    # 60 deg hfov: a 4 m object filling half the width is ~6.9 m away
    assert camera_distance(4.0, 0.5, 60, (1, 100)) == pytest.approx(6.93, abs=0.01)
    assert camera_distance(4.0, 0.01, 60, (1, 100)) == 100
    assert camera_distance(0.1, 0.9, 60, (4, 100)) == 4
