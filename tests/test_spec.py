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
    # Specs come from an LLM: wrong types and names must be SpecErrors, not crashes later on.
    ({"hardware": "esp8266"}, "hardware"),
    ({"generation": {"frames": "many"}}, "generation.frames"),
    ({"classes": ["cow"]}, "classes"),
    ({"classes": [{"id": "cow", "size_m": {"measure": "length", "range": ["a", 2]}}]}, "range"),
    ({"classes": [{"id": "cow", "assets": {"categories": ["cow"], "count": "12"}}]}, "count"),
])
def test_invalid_specs(bad, msg):
    with pytest.raises(SpecError, match=msg):
        from_dict(minimal(**bad))


def test_photo_assets():
    photos = {"id": "cow", "size_m": {"measure": "length", "range": [2.0, 2.5]},
              "assets": {"source": "photos", "photos": "user_data", "queries": ["cow"]}}
    assert from_dict(minimal(classes=[photos])).cls("cow").assets.n_generate == 4
    with pytest.raises(SpecError, match="photos folder"):
        from_dict(minimal(classes=[{**photos, "assets": {"source": "photos"}}]))
    with pytest.raises(SpecError, match="generate"):
        from_dict(minimal(classes=[{**photos, "assets": {"source": "photos", "photos": "p", "generate": 0,
                                                         "queries": ["cow"]}}]))
    with pytest.raises(SpecError, match="English name"):  # the class id may be "vaca": the detector needs English
        from_dict(minimal(classes=[{**photos, "assets": {"source": "photos", "photos": "p"}}]))
    # The new optional fields stay out of fingerprints until they are used.
    assert set(from_dict(minimal()).section("classes")[0]["assets"]) == {"source", "categories", "count"}


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


def test_empty_optional_fields_do_not_change_fingerprints():
    from sim2chip.spec import fingerprint
    base = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                      "assets": {"categories": ["cow"]}}]}
    s = from_dict(base)
    assert "queries" not in s.section("classes")[0]["assets"]  # empty list left out
    explicit = from_dict({**base, "classes": [{**base["classes"][0],
                                               "assets": {"categories": ["cow"], "queries": []}}]})
    assert fingerprint(s.section("classes")) == fingerprint(explicit.section("classes"))


PROP = {"id": "cone", "queries": ["traffic cone"], "size_m": {"measure": "height", "range": [0.5, 0.9]}}


def test_props_parse_and_size_of():
    base = {"name": "x", "classes": [{"id": "car", "size_m": {"measure": "length", "range": [3, 5]}}]}
    s = from_dict({**base, "props": [PROP]})
    assert s.props[0].count == 4 and s.size_of("cone").range == (0.5, 0.9) and s.size_of("car").range == (3, 5)
    for bad in ({**PROP, "id": "car"}, {**PROP, "queries": []}, {**PROP, "max_per_frame": 0},
                {k: v for k, v in PROP.items() if k != "size_m"}):
        with pytest.raises(SpecError):
            from_dict({**base, "props": [bad]})


def test_empty_props_keep_fingerprints():
    from sim2chip.pipeline import Pipeline
    from sim2chip.workspace import Cache, Job
    base = {"name": "x", "classes": [{"id": "car", "size_m": {"measure": "length", "range": [3, 5]},
                                      "assets": {"categories": ["car"]}}]}
    job, cache = Job(Path("/nonexistent/j")), Cache(Path("/nonexistent/c"))
    a = Pipeline(from_dict(base), job, cache).fingerprints()
    b = Pipeline(from_dict({**base, "props": []}), job, cache).fingerprints()
    c = Pipeline(from_dict({**base, "props": [PROP]}), job, cache).fingerprints()
    assert a == b and a["fetch_assets"] != c["fetch_assets"] and a["train"] != c["train"]  # downstream too
