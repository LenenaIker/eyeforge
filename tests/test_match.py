"""Camera look (look.py) and scene levers (match.py): no GPU, no model."""
import random

import numpy as np
import pytest
from PIL import Image

from sim2chip import look, match
from sim2chip.spec import from_dict, to_dict

SPEC = {"name": "t", "task": "count",
        "classes": [{"id": "pot", "size_m": {"measure": "length", "range": [0.2, 0.6]},
                     "assets": {"source": "search", "queries": ["flower pot"]}}],
        "scene": {"template": "generic", "contexts": [{"kind": "balcony", "share": 0.5},
                                                      {"kind": "indoor", "share": 0.5}]}}


def _img(color, noise=0.0, seed=0):
    r = np.random.default_rng(seed)
    a = np.ones((64, 64, 3)) * np.array(color) / 255
    a[16:48, 16:48] = 1 - a[16:48, 16:48]  # an edge to measure sharpness on
    a = np.clip(a + r.normal(0, noise, a.shape), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def test_lab_round_trip():
    rgb = np.random.default_rng(0).random((10, 3))
    assert np.allclose(look.from_lab(look.to_lab(rgb)), rgb, atol=1e-3)


def test_transform_moves_renders_to_the_photos():
    photos = [_img((200, 150, 90), noise=0.03, seed=i) for i in range(6)]
    renders = [_img((100 + 10 * i, 110, 130), noise=0.005, seed=i) for i in range(6)]
    user, synth = look.measure(photos), look.measure(renders)
    assert user["noise"] > synth["noise"]
    t = look.fit(user, synth)
    assert t.noise > 0
    rng = random.Random(0)
    after = look.measure([look.apply(r, t, rng) for r in renders])
    assert look.distance(user, after)["color"] < look.distance(user, synth)["color"] / 2
    flat = look.fit(user, look.measure([_img((120, 120, 120))]))  # no spread at all: stretched, not exploded
    assert np.asarray(look.apply(_img((120, 120, 120)), flat, rng)).std() < 80


def test_no_blur_when_photos_are_sharper():
    assert look.blur_for(0.2, 0.3) == 0.0
    assert 0 < look.blur_for(0.3, 0.2) <= look.MAX_BLUR


def test_spec_look_round_trip():
    s = from_dict({**SPEC, "look": {"lab_mean": [50, 1, 2], "lab_std": [20, 5, 9], "sharpness": 0.2,
                                    "noise": 0.01}})
    assert from_dict(to_dict(s)).look.lab_mean == [50, 1, 2]
    assert from_dict(SPEC).look is None
    assert from_dict(SPEC).section("look") is None  # old specs keep their stage fingerprints


def test_patch_applies_levers_and_rejects_the_rest():
    s = from_dict(SPEC)
    new, applied, rejected = match.patch(s, [
        {"lever": "scene.contexts.balcony.share", "value": 0.8},
        {"lever": "scene.contexts.ground.share", "value": 0.3},     # a new place kind
        {"lever": "camera.height_m", "value": [5, 0.5]},             # sorted
        {"lever": "generation.empty_ratio", "value": 0.95},          # clipped to the range
        {"lever": "scene.weather", "value": ["clear", "snow"]},      # not a weather
        {"lever": "classes.pot.size_m", "value": [1, 2]},            # not a lever
        {"lever": "props.add", "value": {"id": "watering can", "queries": ["watering can"],
                                         "size_m": {"measure": "length", "range": [0.3, 0.5]}}},
    ])
    assert [a["lever"] for a in applied] == ["scene.contexts.balcony.share", "scene.contexts.ground.share",
                                             "camera.height_m", "generation.empty_ratio", "props.add"]
    assert [r["lever"] for r in rejected] == ["scene.weather", "classes.pot.size_m"]
    shares = {c.kind: c.share for c in new.scene.contexts}
    assert shares == {"balcony": 0.8, "indoor": 0.5, "ground": 0.3}
    assert new.camera.height_m == (0.5, 5.0)
    assert new.generation.empty_ratio == 0.6
    assert new.props[0].id == "watering_can"
    assert s.props == []  # the original spec is untouched


def test_patch_removes_a_place_but_not_the_last():
    s = from_dict(SPEC)
    new, applied, _ = match.patch(s, [{"lever": "scene.contexts.indoor.share", "value": 0}])
    assert [c.kind for c in new.scene.contexts] == ["balcony"]
    _, applied, rejected = match.patch(new, [{"lever": "scene.contexts.balcony.share", "value": 0}])
    assert not applied and "at least one" in rejected[0]["error"]


def test_lever_values_lists_what_applies():
    v = match.lever_values(from_dict(SPEC))
    assert v["scene.contexts.balcony.share"] == 0.5
    assert "generation.max_targets" in v
    assert "scene.contexts.facade.share" not in v


def test_kid_is_lower_for_the_same_distribution():
    r = np.random.default_rng(0)
    a, b, c = r.normal(0, 1, (40, 8)), r.normal(0, 1, (40, 8)), r.normal(2, 1, (40, 8))
    assert match.kid(a, b) < match.kid(a, c)


def test_reference_photos_only_from_the_training_side(tmp_path):
    import json
    from sim2chip.workspace import content_key, open_job
    job = open_job(tmp_path, "j")
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(4):
        _img((i * 40, 0, 0)).save(raw / f"{i}.png")
    (job.root / "ingest").mkdir()
    sides = {content_key(raw / f"{i}.png"): "train" if i % 2 else "test" for i in range(4)}
    (job.root / "ingest" / "split.json").write_text(json.dumps(sides))
    spec = from_dict({**SPEC, "user_data": {"raw": str(raw)}})
    assert sorted(p.name for p in match.reference_photos(spec, job)) == ["1.png", "3.png"]


@pytest.mark.parametrize("value", [None, "x", [1]])
def test_pair_levers_need_two_numbers(value):
    _, applied, rejected = match.patch(from_dict(SPEC), [{"lever": "camera.target_fill", "value": value}])
    assert not applied and rejected


def test_patch_checks_tags_and_takes_prop_lists():
    ok = lambda lever, tags: tags == ["grass"]  # noqa: E731
    new, applied, rejected = match.patch(from_dict(SPEC), [
        {"lever": "scene.ground_tags", "value": ["Grass"]},
        {"lever": "scene.ground_tags", "value": ["tiles", "terracotta"]},
        {"lever": "props.add", "value": [
            {"id": "chair", "queries": ["chair"], "size_m": {"measure": "height", "range": [0.8, 1]}},
            {"id": "table", "queries": ["table"], "size_m": {"measure": "length", "range": [0.8, 1.5]}}]},
    ], tags_ok=ok)
    assert new.scene.ground_tags == ["grass"]
    assert "no Poly Haven asset" in rejected[0]["error"]
    assert [p.id for p in new.props] == ["chair", "table"]


def test_context_levers_pass_the_tag_check():
    ok = lambda lever, tags: True  # noqa: E731
    new, applied, rejected = match.patch(from_dict(SPEC), [
        {"lever": "scene.contexts.balcony.share", "value": 0.7},
        {"lever": "camera.pitch_deg", "value": [60, 89]}], tags_ok=ok)
    assert not rejected and new.camera.pitch_deg == (60.0, 89.0)


def test_locked_levers_are_not_changed():
    _, applied, rejected = match.patch(from_dict(SPEC), [{"lever": "camera.height_m", "value": [0.1, 0.3]}],
                                       locked=("camera.height_m",))
    assert not applied and "fixed by the user" in rejected[0]["error"]
