"""Count task without TensorFlow or Isaac Sim: decoding, scoring, placement, points, teacher marks, review."""
import json
import random
import threading

import numpy as np
import pytest
from PIL import Image

from sim2chip import counting, llm, review
from sim2chip.scenes.scatter import cluster
from sim2chip.spec import SpecError, from_dict
from sim2chip.stages import Context
from sim2chip.stages.ingest import Ingest
from sim2chip.stages.label import Label, objects_from
from sim2chip.stages.points import Points
from sim2chip.workspace import Cache, link_input, open_job

LABELS = ["empty", "pot", "cat"]
SPEC = {"name": "t", "task": "count", "classes": [
    {"id": "pot", "description": "flower pots", "size_m": {"measure": "height", "range": [0.2, 0.4]},
     "assets": {"source": "search", "queries": ["flower pot"]}},
    {"id": "cat", "size_m": {"measure": "length", "range": [0.4, 0.5]}, "assets": {"categories": ["cat"]}}]}


def grid(cells: dict, rows=6, cols=8):
    g = np.zeros((rows, cols, 3))
    g[..., 0] = 1.0
    for (i, j), (c, p) in cells.items():
        g[i, j] = 0
        g[i, j, 0], g[i, j, c] = 1 - p, p
    return g


def test_decode_blobs_are_objects():
    g = grid({(1, 1): (1, 0.9), (1, 2): (1, 0.7), (4, 6): (1, 0.8), (4, 5): (2, 0.9), (0, 7): (1, 0.3)})
    pts = counting.decode(g, LABELS)
    assert len(pts) == 3  # two touching pot cells are one pot; a cat next to a pot is its own object
    pot = next(p for p in pts if p[1] < 0.5)
    assert pot[2] == "pot" and 1.5 / 8 < pot[0] < 2.5 / 8 and pot[1] == round(1.5 / 6, 4)
    assert {p[2] for p in pts} == {"pot", "cat"}  # the 0.3 cell is below the threshold


def test_match_and_score():
    truth = [[(0.1, 0.1, "pot"), (0.5, 0.5, "pot")], [], [(0.9, 0.9, "cat")]]
    pred = [[(0.11, 0.1, "pot")], [(0.3, 0.3, "pot")], [(0.9, 0.9, "pot")]]
    s = counting.score(pred, truth, grid=28)
    # tp 1; fp: the pot in an empty photo and the cat called pot; fn: the missed pot and the cat
    assert (s["precision"], s["recall"]) == (round(1 / 3, 4), round(1 / 3, 4))
    assert s["count_mae"] == round((1 + 1 + 2) / 3, 3) and s["exact"] == 0.0
    assert counting.match([(0.5, 0.5, "pot")], [(0.5, 0.58, "pot")], radius=2 / 28) == (0, 1, 1)


def test_cluster_never_overlaps():
    rng = random.Random(0)
    radii = [0.3] * 6
    pos = cluster((1.0, 2.0, 0.0), radii, spread=3.0, rng=rng)
    assert pos[0] == (1.0, 2.0, 0.0)
    placed = [p for p in pos if p is not None]
    assert len(placed) >= 5
    for i, a in enumerate(placed):
        assert np.hypot(a[0] - 1, a[1] - 2) <= 3.0 + 1e-9
        for b in placed[i + 1:]:
            assert np.hypot(a[0] - b[0], a[1] - b[1]) > 0.6
    assert cluster((0, 0, 0), [1.0, 1.0], spread=0.5, rng=rng)[1] is None  # no room


def test_count_spec_rules():
    s = from_dict(SPEC)
    assert s.task == "count" and s.generation.n_targets == 6 and s.section("task") == "count"
    with pytest.raises(SpecError, match="task must be"):
        from_dict({**SPEC, "task": "segment"})
    with pytest.raises(SpecError, match="user_data.raw"):
        from_dict({**SPEC, "user_data": {"labeled": "x", "class_map": {"pot": ["p"]}}})


def ctx_for(tmp_path, **over):
    return Context(from_dict({**SPEC, **over}), open_job(tmp_path, "job"), Cache(tmp_path / "cache"))


def test_points_from_renders(tmp_path):
    ctx = ctx_for(tmp_path)
    r = ctx.job.stage_dir("render")
    r.mkdir(parents=True)
    dt = [("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
          ("occlusionRatio", "<f4")]
    rows = [(0, 10, 20, 50, 60, 0.1),    # pot
            (0, 100, 100, 102, 140, 0.0),  # too thin to count
            (1, 200, 100, 260, 200, 0.95),  # cat, almost hidden
            (1, 280, 10, 300, 40, 0.2),    # cat
            (2, 0, 0, 320, 240, 0.0)]      # a prop: not a class
    for f in range(2):
        Image.new("RGB", (320, 240)).save(r / f"rgb_{f:04d}.png")
    np.save(r / "bounding_box_2d_tight_0000.npy", np.array(rows, dtype=dt))
    (r / "bounding_box_2d_tight_labels_0000.json").write_text(json.dumps(
        {"0": {"class": "pot"}, "1": {"class": "cat"}, "2": {"class": "bench"}}))
    (r / "labels.json").write_text(json.dumps([{"frame": 0, "class": "pot"}, {"frame": 1, "class": "empty"}]))
    info = Points().run(ctx)
    pts = json.loads((ctx.job.stage_dir("points") / "points.json").read_text())
    assert info == {"frames": 2, "objects": 2} and pts["00001.png"] == []
    assert pts["00000.png"] == [[0.0938, 0.1667, "pot"], [0.9062, 0.1042, "cat"]]
    assert Image.open(ctx.job.stage_dir("points") / "images" / "00000.png").size == (96, 96)


class FakeClient:
    answer = staticmethod(lambda text: "{}")
    calls = []

    def __init__(self, usage_log=None, **kw):
        pass

    def model_for(self, role):
        return f"fake-{role}"

    def chat(self, messages, model, purpose, **kw):
        FakeClient.calls.append(messages[0]["content"][0]["text"])
        return FakeClient.answer(messages[0]["content"][0]["text"])


def test_objects_from_boxes():
    ans = {"objects": [{"class": "pot", "box": [100, 50, 300, 150]}, {"class": "dog", "box": [0, 0, 10, 10]},
                       {"class": "cat", "box": [390, 190, 900, 900]}, {"class": "pot", "box": [5, 5, 5, 20]}]}
    objs = objects_from(ans, ["pot", "cat"], 400, 200)
    assert [(o["x"], o["y"], o["class"]) for o in objs] == [(0.5, 0.5, "pot"), (0.9875, 0.975, "cat")]
    assert objs[0]["box"] == [0.25, 0.25, 0.75, 0.75]


def test_label_marks_then_ingest_then_review_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Client", FakeClient)
    FakeClient.calls = []
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        Image.new("RGB", (400, 200), (10 * i, 0, 0)).save(raw / f"p{i}.jpg")
    lock, n = threading.Lock(), iter(range(100))

    def answer(text):
        assert "400 x 200 pixels" in text
        with lock:
            i = next(n)
        if i == 0:
            return "no idea"  # twice for the same photo: it ends up unsure
        return json.dumps({"objects": [{"class": "pot", "box": [0, 0, 40, 20]}] * (i % 3)})
    FakeClient.answer = staticmethod(answer)
    ctx = ctx_for(tmp_path, user_data={"raw": "raw", "test_per_class": 2, "train_per_class": 10})
    link_input(ctx.job, "raw", raw)
    info = Label().run(ctx)
    assert info["labels"]["images"] + info["labels"].get("unsure", 0) == 6
    counts = Ingest().run(ctx)["counts"]
    assert counts["test"]["photos"] == 2 and counts["test"]["photos"] + counts["train"]["photos"] == \
        info["labels"]["images"]
    table = json.loads((ctx.job.stage_dir("ingest") / "test" / "points.json").read_text())
    some = next(iter(table.values()))
    assert some["source"].startswith("label/images/") and all(p[2] == "pot" for p in some["points"])

    got = review.gather(ctx.job, ctx.cache, ctx.spec.labels, task="count")
    assert got["task"] == "count" and len(got["photos"]) == 6
    key = got["photos"][0]["key"]
    assert review.apply(ctx.job, ctx.cache, {}, {}, {key: [[0.2, 0.3, "pot"], [0.7, 0.3, "nothing"]]}) == "label"
    assert review.apply(ctx.job, ctx.cache, {}, {}, {key: [[0.2, 0.3, "pot"], [0.7, 0.3, "nothing"]]}) is None
    calls = len(FakeClient.calls)
    Label().run(ctx)
    assert len(FakeClient.calls) - calls <= 2  # only the unsure photo is asked again
    v = json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text())[key]
    assert v["judge"] == "user" and [(o["x"], o["class"]) for o in v["objects"]] == [(0.2, "pot")]
    assert "teacher_objects" in v


def test_best_threshold_drops_weak_false_positives():
    g = grid({(1, 1): (1, 0.95), (4, 6): (1, 0.55)})  # one real pot, one weak blob that is not a pot
    t, s = counting.best_threshold([g], [[(1.5 / 8, 1.5 / 6, "pot")]], LABELS, grid=8)
    assert t > 0.55 and s["f1"] == 1.0
