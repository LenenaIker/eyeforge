"""Count task without TensorFlow, Isaac Sim or a GPU: decoding, scoring, placement, points, teacher marks, review."""
import json
import random

import numpy as np
import pytest
from PIL import Image

from sim2chip import counting, review, teacher
from sim2chip.scenes.scatter import cluster
from sim2chip.spec import SpecError, from_dict
from sim2chip.stages import Context
from sim2chip.stages.ingest import Ingest
from sim2chip.stages.label import Label
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
    assert pts["00000.png"] == [[0.0938, 0.1667, "pot", 0.125, 0.1667], [0.9062, 0.1042, "cat", 0.0625, 0.125]]
    assert Image.open(ctx.job.stage_dir("points") / "images" / "00000.png").size == (96, 96)


def fake_sam3(monkeypatch, answer):
    """SAM 3 without a GPU: answer(photo path) gives its detections; returns the photos it was asked about."""
    asked = []

    of = {"flower pot": "pot", "cat": "cat"}  # queries, else categories

    def detect(items, log=print, timeline=None):
        for path, outs in items:
            assert set(outs) <= set(of)
            asked.append(path)
            a = answer(path)
            for phrase, out in outs.items():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(a if "error" in a else {"detections": [
                    {"box": d["box"], "score": d["score"]} for d in a["detections"] if d["class"] == of[phrase]]}))
    monkeypatch.setattr(teacher, "detect", detect)
    return asked


def box(x0, y0, x1, y1, score, cls="pot"):
    return {"class": cls, "box": [x0, y0, x1, y1], "score": score}


def test_teacher_threshold_follows_the_user_preference(monkeypatch):
    monkeypatch.setattr(teacher, "MIN_MARKS", 3)
    dets = {"a": [box(0.1, 0.1, 0.3, 0.3, 0.9), box(0.5, 0.5, 0.7, 0.7, 0.3)],
            "b": [box(0.1, 0.1, 0.3, 0.3, 0.8), box(0.6, 0.1, 0.8, 0.3, 0.4), box(0.6, 0.5, 0.8, 0.7, 0.35)]}
    truth = {"a": [[0.2, 0.2, "pot"], [0.6, 0.6, "pot"]], "b": [[0.2, 0.2, "pot"]]}
    # Only the 0.3 box among the weak ones is a pot: balanced drops all three (ties go to the default).
    assert teacher.choose("balanced", dets, truth)["threshold"] == 0.5
    # Missing a pot hurts more: a lower threshold keeps it and pays two false marks.
    assert teacher.choose("find_more", dets, truth)["threshold"] == 0.3
    # Too few corrected photos: the default of the preference.
    assert teacher.choose("fewer_false", dets, {"a": truth["a"]}) == {
        "threshold": 0.6, "preference": "fewer_false", "photos": 1}
    # Too few marked objects, however many photos: the default too.
    monkeypatch.setattr(teacher, "MIN_MARKS", 4)
    assert teacher.choose("find_more", dets, truth)["threshold"] == 0.4
    objs = teacher.objects(dets["a"], 0.5)
    assert [(o["x"], o["y"], o["class"]) for o in objs] == [(0.2, 0.2, "pot")] and objs[0]["box"] == [0.1, 0.1, 0.3, 0.3]


def test_label_marks_then_ingest_then_review_fix(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(6):
        Image.new("RGB", (400, 200), (10 * i, 0, 0)).save(raw / f"p{i}.jpg")

    def answer(path):
        i = int(path.stem[1:])
        if i == 0:
            return {"error": "OSError: broken"}  # that photo ends up unsure
        # i % 3 sure pots and one doubtful box below the default threshold
        return {"detections": [box(0.1 * j, 0, 0.1 * j + 0.1, 0.1, 0.9) for j in range(i % 3)] +
                [box(0.6, 0.2, 0.8, 0.4, 0.35)]}
    asked = fake_sam3(monkeypatch, answer)
    ctx = ctx_for(tmp_path, user_data={"raw": "raw", "test_per_class": 2, "train_per_class": 10})
    link_input(ctx.job, "raw", raw)
    info = Label().run(ctx)
    assert info["labels"] == {"images": 5, "unsure": 1} and info["threshold"] == 0.5 and len(asked) == 6
    assert info["objects"] == sum(i % 3 for i in range(1, 6))
    counts = Ingest().run(ctx)["counts"]
    assert counts["test"]["photos"] == 2 and counts["test"]["photos"] + counts["train"]["photos"] == 5
    table = json.loads((ctx.job.stage_dir("ingest") / "test" / "points.json").read_text())
    some = next(iter(table.values()))
    assert some["source"].startswith("label/images/") and all(p[2] == "pot" for p in some["points"])

    got = review.gather(ctx.job, ctx.cache, ctx.spec.labels, task="count")
    assert got["task"] == "count" and len(got["photos"]) == 6 and "50%" in got["teacher_rules"][0]
    # The user marks the doubtful box as a pot on two photos: the teacher lowers its threshold for all.
    fixes = {"p1.jpg": [[0.05, 0.05, "pot"], [0.7, 0.3, "pot"], [0.9, 0.9, "nothing"]],
             "p2.jpg": [[0.05, 0.05, "pot"], [0.15, 0.05, "pot"], [0.7, 0.3, "pot"]]}
    assert review.apply(ctx.job, ctx.cache, {}, {}, fixes) == "label"
    assert review.apply(ctx.job, ctx.cache, {}, {}, fixes) is None
    info = Label().run(ctx)
    assert info["threshold"] <= 0.35 and len(asked) == 7  # only the failed photo asked again
    v = json.loads((ctx.job.stage_dir("label") / "verdicts.json").read_text())
    assert v["p1.jpg"]["judge"] == "user" and [(o["x"], o["class"]) for o in v["p1.jpg"]["objects"]] == [
        (0.05, "pot"), (0.7, "pot")]
    assert len(v["p1.jpg"]["teacher_objects"]) == 2 and len(v["p4.jpg"]["objects"]) == 2
    assert "2 corrected photos" in review.gather(ctx.job, ctx.cache, ctx.spec.labels, task="count")["teacher_rules"][0]


def test_photos_with_the_same_copy_name_keep_their_own_points(tmp_path, monkeypatch):
    """day1/cam/p.jpg, day2/cam/p.jpg and cam_p.jpg all had the copy name cam_p.jpg: one copy overwrote the
    others and a photo's points went with another photo's pixels."""
    raw = tmp_path / "raw"
    for n, rel in enumerate(["day1/cam/p.jpg", "day2/cam/p.jpg", "cam_p.jpg"], 1):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 200), (60 * n, 0, 0)).save(raw / rel)  # its red says how many pots it has

    def answer(path):
        n = round(Image.open(path).getpixel((5, 5))[0] / 60)
        return {"detections": [box(0.1 * j, 0, 0.1 * j + 0.1, 0.1, 0.9) for j in range(n)]}
    fake_sam3(monkeypatch, answer)
    ctx = ctx_for(tmp_path, user_data={"raw": "raw", "test_per_class": 1, "train_per_class": 10})
    link_input(ctx.job, "raw", raw)
    Label().run(ctx)
    assert len(list((ctx.job.stage_dir("label") / "images").iterdir())) == 3
    Ingest().run(ctx)
    seen = 0
    for side in ("test", "train"):
        d = ctx.job.stage_dir("ingest") / side
        for name, entry in json.loads((d / "points.json").read_text()).items():
            n = round(Image.open(d / "images" / name).getpixel((5, 5))[0] / 60)
            assert len(entry["points"]) == n
            seen += 1
    assert seen == 3
    photos = review.gather(ctx.job, ctx.cache, ctx.spec.labels, task="count")["photos"]
    assert len({p["image"] for p in photos}) == 3 and all(p["image"] for p in photos)


def test_class_without_3d_models_is_looked_for_in_english(tmp_path, monkeypatch):
    """No 3D search, so no English queries: the id "maceta" is translated once, cached, then given to SAM 3."""
    spec = {"name": "t", "task": "count", "classes": [
        {"id": "maceta", "description": "macetas desde cualquier ángulo", "size_m": {"measure": "length", "range": [0.1, 0.5]}}]}
    asked = []
    monkeypatch.setattr(teacher, "english_phrases", lambda m: asked.append(m) or {"maceta": ["flower pot", "plant pot"]})
    ctx = Context(from_dict(spec), open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    assert Label._phrases(ctx) == {"maceta": ["flower pot", "plant pot"]}
    assert Label._phrases(ctx) == {"maceta": ["flower pot", "plant pot"]} and len(asked) == 1
    assert asked[0] == {"maceta": "macetas desde cualquier ángulo"}
    # Classes with English queries are never sent.
    assert Label._phrases(ctx_for(tmp_path / "b")) == {"pot": ["flower pot"], "cat": ["cat"]}


def test_label_stops_when_the_teacher_finds_nothing_the_user_marked(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        Image.new("RGB", (400, 200), (10 * i, 0, 0)).save(raw / f"p{i}.jpg")
    fake_sam3(monkeypatch, lambda path: {"detections": []})
    ctx = ctx_for(tmp_path, user_data={"raw": "raw", "test_per_class": 1, "train_per_class": 10})
    link_input(ctx.job, "raw", raw)
    Label().run(ctx)  # nothing marked by the user yet: nothing to contradict
    review.apply(ctx.job, ctx.cache, {}, {}, {"p1.jpg": [[0.5, 0.5, "pot"]]})
    with pytest.raises(RuntimeError, match="found nothing in any of the 3 photos"):
        Label().run(ctx)


def test_best_threshold_drops_weak_false_positives():
    g = grid({(1, 1): (1, 0.95), (4, 6): (1, 0.55)})  # one real pot, one weak blob that is not a pot
    t, s = counting.best_threshold([g], [[(1.5 / 8, 1.5 / 6, "pot")]], LABELS, grid=8)
    assert t > 0.55 and s["f1"] == 1.0


def test_split_is_by_content_and_3d_photos_stay_out_of_test(tmp_path, monkeypatch):
    """Same photos under other names in another job: same teacher answers (shared cache), same test side;
    photos a 3D model was made from never land in test."""
    asked = fake_sam3(monkeypatch, lambda path: {"detections": [box(0, 0, 0.1, 0.1, 0.9)]})
    colors = [(20 * i, 5, 5) for i in range(10)]
    three_d = tmp_path / "three_d"
    three_d.mkdir()
    Image.new("RGB", (400, 200), colors[0]).save(three_d / "model.png")
    tests = []
    for job, names in (("a", [f"p{i}.png" for i in range(10)]), ("b", [f"other_{9 - i}.png" for i in range(10)])):
        raw = tmp_path / f"raw_{job}"
        raw.mkdir()
        for i, n in enumerate(names):
            Image.new("RGB", (400, 200), colors[i]).save(raw / n)
        spec = {**SPEC, "user_data": {"raw": "raw", "test_per_class": 3, "train_per_class": 10}}
        spec["classes"] = [dict(SPEC["classes"][0], assets={**SPEC["classes"][0]["assets"], "photos": "three_d"}),
                           SPEC["classes"][1]]
        ctx = Context(from_dict(spec), open_job(tmp_path, job), Cache(tmp_path / "cache"))
        link_input(ctx.job, "raw", raw)
        link_input(ctx.job, "three_d", three_d)
        Label().run(ctx)
        Ingest().run(ctx)
        table = json.loads((ctx.job.stage_dir("ingest") / "test" / "points.json").read_text())
        tests.append(sorted(Image.open(raw / v["key"]).getpixel((0, 0)) for v in table.values()))
    assert len(asked) == 10  # job b reused all ten answers
    assert tests[0] == tests[1] and len(tests[0]) == 3
    assert colors[0] not in tests[0]  # the 3D photo is in train


def test_big_objects_match_within_their_core():
    big = (0.5, 0.5, "pot", 0.6, 0.8)  # a pot filling most of the picture
    assert counting.match([(0.6, 0.62, "pot")], [big], radius=2 / 28) == (1, 0, 0)
    assert counting.match([(0.6, 0.62, "pot")], [big[:3]], radius=2 / 28) == (0, 1, 1)  # a bare point is strict


def test_objects_smaller_than_a_cell_are_reported():
    """Pots in street photos: median 6 px at 224 and F1 9% (2026-10-01). Say why before the user wonders."""
    from sim2chip import counting
    tiny = [[0.02, 0.03]] * 7 + [[0.3, 0.3]] * 3  # 5 px and 67 px at 224
    small = counting.small_objects(tiny, 224)
    assert small["share"] == 0.7 and small["median_px"] < 8 and "224 px" in small["note"]
    assert counting.small_objects(tiny, 1024) is None  # the same photos at a bigger input are fine
    assert counting.small_objects([], 224) is None


def test_crowded_photos_warn_the_user():
    """FSC-147: over 30 per photo the count was off by 15 to 70% (2026-10-01). Say what to expect."""
    from sim2chip import counting
    dense = counting.crowded([5, 12, 40, 80])
    assert dense["share"] == 0.5 and dense["max"] == 80 and "a third to a half" in dense["note"]
    assert counting.crowded([40] + [8] * 24) is None  # one crowded photo in 25: no warning
    assert counting.crowded([]) is None


def test_few_photos_keep_the_threshold_near_the_middle():
    """Cross-validation on 19 photos chose 0.9 by a hair; 0.5 scored twice as well on the test (2026-10-01)."""
    from sim2chip import counting
    assert counting.steady_threshold({0.3: 0.2, 0.5: 0.33, 0.7: 0.34, 0.9: 0.35}) == 0.5
    assert counting.steady_threshold({0.3: 0.1, 0.5: 0.2, 0.7: 0.3, 0.9: 0.35}) == 0.9  # a clear winner stays


def test_count_seeds_share_one_threshold():
    from sim2chip.stages.train import common_threshold

    class Ctx:
        log = staticmethod(lambda m: None)
    curve = lambda a, b: {"0.5": {"f1": a, "count_mae": 1.0}, "0.9": {"f1": b, "count_mae": 2.0}}
    summary = {0: {"cv_curve": curve(0.30, 0.40), "threshold": 0.9}, 1: {"cv_curve": curve(0.30, 0.20), "threshold": 0.5}}
    common_threshold(Ctx(), summary)
    assert {r["threshold"] for r in summary.values()} == {0.5} and summary[1]["val_f1"] == 0.30


def test_count_error_comes_from_the_density_when_given():
    """Points merge a crowd into fewer blobs; the density's rounded sum is the count that is scored."""
    import numpy as np
    from sim2chip import counting
    truth = [[(0.1, 0.1, "a"), (0.12, 0.1, "a"), (0.5, 0.5, "a")]]
    pred = [[(0.11, 0.1, "a"), (0.5, 0.5, "a")]]
    assert counting.score(pred, truth, 28)["count_mae"] == 1
    d = np.zeros((28, 28, 1))
    d[2, 2, 0], d[14, 14, 0] = 1.8, 1.1
    counts = [counting.density_counts(d, ["empty", "a"])]
    assert counts == [{"a": 3}]
    sc = counting.score(pred, truth, 28, counts)
    assert sc["count_mae"] == 0 and sc["exact"] == 1 and sc["f1"] == counting.score(pred, truth, 28)["f1"]
