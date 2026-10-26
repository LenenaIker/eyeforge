"""Frames in walk scenes (sim2chip/walkscene, stages/scene_render.py): the plan puts cameras and objects on free
ground, objects in front of the camera and apart, one class per classifying frame; hidden objects lose their box."""
import json
import math

import numpy as np

from sim2chip.spec import from_dict
from sim2chip.stages.scene_render import frames_of
from sim2chip.walkscene import Scene, w2c
from sim2chip.walkscene.plan import plan

CELL, SPAN = 0.25, 12.0


def bundle(tmp_path, wall_x=None):
    """A scene folder: everything free but, when given, a wall across x = wall_x."""
    n = int(2 * SPAN / CELL)
    free = np.ones((n, n), bool)
    occ = np.zeros((n, n), bool)
    if wall_x is not None:
        i = int((wall_x + SPAN) / CELL)
        occ[i, :] = True
        free[i - 2:, :] = False
    np.savez_compressed(tmp_path / "free.npz", free=free, occupied=occ)
    (tmp_path / "scene.json").write_text(json.dumps({
        "camera_height": 1.6, "centre": [0.0, 0.0], "radius": 10.0,
        "grid": {"file": "free.npz", "cell_m": CELL, "span_m": SPAN}}))
    return Scene(tmp_path)


def spec(task="roi_classification"):
    return from_dict({"name": "t", "task": task, "classes": [
        {"id": "car", "size_m": {"measure": "width", "range": [1.7, 1.9]}},
        {"id": "bin", "size_m": {"measure": "height", "range": [1.0, 1.1]}}],
        "camera": {"hfov_deg": 60, "resolution": [320, 240]}, "generation": {"empty_ratio": 0.2}})


ASSETS = [{"uid": "c1", "class": "car", "usd": "usd/norm/c1.usd", "dims": [450, 180, 150]},
          {"uid": "b1", "class": "bin", "usd": "usd/norm/b1.usd", "dims": [60, 60, 100]}]


def test_objects_stand_on_free_ground_in_front_of_the_camera(tmp_path):
    sc = bundle(tmp_path, wall_x=6.0)
    p = plan({"s": sc}, ASSETS, spec(), tmp_path, 40, (640, 480))
    assert len(p["views"]) == 40 - p["missed"] and p["views"]
    for v in p["views"]:
        m = np.asarray(v["w2c"])
        for o in v["objects"]:
            assert sc.is_free([o["xy"]])[0] and o["xy"][0] < 6.0 - 0.5
            cam = m[:3, :3] @ np.array([*o["xy"], -1.0]) + m[:3, 3]
            assert cam[2] > 0 and abs(math.degrees(math.atan2(cam[0], cam[2]))) < 30  # in front, within the view
        if v["class"] in ("car", "bin"):
            assert [o["class"] for o in v["objects"]] == [v["class"]]
        else:
            assert v["class"] == "empty" and not v["objects"]
    assert {v["class"] for v in p["views"]} == {"car", "bin", "empty"}


def test_counting_objects_keep_apart(tmp_path):
    p = plan({"s": bundle(tmp_path)}, ASSETS, spec("count"), tmp_path, 30, (640, 480))
    for v in p["views"]:
        assert v["class"] is None
        xy = [np.asarray(o["xy"]) for o in v["objects"]]
        assert all(np.hypot(*(a - b)) > 1.0 for i, a in enumerate(xy) for b in xy[i + 1:])


def test_eye_height_stays_near_the_walks(tmp_path):
    s = spec()
    s.camera.height_m = (4.0, 6.0)  # a pole camera: the splat holds up near 1.6 m only
    p = plan({"s": bundle(tmp_path)}, ASSETS, s, tmp_path, 10, (640, 480))
    for v in p["views"]:
        z = np.linalg.inv(np.asarray(v["w2c"]))[2, 3]
        assert -0.3 - 1e-4 <= z <= 0.3 + 1e-4


def test_no_models_no_views(tmp_path):
    p = plan({"s": bundle(tmp_path)}, [], spec(), tmp_path, 10, (640, 480))
    assert p["views"] == [] and p["why"]


def test_hidden_objects_lose_their_box_and_a_hidden_class_its_frame():
    views = [{"name": "0000", "scene": "s", "class": "car", "objects": [{"class": "car"}]},
             {"name": "0001", "scene": "s", "class": "car", "objects": [{"class": "car"}]},
             {"name": "0002", "scene": "s", "class": None, "objects": [{"class": "car"}, {"class": "bin"}]},
             {"name": "0003", "scene": "s", "class": "empty", "objects": []}]
    report = [{"name": "0000", "objects": [{"box": [1, 2, 3, 4]}]}, {"name": "0001", "objects": [None]},
              {"name": "0002", "objects": [None, {"box": [5, 6, 7, 8]}]}, {"name": "0003", "objects": []}]
    frames = frames_of(views, report, None)
    assert [f["file"] for f in frames] == ["render_0000.png", "render_0002.png", "render_0003.png"]
    assert frames[0] == {"file": "render_0000.png", "scene": "s", "boxes": [["car", 1, 2, 3, 4]], "class": "car"}
    assert frames[1]["boxes"] == [["bin", 5, 6, 7, 8]] and "class" not in frames[1]
    assert frames[2]["class"] == "empty" and frames[2]["boxes"] == []


def test_w2c_looks_along_x():
    m = w2c([0, 0, 0], 0, 0)
    assert np.allclose(m[:3, :3] @ [1, 0, 0], [0, 0, 1])  # +x is straight ahead
