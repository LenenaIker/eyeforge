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


def test_the_spec_pitch_is_kept_within_what_the_walks_hold(tmp_path):
    s = spec()
    s.camera.height_m = (1.5, 1.7)
    s.camera.pitch_deg = (5.0, 60.0)  # degrees down; past 20 the splat shows what no walk filmed
    p = plan({"s": bundle(tmp_path)}, ASSETS, s, tmp_path, 10, (640, 480))
    for v in p["views"]:
        fwd = np.asarray(v["w2c"])[2, :3]
        assert -20 - 1e-3 <= math.degrees(math.asin(fwd[2])) <= -5 + 1e-3
    assert "pitch 5-60 deg down kept to 5-20" in p["camera_note"]
    s.camera.pitch_deg = None
    assert plan({"s": bundle(tmp_path)}, ASSETS, s, tmp_path, 4, (640, 480))["camera_note"] is None


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


def test_objects_are_whole_in_the_frame_top_to_bottom(tmp_path):
    from sim2chip.walkscene import intrinsic
    p = plan({"s": bundle(tmp_path)}, ASSETS, spec("count"), tmp_path, 30, (640, 480))
    K = intrinsic(640, 480, 60)
    heights = {"c1": 150 * 1.8 / 180, "b1": 1.05}
    for v in p["views"]:
        m = np.asarray(v["w2c"])
        for o in v["objects"]:
            for z in (-1.6, -1.6 + heights[o["key"]] * 0.95):  # base centre and (about) top
                c = m[:3, :3] @ np.array([*o["xy"], z]) + m[:3, 3]
                assert 0 <= (K @ c)[1] / c[2] <= 480


def test_scene_proposals_prefer_open_level_daylight_places(tmp_path, monkeypatch):
    from sim2chip import scene_job, scene_pick
    monkeypatch.setattr(scene_job, "no_focal", lambda f: "no focal length" if f.name == "nofocal.jpg" else None)
    open_road = {"context": "ground", "light": "overcast", "camera_pitch_deg": 5, "camera_height_m": 1.6,
                 "objects": [{"name": "bench", "count": 1, "size_m": 1.2}]}
    parked = {**open_road, "objects": [{"name": "car", "count": 3, "size_m": 4.5}]}
    photos = [{"id": 1, "file": tmp_path / "road.jpg", "description": open_road},
              {"id": 2, "file": tmp_path / "parked.jpg", "description": parked},
              {"id": 3, "file": tmp_path / "room.jpg", "description": {**open_road, "context": "indoor"}},
              {"id": 4, "file": tmp_path / "down.jpg", "description": {**open_road, "camera_pitch_deg": 80}},
              {"id": 5, "file": tmp_path / "night.jpg", "description": {**open_road, "light": "night"}},
              {"id": 6, "file": tmp_path / "nofocal.jpg", "description": open_road}]
    got = scene_pick.propose(photos, tmp_path)
    assert [p["photo_id"] for p in got] == [1, 2] and got[0]["score"] > got[1]["score"]


def test_the_chat_sees_ready_scenes_and_only_those_ids_pass(tmp_path):
    from sim2chip import interview
    from sim2chip.library import Library
    lib = Library(tmp_path, tmp_path / "cache")
    b = tmp_path / "f3c_x" / "bundle"
    b.mkdir(parents=True)
    (b / "scene.json").write_text(json.dumps({"route": "walk", "free_m2": 170.4}))
    sid = lib.import_scene("admin", b, "Road")
    lib.set_public(sid, True)
    text, ids = lib.for_chat("someone")
    assert f'scene "Road" (id {sid}, public, 170 m2 to walk)' in text and sid in ids

    class S:
        library_ids = ids
    spec = from_dict({"name": "t", "classes": [{"id": "bin", "size_m": {"measure": "height", "range": [1, 1.2]},
                                                "assets": {"source": "search", "queries": ["wheelie bin"]}}],
                      "generation": {"scenes": [sid]}})
    assert interview._walk_scenes_problem(S, spec) is None
    spec.generation.scenes = ["0123456789abcdef"]
    assert "not in the list" in interview._walk_scenes_problem(S, spec)
