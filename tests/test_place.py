"""photo2scene's street and room routes in the product (sim2chip/photo2scene/place.py, roadmap C.1): no step is
run, the tests write what the steps would."""
import json
from pathlib import Path

import pytest
from PIL import Image

from sim2chip.photo2scene import STEPS, Stop, place


class FakeFlow:
    """Runs no step; records which ones were asked for."""
    def __init__(self, d, codes=None):
        self.d, self.codes, self.ran, self.said = d, codes or {}, [], []

    def run(self, step, python, script, args, outputs=(), ok=(0,)):
        self.ran.append((step, python, script))
        return self.codes.get(step, 0)

    def json(self, name):
        return json.loads((self.d / name).read_text())

    def log(self, m):
        self.said.append(m)


def write(d: Path, name: str, value):
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(value))


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "in.jpg"
    Image.new("RGB", (64, 48)).save(p)
    return p


def routed(d, route, template="crossroads", built=True, why=()):
    write(d, "observe.json", {"final": {"place": "village_street", "objects": ["car"], "ground": ["road"],
                                        "surroundings": ["house"], "weather": "cloudy"}})
    write(d, "route.json", {"route": route, "template": template, "template_built": built,
                            "hdri_group": "countryside", "why": list(why)})


def test_a_photo_out_of_scope_makes_no_place(tmp_path, photo):
    d = tmp_path / "s"
    d.mkdir()
    routed(d, "out_of_scope", "none", False, ["night"])
    with pytest.raises(Stop, match="out of scope: night"):
        place.build(FakeFlow(d, {"1.2_route": 3}), photo, [])


@pytest.mark.parametrize("route,template,built,why", [
    ("support", "none", False, "working surface"),
    ("village_street_day", "square", False, "no template builds a square"),
])
def test_routes_this_module_does_not_build_stop_before_any_model(tmp_path, photo, route, template, built, why):
    d = tmp_path / "s"
    d.mkdir()
    routed(d, route, template, built)
    flow = FakeFlow(d)
    with pytest.raises(Stop, match=why):
        place.build(flow, photo, [])
    assert [s for s, *_ in flow.ran] == ["1.1_observe", "1.2_route"]
    assert (d / "photo.png").exists()


def test_compose_keeps_the_views_the_check_passed_without_the_photo_camera(tmp_path):
    """populate keeps the photo's camera and the views the check passed (extended/views.json), numbered again:
    kept [2, 5, 6] render as render_00 (the photo) to render_03. The photo 21 of the roads job lost 4 of its 11
    kept views to the old numbering (2026-10-12)."""
    d = tmp_path
    routed(d, "village_street_day")
    write(d, "extended/views.json", {"kept": [2, 5, 6]})
    flow = FakeFlow(d)
    # compose() clears the job's last things first: the step outputs appear after that, as the steps would write.
    real_run = flow.run
    asked = {}

    def run(step, python, script, args, *a, **kw):
        if step == "3.5_populate":
            asked["args"] = [str(x) for x in args]
            write(d, "populated/populate.json", {"placed_per_view": [2, 3, 0, 1]})
        if step == "3.5_labels":
            write(d, "populated/isaac/labels.json", {f"render_{k:02d}.png": [{"box": [0, 0, 1, 1]}] for k in range(4)})
            for k in range(4):
                (d / f"populated/isaac/render_{k:02d}.png").write_bytes(b"")
        return real_run(step, python, script, args, *a, **kw)
    flow.run = run
    frames = place.compose(flow, 7)
    assert [f["file"] for f in frames] == [f"populated/isaac/render_{k:02d}.png" for k in (1, 2, 3)]
    assert [s for s, *_ in flow.ran] == ["3.5_populate", "3.5_render", "3.5_labels"]
    assert "--groups" not in asked["args"]
    # Classifying: the groups go to populate, one class per view.
    place.compose(flow, 7, {"red car": "car", "bus": "bus"}, 0.1)
    assert asked["args"][-4:] == ["--groups", str(d / "populated/groups.json"), "--empty", "0.1"]
    assert json.loads((d / "populated/groups.json").read_text()) == {"red car": "car", "bus": "bus"}


def test_the_steps_need_nothing_from_the_experiments():
    """The product runs its steps from its own package (roadmap C.1)."""
    for f in STEPS.glob("*.py"):
        code = [line for line in f.read_text().splitlines() if line.lstrip().startswith(("import", "from", "sys.path"))]
        assert not [line for line in code if "experiments" in line], f.name


def test_the_stage_routes_each_photo(tmp_path, monkeypatch):
    """Counting: a night photo is out of scope and noted with why, the next photo is a street and gets the place
    route, its views counted with the user's classes (roadmap C.3)."""
    from sim2chip import llm, photo2scene
    from sim2chip.spec import from_dict
    from sim2chip.stages import Context, label
    from sim2chip.stages import scene_from_photo as sfp
    from sim2chip.stages.scene_from_photo import SceneFromPhoto
    from sim2chip.workspace import Cache, open_job

    class Rater:
        def __init__(self, usage_log=None):
            pass

        def model_for(self, role):
            return "m"

        def chat(self, *a, **kw):
            return json.dumps({"count": 2, "score": 7, "why": "clear", "things": True})
    photos = []
    for i in (1, 2):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(tmp_path / f"{i}.jpg")
        photos.append((i, tmp_path / f"{i}.jpg"))
    routes = {photos[0][1]: None, photos[1][1]: "village_street_day"}

    def observe(flow, photo):
        if routes[photo] is None:
            raise Stop("the photo is out of scope: night")
        return {"route": routes[photo]}

    def build(flow, photo, targets):
        (flow.d / "extended").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(flow.d / "photo.png")
        import numpy as np
        np.savez(flow.d / "assets.npz", masks=np.zeros((0, 8, 8), bool))
        return {"route": "village_street_day", "template": "crossroads", "hdri_group": "countryside", "objects": []}
    composed = []
    monkeypatch.setattr(llm, "Client", Rater)
    monkeypatch.setattr(photo2scene, "missing_pythons", lambda: [])
    monkeypatch.setattr(sfp, "STREET_ROUTE", True)
    monkeypatch.setattr(photo2scene, "build", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("support")))
    monkeypatch.setattr(place, "observe", observe)
    monkeypatch.setattr(place, "build", build)
    monkeypatch.setattr(place, "compose", lambda flow, seed, groups, empty: composed.append(seed) or [
        {"file": "populated/isaac/render_01.png", "boxes": [{"box": [1, 1, 5, 5], "name": "car", "role": "target"},
                                                            {"box": [2, 2, 3, 3], "name": "bench", "role": "distractor"}]}])
    monkeypatch.setattr(label, "phrases_of", lambda spec, cache, log=print: {"car": ["car"]})
    monkeypatch.setattr(SceneFromPhoto, "_candidates", staticmethod(lambda ctx, tried: [(None, photos)]))
    spec = from_dict({"name": "r", "task": "count", "user_data": {"raw": "photos"},
                      "classes": [{"id": "car", "names": ["car"], "size_m": {"measure": "height", "range": [1.4, 1.9]},
                                   "assets": {"source": "search", "queries": ["car"]}}]})
    job = open_job(tmp_path, "j")
    SceneFromPhoto().run(Context(spec, job, Cache(tmp_path / "cache")))
    out = job.stage_dir("scene_from_photo")
    scenes = json.loads((out / "scenes.json").read_text())
    assert [(s["photo_id"], s["route"], s["status"]) for s in scenes] == [(1, "none", "failed"), (2, "street", "ready")]
    assert "night" in scenes[0]["why"] and scenes[1]["info"]["template"] == "crossroads"
    frames = json.loads((out / "frames.json").read_text())
    assert frames == [{"file": "photo_2/populated/isaac/render_01.png", "scene": "photo_2", "boxes": [["car", 1, 1, 5, 5]]}]
    # The street route off (the default since C.6): the street photo makes no scene and says why.
    monkeypatch.setattr(sfp, "STREET_ROUTE", False)
    job2 = open_job(tmp_path, "j2")
    SceneFromPhoto().run(Context(spec, job2, Cache(tmp_path / "cache")))
    scenes = json.loads((job2.stage_dir("scene_from_photo") / "scenes.json").read_text())
    assert [(s["route"], s["status"], s["why"]) for s in scenes] == [
        ("none", "failed", "the photo is out of scope: night"), ("street", "failed", sfp.NO_STREETS)]


def test_the_street_bench_routes_by_rule():
    """The 9 street bench photos as Qwen3.8-27B observed them (2026-10-12): the 7 by day take the street route with a
    template that is built, the 2 at night are out of scope (roadmap C.3)."""
    import sys
    sys.path.insert(0, str(STEPS))
    from flow_route import route
    seen = json.loads((Path(__file__).parent / "fixtures" / "street_bench_observe.json").read_text())
    got = {n: route(o) for n, o in seen.items()}
    assert {n for n, r in got.items() if r["route"] == "out_of_scope"} == {"night_market", "night_street"}
    day = {n: r for n, r in got.items() if r["route"] != "out_of_scope"}
    assert len(day) == 7 and all(r["route"] == "village_street_day" and r["template_built"] for r in day.values())


def test_classifying_in_a_street_shows_one_class_per_view(tmp_path, monkeypatch):
    """Classifying, each class gets its street scene from a photo of it and each view shows one class, or none for
    the background class; a view with two classes is left out (roadmap C.5, as B.4 on the support route)."""
    from sim2chip import llm, photo2scene
    from sim2chip.spec import from_dict
    from sim2chip.stages import Context, label
    from sim2chip.stages import scene_from_photo as sfp
    from sim2chip.stages.scene_from_photo import SceneFromPhoto
    from sim2chip.workspace import Cache, open_job

    class Rater:
        def __init__(self, usage_log=None):
            pass

        def model_for(self, role):
            return "m"

        def chat(self, *a, **kw):
            return json.dumps({"count": 1, "score": 7, "why": "clear", "things": True})
    photos = {}
    for i, cls in ((1, "car"), (2, "bus")):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(tmp_path / f"{i}.jpg")
        photos[cls] = (i, tmp_path / f"{i}.jpg")

    def build(flow, photo, targets):
        (flow.d / "extended").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(flow.d / "photo.png")
        import numpy as np
        np.savez(flow.d / "assets.npz", masks=np.zeros((0, 8, 8), bool))
        thing = "red car" if photo == photos["car"][1] else "bus"
        write(flow.d, "assets.json", {"assets": [{"name": thing}, {"name": "bench"}]})
        return {"route": "village_street_day", "template": "crossroads", "hdri_group": "countryside", "objects": []}
    asked = []

    def compose(flow, seed, groups=None, empty=0.0):
        asked.append((groups, empty))
        (thing,) = set(groups)
        box = lambda name, role="target": {"box": [1, 1, 5, 5], "name": name, "role": role}  # noqa: E731
        return [{"file": "populated/isaac/render_01.png", "boxes": [box(thing), box("bench", "distractor")]},
                {"file": "populated/isaac/render_02.png", "boxes": [box("bench", "distractor")]},
                {"file": "populated/isaac/render_03.png", "boxes": [box(thing), box("car" if thing == "bus" else "bus")]}]
    monkeypatch.setattr(llm, "Client", Rater)
    monkeypatch.setattr(photo2scene, "missing_pythons", lambda: [])
    monkeypatch.setattr(sfp, "STREET_ROUTE", True)
    monkeypatch.setattr(photo2scene, "build", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("support")))
    monkeypatch.setattr(place, "observe", lambda flow, photo: {"route": "village_street_day"})
    monkeypatch.setattr(place, "build", build)
    monkeypatch.setattr(place, "compose", compose)
    monkeypatch.setattr(label, "phrases_of", lambda spec, cache, log=print: {"car": ["car"], "bus": ["bus"]})
    monkeypatch.setattr(SceneFromPhoto, "_candidates",
                        staticmethod(lambda ctx, tried: [(c, [photos[c]]) for c in ("bus", "car")]))
    size = {"measure": "height", "range": [1.4, 4.0]}
    spec = from_dict({"name": "r", "task": "roi_classification", "background_class": "empty",
                      "user_data": {"raw": "photos"},
                      "classes": [{"id": c, "names": [c], "size_m": size, "assets": {"source": "search", "queries": [c]}}
                                  for c in ("car", "bus")]})
    job = open_job(tmp_path, "j")
    SceneFromPhoto().run(Context(spec, job, Cache(tmp_path / "cache")))
    out = job.stage_dir("scene_from_photo")
    scenes = json.loads((out / "scenes.json").read_text())
    assert [(s["photo_id"], s["route"], s["status"]) for s in scenes] == [(2, "street", "ready"), (1, "street", "ready")]
    assert asked == [({"bus": "bus"}, spec.generation.empty_ratio), ({"red car": "car"}, spec.generation.empty_ratio)]
    frames = json.loads((out / "frames.json").read_text())
    assert [(f["file"], f["class"]) for f in frames] == [
        ("photo_2/populated/isaac/render_01.png", "bus"), ("photo_2/populated/isaac/render_02.png", "empty"),
        ("photo_1/populated/isaac/render_01.png", "car"), ("photo_1/populated/isaac/render_02.png", "empty")]


def test_trees_and_shrubs_go_to_nvidia_assets_hedges_stay_with_sam3d():
    import sys
    sys.path.insert(0, str(STEPS))
    from flow_vegetation import group_of
    assert group_of("tree") == group_of("tall pine trees") == "tree"
    assert group_of("bush") == group_of("shrubs") == "shrub"
    assert group_of("hedge") is None and group_of("house") is None and group_of("low stone wall") is None


def test_the_corner_template_plants_the_photo_bands_off_its_roads():
    """crossroad.py plants the photo's street bands again from its own curbs (section_fill.curb_section): only the
    plant bands on the block's side of the photo's road, starting past the template's curb, never on its road.
    Copied with the block, the trees across the photo's road stood in the template's roads (terrace, 2026-10-14)."""
    import sys
    import numpy as np
    sys.path.insert(0, str(STEPS))
    import section_fill
    section = {"road": {"centre": [5, 0, 5], "along": [0, 0, 1], "across": [1, 0, 0], "half": [2.5, 2.5]},
               "bands": [{"kind": "sidewalk", "side": 1, "from_m": 0, "to_m": 2},
                         {"kind": "building", "side": 1, "from_m": 2, "to_m": 9},
                         {"kind": "tree", "side": 1, "from_m": 20, "to_m": 28},
                         {"kind": "tree", "side": -1, "from_m": 0.1, "to_m": 0.4}]}
    up, a, b, centre = np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.zeros(3)
    sec = section_fill.curb_section(section, 1, centre, a, b, 3.5)
    assert [(x["kind"], x["side"], x["from_m"]) for x in sec["bands"]] == [("tree", 1, 20)]
    edge, at = section_fill.frame(sec, up, lambda X: np.zeros(len(np.atleast_2d(X))))
    lo, hi = section_fill.band_span(sec["bands"][0], edge)
    for u in (lo, hi):
        p = at(u, 4.5)
        assert p @ b >= 3.5 + 20 and p @ a > 3.5  # in the block's quadrant, past both roads
    assert section_fill.curb_section({**section, "bands": section["bands"][:2]}, 1, centre, a, b, 3.5) is None
