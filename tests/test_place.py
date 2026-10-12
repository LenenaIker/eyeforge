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
    d = tmp_path
    routed(d, "village_street_day")
    write(d, "extended/views.json", {"kept": [1, 3]})
    write(d, "populated/populate.json", {"placed_per_view": [2, 3, 0, 1]})
    flow = FakeFlow(d)
    # compose() clears the job's last things first: the step outputs appear after that, as the steps would write.
    real_run = flow.run

    def run(step, *a, **kw):
        if step == "3.5_populate":
            write(d, "populated/populate.json", {"placed_per_view": [2, 3, 0, 1]})
        if step == "3.5_labels":
            write(d, "populated/isaac/labels.json", {f"render_{k:02d}.png": [{"box": [0, 0, 1, 1]}] for k in range(4)})
            for k in range(4):
                (d / f"populated/isaac/render_{k:02d}.png").write_bytes(b"")
        return real_run(step, *a, **kw)
    flow.run = run
    frames = place.compose(flow, 7)
    assert [f["file"] for f in frames] == ["populated/isaac/render_01.png", "populated/isaac/render_03.png"]
    assert [s for s, *_ in flow.ran] == ["3.5_populate", "3.5_render", "3.5_labels"]


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
    monkeypatch.setattr(photo2scene, "build", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("support")))
    monkeypatch.setattr(place, "observe", observe)
    monkeypatch.setattr(place, "build", build)
    monkeypatch.setattr(place, "compose", lambda flow, seed: composed.append(seed) or [
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
