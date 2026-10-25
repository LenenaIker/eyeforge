"""The scene photo is the cleanest of the typical ones, as the vision model rates them (roadmap B.3); surfaces
make no scene, and classifying views show one class each (B.4)."""
import json
from pathlib import Path

from PIL import Image

from sim2chip import llm
from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.scene_from_photo import PICK_AMONG, SceneFromPhoto
from sim2chip.workspace import Cache, open_job

SPEC = {"name": "b", "task": "count", "classes": [{"id": "bottle", "names": ["bottle"]}],
        "camera": {"pitch_deg": [60, 90]}, "user_data": {"raw": "photos"}}


class Rater:
    """A vision model whose score is written in the photo's red channel."""
    calls = 0

    def __init__(self, usage_log=None):
        pass

    def model_for(self, role):
        return "m"

    def chat(self, msg, model, purpose, **kw):
        import base64
        import io
        Rater.calls += 1
        url = msg[0]["content"][1]["image_url"]["url"]
        red = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0))[0]
        # Blue marks a photo without the class.
        blue = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0))[2]
        return json.dumps({"score": red // 20, "why": f"red {red}", **({"count": 0} if blue > 128 else {})})


def test_the_cleanest_typical_photo_is_built_first(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "Client", Rater)
    job = open_job(tmp_path, "j")
    ctx = Context(from_dict(SPEC), job, Cache(tmp_path / "cache"))
    seen = []
    ctx.log = seen.append
    photos = []
    for i, red in enumerate([40, 200, 120, 0, 0, 255, 0, 220]):  # the last one is past PICK_AMONG
        p = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 48), (red, 0, 255 if i == 5 else 0)).save(p, quality=100)
        photos.append((i, p))
    picked, things = SceneFromPhoto._pick(ctx, photos, ["bottle"])
    assert [pid for pid, _ in picked[:3]] == [1, 2, 0] and picked[PICK_AMONG:] == photos[PICK_AMONG:]
    assert picked[PICK_AMONG - 1][0] == 5  # the reddest, but without the class: last
    assert things == [True] * PICK_AMONG
    assert any(line.startswith("picked photo 1 (1.jpg): 10/10, red") for line in seen)
    # Kept by photo and classes: asked again, the model is not called.
    before = Rater.calls
    assert SceneFromPhoto._pick(ctx, photos, ["bottle"]) == (picked, things) and Rater.calls == before


def test_no_answer_keeps_the_typical_order(tmp_path, monkeypatch):
    class Silent(Rater):
        def chat(self, *a, **kw):
            raise llm.LLMError("down")
    monkeypatch.setattr(llm, "Client", Silent)
    ctx = Context(from_dict(SPEC), open_job(tmp_path, "j"), Cache(tmp_path / "cache"))
    seen = []
    ctx.log = seen.append
    photos = []
    for i in range(3):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i}.jpg")
        photos.append((i, tmp_path / f"{i}.jpg"))
    assert SceneFromPhoto._pick(ctx, photos, ["bottle"]) == (photos, []) and "did not answer" in seen[-1]


def test_surfaces_make_no_scene_when_counting(tmp_path, monkeypatch):
    """The vision model says the classes are surfaces: counting builds no scene and the run says why."""
    from sim2chip import photo2scene
    from sim2chip.stages import label
    from sim2chip.stages.render import no_synthetic

    class Lawn(Rater):
        def chat(self, *a, **kw):
            return json.dumps({"score": 6, "why": "clear", "things": False})
    monkeypatch.setattr(llm, "Client", Lawn)
    monkeypatch.setattr(photo2scene, "missing_pythons", lambda: [])
    monkeypatch.setattr(photo2scene, "build", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("built")))
    monkeypatch.setattr(label, "phrases_of", lambda spec, cache, log=print: {"grass": ["grass"]})
    Image.new("RGB", (8, 8)).save(tmp_path / "1.jpg")
    monkeypatch.setattr(SceneFromPhoto, "_candidates", staticmethod(lambda ctx, tried: [("grass", [(1, tmp_path / "1.jpg")])]))
    spec = from_dict({**SPEC, "classes": [{"id": c, "size_m": {"measure": "length", "range": [1, 2]},
                                           "assets": {"source": "search", "queries": [c]}} for c in ("grass", "border")]})
    job = open_job(tmp_path, "j")
    SceneFromPhoto().run(Context(spec, job, Cache(tmp_path / "cache")))
    (job.stage_dir("render")).mkdir(parents=True)
    (job.stage_dir("render") / "labels.json").write_text("[]")
    assert "surfaces" in no_synthetic(spec, job)


def test_classifying_views_show_one_class(tmp_path, monkeypatch):
    """Each view's label is its one class, a view with none is the background class, and the scene's things are
    grouped by class for the build."""
    from sim2chip import photo2scene
    asked = {}

    def compose(flow, views, seed, groups=None, empty=0.0, pool=None):
        asked.update(groups=groups, empty=empty)
        box = lambda name: {"box": [1, 1, 9, 9], "name": name, "role": "target"}  # noqa: E731
        return [{"file": "isaac/render_01.png", "boxes": [box("sun cream bottle")]},
                {"file": "isaac/render_02.png", "boxes": [box("water bottle"), box("water bottle")]},
                {"file": "isaac/render_03.png", "boxes": []}]
    monkeypatch.setattr(photo2scene, "compose", compose)
    d = tmp_path / "scene"
    d.mkdir()
    (d / "assets.json").write_text(json.dumps({"assets": [{"name": "sun cream bottle"}, {"name": "water bottle"},
                                                          {"name": "mug"}]}))
    phrases = {"cream": ["sun cream bottle"], "water": ["water bottle"]}
    spec = {**SPEC, "task": "roi_classification", "classes": [{"id": "cream"}, {"id": "water"}]}
    ctx = Context(from_dict({**spec, "background_class": "empty"}), open_job(tmp_path, "j"), Cache(tmp_path / "c"))
    frames = SceneFromPhoto()._compose(ctx, photo2scene.Flow(d, tmp_path / "u.jsonl", print), "s", phrases)
    assert [f["class"] for f in frames] == ["cream", "water", "empty"]
    assert asked["groups"] == {"sun cream bottle": "cream", "water bottle": "water"} and asked["empty"] > 0
    # Without a background class, a view with no class is left out.
    ctx = Context(from_dict({**spec, "background_class": None}), open_job(tmp_path, "k"), Cache(tmp_path / "c"))
    frames = SceneFromPhoto()._compose(ctx, photo2scene.Flow(d, tmp_path / "u.jsonl", print), "s", phrases)
    assert [f["class"] for f in frames] == ["cream", "water"] and asked["empty"] == 0


def test_a_class_tries_its_next_photo_when_one_makes_no_scene():
    built = []
    picks = [[(1, "a"), (2, "b"), (3, "c")], [(4, "d"), (5, "e")]]
    tried = []
    for _, (pid, _) in SceneFromPhoto._tries(picks, built):
        tried.append(pid)
        built.append({"photo_id": pid, "status": "failed" if pid in (1, 2) else "ready"})
    assert tried == [1, 2, 4]  # the first class gives up after BUILD_TRIES photos; the second is ready at once


def test_more_scenes_or_every_photo_for_the_pool():
    """want=2 builds until two photos of a group made a scene (one more try than that at most); want=None builds
    from every photo, to pool all their things."""
    def run(picks, want, failing=()):
        built, tried = [], []
        for gi, (pid, _) in SceneFromPhoto._tries(picks, built, want):
            tried.append((gi, pid))
            built.append({"photo_id": pid, "status": "failed" if pid in failing else "ready"})
        return tried
    picks = [[(1, "a"), (2, "b"), (3, "c"), (4, "d")]]
    assert run(picks, 2) == [(0, 1), (0, 2)]
    assert run(picks, 2, failing=(1,)) == [(0, 1), (0, 2), (0, 3)]
    assert run(picks, 2, failing=(1, 2)) == [(0, 1), (0, 2), (0, 3)]  # gives up: BUILD_TRIES - 1 extra tries
    assert run(picks, None, failing=(2,)) == [(0, 1), (0, 2), (0, 3), (0, 4)]
    assert run(picks, 0) == []


def test_the_pool_takes_every_scene_thing_with_a_mesh_and_a_size(tmp_path):
    from sim2chip import photo2scene
    for name, rows in (("a", [{"index": 0, "name": "bottle", "role": "target", "height_m": 0.2, "width_m": 0.08,
                               "mesh": "assets/sam3d/000_canon.glb"},
                              {"index": 1, "name": "mug", "role": "distractor", "height_m": 0.1, "width_m": 0.1,
                               "mesh": "assets/sam3d/001_canon.glb"}]),
                       ("b", [{"index": 0, "name": "bottle", "role": "target", "height_m": None, "width_m": 0.08,
                               "mesh": "assets/sam3d/000_canon.glb"}])):
        (tmp_path / name / "assets/sam3d").mkdir(parents=True)
        (tmp_path / name / "assets.json").write_text(json.dumps({"assets": rows}))
        (tmp_path / name / "assets/sam3d/000_canon.glb").write_text("x")
    pool = photo2scene.pool_of([tmp_path / "a", tmp_path / "b", tmp_path / "missing"])
    # a's mug has no mesh file; b's bottle no height.
    assert [(p["name"], p["index"], Path(p["dir"]).name) for p in pool] == [("bottle", 0, "a")]


def test_classifying_surfaces_reframes_the_photos(tmp_path, monkeypatch):
    """The lawn: surfaces, classifying, so the training photos are re-framed and each kept view has its class (with
    the surface route on; it is off until it helps)."""
    from sim2chip import photo2scene
    from sim2chip.photo2scene import surface
    from sim2chip.stages import ingest, label, scene_from_photo
    monkeypatch.setattr(scene_from_photo, "SURFACE_ROUTE", True)

    class Lawn(Rater):
        def chat(self, *a, **kw):
            return json.dumps({"score": 6, "why": "clear", "things": False})
    monkeypatch.setattr(llm, "Client", Lawn)
    monkeypatch.setattr(photo2scene, "missing_pythons", lambda: [])
    monkeypatch.setattr(label, "phrases_of", lambda spec, cache, log=print: {"grass": ["grass"], "edge": ["edge"]})
    photos = []
    for i, cls in enumerate(["grass", "edge"]):
        (tmp_path / cls).mkdir()
        Image.new("RGB", (8, 8)).save(tmp_path / cls / f"{i}.jpg")
        photos.append(tmp_path / cls / f"{i}.jpg")
    monkeypatch.setattr(SceneFromPhoto, "_candidates", staticmethod(lambda ctx, tried: [("grass", [(1, photos[0])])]))
    monkeypatch.setattr(ingest, "train_photos", lambda ctx: photos)

    def build(flow, given, frames, pitch, height, seed, classify, min_conf, log=print):
        assert [c for _, c in given] == ["grass", "edge"] and pitch == (60.0, 90.0) and min_conf == 0.6
        (flow.d / "views").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(flow.d / "views" / "0_v00.png")
        return [{"file": "views/0_v00.png", "source": "0", "class": "grass"}]
    monkeypatch.setattr(surface, "build", build)
    spec = from_dict({**SPEC, "task": "roi_classification", "background_class": None,
                      "classes": [{"id": c, "size_m": {"measure": "length", "range": [1, 2]},
                                   "assets": {"source": "search", "queries": [c]}} for c in ("grass", "edge")]})
    job = open_job(tmp_path, "j")
    job.stage_dir("label").mkdir(parents=True)
    (job.stage_dir("label") / "teacher.json").write_text(json.dumps({"centres": {"grass": [1, 0], "edge": [0, 1]}}))
    SceneFromPhoto().run(Context(spec, job, Cache(tmp_path / "cache")))
    frames = json.loads((job.stage_dir("scene_from_photo") / "frames.json").read_text())
    assert frames == [{"file": "surface/views/0_v00.png", "scene": "surface", "boxes": [], "class": "grass"}]
    assert (job.stage_dir("scene_from_photo") / frames[0]["file"]).exists()


class FakeFlow:
    """Runs no step: the test writes what the steps would."""
    def __init__(self, d):
        self.d = d

    def run(self, *a, **kw):
        return 0


def test_a_view_keeps_its_photo_class_when_the_teacher_agrees(tmp_path, monkeypatch):
    """Re-framed views keep their photo's class only when the teacher, with what it learnt from the user, gives them
    that class sure enough; joined grounds take the teacher's class (a tenth of the views)."""
    from sim2chip.photo2scene import surface
    monkeypatch.setattr(surface, "MIX_SHARE", 0.5)
    (tmp_path / "views").mkdir()
    (tmp_path / "views" / "views.json").write_text(json.dumps({"photos": {}, "views": [
        {"file": "g0_v00.png", "source": "g0"}, {"file": "e0_v00.png", "source": "e0"},
        {"file": "e1_v00.png", "source": "e1"}]}))
    (tmp_path / "mix").mkdir()
    (tmp_path / "mix" / "mix.json").write_text(json.dumps([
        {"file": "mix_000.png", "a": "g0", "b": "e0"}, {"file": "mix_001.png", "a": "g1", "b": "e1"}]))
    said = {"g0_v00.png": ("grass", 0.9), "e0_v00.png": ("grass", 0.9),  # the edge framed out of sight
            "e1_v00.png": ("edge", 0.5),                                  # not sure enough
            "mix_000.png": ("edge", 0.8), "mix_001.png": ("edge", 0.3)}
    photos = [(tmp_path / "g0.jpg", "grass"), (tmp_path / "e0.jpg", "edge"), (tmp_path / "e1.jpg", "edge")]
    views = surface.build(FakeFlow(tmp_path), photos, 12, None, None, 0, lambda files: [said[f.name] for f in files],
                          0.6)
    assert [(v["file"], v["class"]) for v in views] == [("views/g0_v00.png", "grass"), ("mix/mix_000.png", "edge")]
