"""The scene template off (stages.base.TEMPLATE, roadmap docs/roadmap_photo2scene.md A.2): synthetic images come
only from photo2scene, a job no route fits trains on the user's photos and says why, and nothing trains without
photos."""
import json

from sim2chip import interview
from sim2chip.spec import from_dict, no_photos
from sim2chip.stages import STAGES, Context
from sim2chip.stages.render import Render, no_synthetic
from sim2chip.workspace import Cache, open_job

COUNT = {"name": "bottles", "task": "count", "background_class": "empty",
         "classes": [{"id": "bottle", "names": ["bottle"], "size_m": {"measure": "height", "range": [0.2, 0.3]},
                      "assets": {"source": "search", "queries": ["bottle"]}}],
         "camera": {"pitch_deg": [60, 90]}, "user_data": {"raw": "photos"}}
TEMPLATE_STAGES = {"generate_assets", "fetch_assets", "convert_assets", "preview_assets", "qa_assets"}


def enabled(spec: dict) -> list[str]:
    return [s.name for s in STAGES if s.enabled(from_dict(spec))]


def test_photo2scene_runs_without_the_template_stages():
    names = enabled(COUNT)
    assert {"scene_from_photo", "render", "points"} <= set(names) and not TEMPLATE_STAGES & set(names)
    # The render only collects the scene frames and compares them with the photos (SigLIP 2): no Isaac Sim.
    assert Render().runtime == "cpu" and Render().gpu == "any"


def test_no_route_means_no_synthetic_stages_and_a_reason():
    # Classifying takes the support route too (roadmap B.4); level photos still have none.
    classify = {**COUNT, "task": "roi_classification"}
    assert {"scene_from_photo", "render", "crops"} <= set(enabled(classify)) and no_synthetic(from_dict(classify)) is None
    assert not {"scene_from_photo", "render", "crops"} & set(enabled({**classify, "camera": {}}))
    off = {**COUNT, "generation": {"photo_scenes": False}}
    assert "render" not in enabled(off) and "turned off" in no_synthetic(from_dict(off))
    level = {**COUNT, "camera": {}}
    assert "render" not in enabled(level) and "look down" in no_synthetic(from_dict(level))
    assert no_synthetic(from_dict({**COUNT, "generation": {"frames": 0}})) == "the setup asks for your photos only"
    assert no_synthetic(from_dict(COUNT)) is None


def test_render_gives_no_template_frames(tmp_path):
    """Scenes off and the user's photos: no frame at all, even with an earlier run's scene frames in the folder."""
    job = open_job(tmp_path, "j")
    src = job.stage_dir("scene_from_photo")
    (src / "s").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (32, 32)).save(src / "s" / "v0.png")
    (src / "frames.json").write_text(json.dumps([{"file": "s/v0.png", "scene": "s",
                                                  "boxes": [["bottle", 1, 1, 9, 9]]}]))
    off = Context(from_dict({**COUNT, "generation": {"photo_scenes": False}}), job, Cache(tmp_path / "cache"))
    Render().run(off)
    assert json.loads((job.stage_dir("render") / "labels.json").read_text()) == []
    # Scenes on: the scene's frames and nothing else.
    Render().run(Context(from_dict(COUNT), job, Cache(tmp_path / "cache")))
    labels = json.loads((job.stage_dir("render") / "labels.json").read_text())
    assert [lab["context"] for lab in labels] == ["photo"] and labels[0]["boxes"] == [["bottle", 1, 1, 9, 9]]
    assert sorted(p.name for p in job.stage_dir("render").glob("*.png")) == ["rgb_0000.png"]


def test_the_run_says_why_it_had_no_synthetic_images(tmp_path):
    spec = from_dict(COUNT)
    job = open_job(tmp_path, "j")
    job.mark_done("scene_from_photo", "fp", {"left_out": "photo2scene needs sam3d on this machine"})
    assert no_synthetic(spec, job) == "rebuilding your place failed (photo2scene needs sam3d on this machine)"
    job.clear_done("scene_from_photo")
    (job.stage_dir("scene_from_photo") / "scenes.json").parent.mkdir(parents=True, exist_ok=True)
    (job.stage_dir("scene_from_photo") / "scenes.json").write_text(json.dumps(
        [{"photo_id": 3, "route": "support", "status": "failed", "why": "no clear surface"}]))
    job.stage_dir("render").mkdir(parents=True)
    (job.stage_dir("render") / "labels.json").write_text("[]")
    assert no_synthetic(spec, job) == "no photo made a scene (no clear surface)"


def test_nothing_trains_without_photos():
    bare = {k: v for k, v in COUNT.items() if k != "user_data"}
    assert "photos" in no_photos(from_dict(bare)) and no_photos(from_dict(COUNT)) is None
    assert "photos action" in interview._no_photos_problem(None, from_dict(bare))


def test_frames_unlike_the_photos_stay_out(tmp_path, monkeypatch):
    """The threshold is the least alike pair of neighbours among the training photos; a frame closer to some photo
    than that goes in, one farther from all of them stays out, and the report and the reason say so."""
    from PIL import Image
    from sim2chip import likeness
    from sim2chip.stages import render

    job = open_job(tmp_path, "j")
    photos = [job.root / "label" / "images" / f"{i}.jpg" for i in range(3)]
    src = job.stage_dir("scene_from_photo")
    (src / "s").mkdir(parents=True)
    for i, p in enumerate(photos):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (i, 0, 0)).save(p)
    table = {f"{i}.png": {"points": [], "key": f"{i}.jpg", "source": f"label/images/{i}.jpg"} for i in range(3)}
    (job.stage_dir("ingest") / "train").mkdir(parents=True)
    (job.stage_dir("ingest") / "train" / "points.json").write_text(json.dumps(table))
    for k in range(2):
        Image.new("RGB", (32, 32), (0, k, 0)).save(src / "s" / f"v{k}.png")
    (src / "frames.json").write_text(json.dumps([{"file": f"s/v{k}.png", "scene": "s", "boxes": []} for k in range(2)]))
    # Photos at 0, 20 and 40 degrees: each one's nearest neighbour is 20 degrees away. Frame 0 sits at 10 degrees,
    # frame 1 at 90 (a sky over a lawn).
    import math
    unit = lambda deg: [math.cos(math.radians(deg)), math.sin(math.radians(deg))]  # noqa: E731
    angle = {p: 20 * i for i, p in enumerate(photos)}

    def fake(files, cache, log=print, timeline=None):
        return {f: unit(angle[f] if f in angle else 10 if f.name == "rgb_0000.png" else 90) for f in files}
    monkeypatch.setattr(likeness, "embeddings", fake)
    ctx = Context(from_dict(COUNT), job, Cache(tmp_path / "cache"))
    Render().run(ctx)
    labels = json.loads((job.stage_dir("render") / "labels.json").read_text())
    like = json.loads((job.stage_dir("render") / render.LIKENESS).read_text())
    assert [lab["frame"] for lab in labels] == [0] and like["kept"] == 1 and like["left_out"] == 1
    assert abs(like["threshold"] - math.cos(math.radians(20))) < 1e-3
    assert no_synthetic(ctx.spec, job) is None
    # Every frame unlike the photos: the reason says so.
    angle_all = lambda files, cache, log=print, timeline=None: {f: unit(angle.get(f, 90)) for f in files}  # noqa: E731
    monkeypatch.setattr(likeness, "embeddings", angle_all)
    Render().run(ctx)
    assert no_synthetic(ctx.spec, job) == "none of the 2 synthetic images looked like your photos"
