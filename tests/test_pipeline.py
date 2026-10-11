"""Pipeline planning, fingerprints and skipping, with fake in-process stages (no GPU)."""
import json

import pytest

from sim2chip.pipeline import Pipeline, StageFailed
from sim2chip.spec import from_dict
from sim2chip.stages import Context, Stage
from sim2chip.workspace import Cache, open_job

SPEC = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}}]}


class Fake(Stage):
    def __init__(self, name, uses=(), after=(), enabled=True, skippable=None):
        self.name, self.uses, self.after, self._enabled = name, uses, after, enabled
        self.skippable = skippable

    def enabled(self, spec):
        return self._enabled

    def run(self, ctx):
        return {}


class InProcess:
    name = "local"

    def __init__(self, fail=()):
        self.ran, self.fail = [], set(fail)

    def run_stage(self, stage, job, cache, fp):
        self.ran.append(stage.name)
        if stage.name not in self.fail:
            job.mark_done(stage.name, fp, {"dropped": 29})  # a stage's own info may use any key


def make(tmp_path, spec=SPEC, stages=None):
    stages = stages or [Fake("a", uses=("generation",)), Fake("b", uses=("training",), after=("a",)),
                        Fake("c", after=("b",))]
    return Pipeline(from_dict(spec), open_job(tmp_path, "job"), Cache(tmp_path / "cache"), stages)


def test_runs_then_skips(tmp_path):
    be = InProcess()
    make(tmp_path).run(be, log=lambda m: None)
    assert be.ran == ["a", "b", "c"]
    be2 = InProcess()
    make(tmp_path).run(be2, log=lambda m: None)
    assert be2.ran == []


def test_spec_change_reruns_only_dependents(tmp_path):
    make(tmp_path).run(InProcess(), log=lambda m: None)
    spec = {**SPEC, "training": {"epochs": 3}}
    be = InProcess()
    make(tmp_path, spec).run(be, log=lambda m: None)
    assert be.ran == ["b", "c"]


def test_from_and_only(tmp_path):
    make(tmp_path).run(InProcess(), log=lambda m: None)
    be = InProcess()
    make(tmp_path).run(be, start="b", log=lambda m: None)
    assert be.ran == ["b", "c"]
    be = InProcess()
    make(tmp_path).run(be, only="b", log=lambda m: None)
    assert be.ran == ["b"]


def test_only_requires_fresh_upstream(tmp_path):
    with pytest.raises(ValueError, match="upstream"):
        make(tmp_path).plan(only="c")


def test_failure_stops(tmp_path):
    be = InProcess(fail={"b"})
    with pytest.raises(StageFailed):
        make(tmp_path).run(be, log=lambda m: None)
    assert be.ran == ["a", "b"]


def test_disabled_upstream_is_ignored(tmp_path):
    stages = [Fake("ingest", enabled=False), Fake("train", after=("ingest",))]
    be = InProcess()
    make(tmp_path, stages=stages).run(be, log=lambda m: None)
    assert be.ran == ["train"]


def test_unknown_dependency(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        make(tmp_path, stages=[Fake("x", after=("nope",))])


def test_real_stage_registry_plans(tmp_path):
    from sim2chip.stages import STAGES
    spec = {**SPEC, "user_data": {"labeled": "photos"}}
    pipe = Pipeline(from_dict(spec), open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    assert [p.stage.name for p in pipe.plan()] == [s.name for s in STAGES
                                                   if s.name not in ("preview_assets", "label", "generate_assets",
                                                                     "points", "scene_from_photo")]
    auto = Pipeline(from_dict({**spec, "qa": {"mode": "auto"}}), open_job(tmp_path, "j3"), Cache(tmp_path / "cache"))
    assert "preview_assets" in [p.stage.name for p in auto.plan()]
    no_photos = Pipeline(from_dict(SPEC), open_job(tmp_path, "j2"), Cache(tmp_path / "cache"))
    names = [p.stage.name for p in no_photos.plan()]
    assert "ingest" not in names and "evaluate" not in names and "train" in names
    count = Pipeline(from_dict({**SPEC, "task": "count", "user_data": {"raw": "photos"}}), open_job(tmp_path, "j4"),
                     Cache(tmp_path / "cache"))
    names = [p.stage.name for p in count.plan()]
    assert "points" in names and "crops" not in names and "label" in names
    assert "scene_from_photo" not in names  # the camera is not looking down at one surface
    down = from_dict({**SPEC, "task": "count", "user_data": {"raw": "photos"}})
    down.camera.pitch_deg = (60.0, 85.0)
    assert "scene_from_photo" in [p.stage.name for p in Pipeline(down, open_job(tmp_path, "j6"),
                                                                   Cache(tmp_path / "cache")).plan()]
    # The count task changes what the task-aware stages fingerprint; the default task changes nothing.
    assert from_dict(SPEC).section("task") is None
    assert count.fingerprints()["render"] != Pipeline(from_dict({**SPEC, "user_data": {"raw": "photos"}}),
                                                      open_job(tmp_path, "j5"), Cache(tmp_path / "cache")
                                                      ).fingerprints()["render"]


def test_context_hardware(tmp_path):
    ctx = Context(from_dict(SPEC), open_job(tmp_path, "job"), Cache(tmp_path))
    assert ctx.hardware.input_size == 96


def test_generated_assets_feed_fetch(tmp_path):
    from sim2chip.stages import STAGES
    spec = from_dict(SPEC)
    plain = Pipeline(spec, open_job(tmp_path, "a"), Cache(tmp_path / "cache")).fingerprints()
    photos = json.loads(json.dumps(SPEC))
    photos["classes"][0]["assets"]["photos"] = "cow_photos"
    fps = Pipeline(from_dict(photos), open_job(tmp_path, "b"), Cache(tmp_path / "cache")).fingerprints()
    names = [s.name for s in STAGES]
    assert "generate_assets" in fps and names.index("generate_assets") < names.index("fetch_assets")
    assert "generate_assets" not in plain and fps["fetch_assets"] != plain["fetch_assets"]


def test_powered_by_says_where_it_really_runs():
    from sim2chip.stages import BY_NAME
    spec = from_dict({**SPEC, "qa": {"mode": "auto"}})
    assert BY_NAME["render"].powered_by(spec, "local") == "NVIDIA Isaac Sim | local GPU"
    assert BY_NAME["render"].powered_by(spec, "nebius") == "NVIDIA Isaac Sim | Nebius AI Cloud, L40S"
    assert BY_NAME["qa_assets"].powered_by(spec, "local") == \
        "vision model on Nebius Token Factory, called from local CPU"
    assert BY_NAME["label"].powered_by(spec, "nebius") == "Google SigLIP 2 | Nebius AI Cloud, L40S"
    assert BY_NAME["label"].powered_by(from_dict({**SPEC, "task": "count"}), "local") == "Meta SAM 3 | local GPU"
    # Without auto QA the stage makes no Token Factory call, so it must not claim one.
    assert BY_NAME["qa_assets"].powered_by(from_dict(SPEC), "local") == "local CPU"


def real_shape():
    """The real graph's shape: user photos (label, ingest) and the synthetic branch (generate_assets, render)
    meet at train."""
    gen = Fake("generate_assets", after=("label",))
    gen.follows_upstream = False
    return [Fake("label"), gen, Fake("ingest", after=("label", "generate_assets")),
            Fake("fetch", after=("generate_assets",)), Fake("qa", after=("fetch",)), Fake("render", after=("qa",)),
            Fake("train", after=("render", "ingest")), Fake("export", after=("train", "qa"))]


def test_from_reruns_dependents_only(tmp_path):
    make(tmp_path, stages=real_shape()).run(InProcess(), log=lambda m: None)
    be = InProcess()
    make(tmp_path, stages=real_shape()).run(be, start="label", log=lambda m: None)
    assert be.ran == ["label", "ingest", "train", "export"]  # no 3D models, no render
    be = InProcess()
    make(tmp_path, stages=real_shape()).run(be, start="qa", log=lambda m: None)
    assert be.ran == ["qa", "render", "train", "export"]  # the user photos are untouched
    be = InProcess()
    make(tmp_path, stages=real_shape()).run(be, start="generate_assets", log=lambda m: None)
    assert be.ran == ["generate_assets", "ingest", "fetch", "qa", "render", "train", "export"]  # named: it runs
    be = InProcess()
    make(tmp_path, stages=real_shape()).run(be, force=True, log=lambda m: None)
    assert len(be.ran) == 8


def test_a_step_the_model_can_do_without_is_left_out(tmp_path):
    """The synthetic branch fails: with the user's photos the run goes on without it and what hangs from it,
    says why. A run from a later step keeps it out; a whole run ("Try again") tries it again, and what read its
    absence. Without photos it stops the run."""
    stages = lambda: [Fake("render", skippable="with photos"), Fake("crops", after=("render",), skippable="with photos"),
                      Fake("train", after=("crops",))]
    photos = {**SPEC, "user_data": {"raw": "photos"}}
    be, said = InProcess(fail={"render"}), []
    make(tmp_path, photos, stages()).run(be, log=said.append)
    job = open_job(tmp_path, "job")
    assert be.ran == ["render", "train"] and job.dropped("render")
    be = InProcess()  # a stage that dropped some of its items is not itself left out
    make(tmp_path / "ok", photos, stages()).run(be, log=lambda m: None)
    assert be.ran == ["render", "crops", "train"] and "render was left out" in job.dropped("crops")
    assert any(m.startswith("WARNING: render left out") for m in said)
    be = InProcess()
    make(tmp_path, photos, stages()).run(be, start="train", log=lambda m: None)
    assert be.ran == ["train"] and job.dropped("render")
    be = InProcess()
    make(tmp_path, photos, stages()).run(be, log=lambda m: None)
    assert be.ran == ["render", "crops", "train"] and not job.dropped("render") and not job.dropped("crops")
    with pytest.raises(StageFailed):
        make(tmp_path / "other", SPEC, stages()).run(InProcess(fail={"render"}), log=lambda m: None)
    # A stage goes on without an upstream stage it names in `extra` (render without the user's place).
    render = Fake("render", after=("scene",), skippable="with photos")
    render.extra = ("scene",)
    be = InProcess(fail={"scene"})
    make(tmp_path / "place", photos, [Fake("scene", skippable="with photos"), render]).run(be, log=lambda m: None)
    assert be.ran == ["scene", "render"] and not open_job(tmp_path / "place", "job").dropped("render")


def test_render_leaves_out_the_scene_frames_of_a_run_that_had_them(tmp_path):
    """Scenes turned off: the frames an earlier run left in scene_from_photo/ are not this spec's."""
    from sim2chip.stages.render import scene_frames
    spec = from_dict({**SPEC, "task": "count", "user_data": {"raw": "photos"},
                      "camera": {"pitch_deg": [60, 85]}})
    ctx = Context(spec, open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    src = ctx.job.stage_dir("scene_from_photo")
    (src / "s").mkdir(parents=True)
    (src / "s" / "v.png").write_bytes(b"png")
    (src / "frames.json").write_text(json.dumps([{"file": "s/v.png", "scene": "s", "boxes": [["cow", 0, 0, 9, 9]]}]))
    out = tmp_path / "render"
    out.mkdir()
    assert len(scene_frames(ctx, out, 0)) == 1
    spec.generation.photo_scenes = False
    assert scene_frames(ctx, out, 0) == []
