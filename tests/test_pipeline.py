"""Pipeline planning, fingerprints and skipping, with fake in-process stages (no GPU)."""
import pytest

from sim2chip.pipeline import Pipeline, StageFailed
from sim2chip.spec import from_dict
from sim2chip.stages import Context, Stage
from sim2chip.workspace import Cache, open_job

SPEC = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}}]}


class Fake(Stage):
    def __init__(self, name, uses=(), after=(), enabled=True):
        self.name, self.uses, self.after, self._enabled = name, uses, after, enabled

    def enabled(self, spec):
        return self._enabled

    def run(self, ctx):
        return {}


class InProcess:
    def __init__(self, fail=()):
        self.ran, self.fail = [], set(fail)

    def run_stage(self, stage, job, cache, fp):
        self.ran.append(stage.name)
        if stage.name not in self.fail:
            job.mark_done(stage.name, fp)


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
    assert [p.stage.name for p in pipe.plan()] == [s.name for s in STAGES if s.name not in ("preview_assets", "label")]
    auto = Pipeline(from_dict({**spec, "qa": {"mode": "auto"}}), open_job(tmp_path, "j3"), Cache(tmp_path / "cache"))
    assert "preview_assets" in [p.stage.name for p in auto.plan()]
    no_photos = Pipeline(from_dict(SPEC), open_job(tmp_path, "j2"), Cache(tmp_path / "cache"))
    names = [p.stage.name for p in no_photos.plan()]
    assert "ingest" not in names and "evaluate" not in names and "train" in names


def test_context_hardware(tmp_path):
    ctx = Context(from_dict(SPEC), open_job(tmp_path, "job"), Cache(tmp_path))
    assert ctx.hardware.input_size == 96
