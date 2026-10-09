"""Versions of a job's model: the snapshot before a rerun, and the old model scored on the new test photos."""
import json

import numpy as np
import pytest
from PIL import Image

from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.workspace import Cache, current_version, open_job, snapshot, versions

SPEC = {"name": "t", "classes": [{"id": "cow", "size_m": {"measure": "length", "range": [2, 2.5]},
                                  "assets": {"categories": ["cow"]}}],
        "user_data": {"labeled": "photos"}}


def ctx_for(tmp_path, spec=SPEC):
    return Context(from_dict(spec), open_job(tmp_path, "job"), Cache(tmp_path / "cache"))


def fake_export(job, score, labels=("empty", "cow"), size=96):
    out = job.stage_dir("export")
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_int8.tflite").write_bytes(f"model {score}".encode())
    (out / "labels.txt").write_text("\n".join(labels))
    (out / "report.json").write_text(json.dumps({
        "labels": list(labels), "task": "roi_classification", "input": [size, size, 3], "metric": "accuracy",
        "seed": 0, "real_test": {"mean": score, "per_seed": {"0": score}, "test_size": 4}}))


def test_snapshot_keeps_the_model_once(tmp_path):
    job = ctx_for(tmp_path).job
    assert snapshot(job, "nothing yet") is None and not (job.input / "version.json").exists()
    fake_export(job, 0.5)
    kept = snapshot(job, "added 6 photos")
    assert kept["n"] == 1 and kept["reason"] == "first model" and kept["headline"] == 0.5
    assert (job.root / "versions" / "v1" / "model_int8.tflite").read_bytes() == b"model 0.5"
    assert (job.input / "previous" / "model_int8.tflite").read_bytes() == b"model 0.5"
    assert current_version(job) | {"started": 0} == {"n": 2, "reason": "added 6 photos", "started": 0}
    # The rerun failed: export/ still holds v1. Trying again does not keep it twice.
    snapshot(job, "added 6 photos, again")
    assert len(versions(job)) == 1 and current_version(job)["n"] == 2
    assert current_version(job)["reason"] == "added 6 photos, again"
    fake_export(job, 0.75)  # v2 finished
    assert snapshot(job, "corrected labels")["n"] == 2 and current_version(job)["n"] == 3
    assert [m["headline"] for m in versions(job)] == [0.5, 0.75]


def test_evaluate_scores_the_previous_model_on_the_same_photos(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    from sim2chip import student
    from sim2chip.stages.evaluate import Evaluate
    from sim2chip.stages.export import Export

    ctx = ctx_for(tmp_path)
    for cls, n in (("cow", 3), ("empty", 1)):
        d = ctx.job.stage_dir("ingest") / "test" / cls
        d.mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (96, 96)).save(d / f"{i}.png")
    seed = ctx.job.stage_dir("train") / "seed_0"
    seed.mkdir(parents=True)
    (seed / "model_int8.tflite").write_bytes(b"new")
    (seed / "model_float_io.tflite").write_bytes(b"new float")
    (ctx.job.stage_dir("train") / "summary.json").write_text(json.dumps({"0": {"val_acc": 0.9}}))
    fake_export(ctx.job, 0.25)
    snapshot(ctx.job, "added photos")
    cow = ctx.spec.labels.index("cow")
    # The new model says cow for everything (3 of 4 right), the old one says empty (1 of 4).
    monkeypatch.setattr(student, "predict_int8",
                        lambda blob, x: np.full(len(x), cow if blob == b"new" else 1 - cow))
    Evaluate().run(ctx)
    Export().run(ctx)
    report = json.loads((ctx.job.stage_dir("export") / "report.json").read_text())
    # then: v1's score when it was trained (its report), the same here since the test did not change.
    assert report["previous"] == {"version": 1, "score": 0.25, "now": 0.75, "then": 0.25}
    assert report["shipped_test"] == {"score": 0.75}

    fake_export(ctx.job, 0.75, labels=("empty", "cow", "dog"))  # the classes changed since
    snapshot(ctx.job, "spec changed")
    Evaluate().run(ctx)
    prev = json.loads((ctx.job.stage_dir("evaluate") / "previous.json").read_text())
    assert prev["version"] == 2 and "setup changed" in prev["skipped"]


def test_each_run_notes_the_eyeforge_that_ran_it(tmp_path, monkeypatch):
    from sim2chip import version
    from sim2chip.workspace import last_engine, note_run
    monkeypatch.setattr(version, "engine", lambda: "0.9.0+abc1234")
    job = open_job(tmp_path, "j")
    assert last_engine(job) is None  # runs from before it was noted
    note_run(job, "label")
    assert last_engine(job) == "0.9.0+abc1234"
