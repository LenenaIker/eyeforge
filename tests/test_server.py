"""Web API with a scripted interview and a fake launcher: no LLM, no pipeline run."""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sim2chip import interview  # noqa: E402
from sim2chip.server import create_app  # noqa: E402

GOOD = {"name": "Barn Cam", "classes": [{"id": "cow", "description": "cows", "size_m": {"measure": "length",
        "range": [2, 2.5]}, "assets": {"categories": ["cow"], "count": 10}}]}


class Scripted:
    def __init__(self, actions):
        self.actions = list(actions)

    def chat(self, messages, model, purpose, **kw):
        return json.dumps(self.actions.pop(0))


class FakeProc:
    def poll(self):
        return None


@pytest.fixture
def app(tmp_path):
    launched = []

    def interviewer():
        cats = interview.CategoryIndex(counts={"cow": 62})
        return interview.Session(Scripted([{"say": "Where is the camera?"}, {"search": ["cow"]}, {"spec": GOOD}]),
                                 "m", cats)

    def launcher(home, job, request, backend, max_usd, start=None):
        launched.append((job.name, request, backend, start))
        (job.logs / "pipeline.log").write_text("job x\n> fetch_assets (cpu)\n")
        return FakeProc()

    client = TestClient(create_app(tmp_path, backend="nebius", interviewer=interviewer, launcher=launcher))
    client.launched = launched
    return client


def test_interview_over_http(app):
    sid = app.post("/api/interview").json()["session"]
    assert app.post(f"/api/interview/{sid}", json={"text": "cows"}).json() == {"kind": "say",
                                                                                "text": "Where is the camera?"}
    t = app.post(f"/api/interview/{sid}", json={"text": "on a post"}).json()
    assert t["kind"] == "confirm" and t["spec"]["name"] == "Barn Cam"
    t = app.post(f"/api/interview/{sid}", json={"text": "sí, adelante"}).json()
    assert t["kind"] == "done"
    assert app.post(f"/api/interview/{sid}", json={"text": "x"}).status_code == 404


def test_job_with_photos_runs_and_reports(app, tmp_path):
    files = [("photos", ("a.jpg", b"jpg", "image/jpeg")), ("photos", ("notes.txt", b"hi", "text/plain"))]
    r = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}, files=files)
    assert r.status_code == 200 and r.json() == {"job": "barn_cam", "photos": 1}
    job = tmp_path / "jobs" / "barn_cam"
    assert (job / "input" / "photos" / "a.jpg").exists() and not (job / "input" / "photos" / "notes.txt").exists()
    assert json.loads((job / "request.json").read_text())["user_data"]["raw"] == "photos"
    assert app.launched[0][0] == "barn_cam" and app.launched[0][2] == "nebius"
    assert app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).status_code == 409  # already running

    s = app.get("/api/jobs/barn_cam").json()
    states = {st["name"]: st["state"] for st in s["stages"]}
    assert states["fetch_assets"] == "running" and states["render"] == "pending" and "label" in states
    assert app.get("/api/jobs").json() == {"jobs": ["barn_cam"]}
    assert app.get("/api/jobs/barn_cam/files/model_data.h").status_code == 404

    export = job / "export"
    export.mkdir()
    (export / "model_data.h").write_text("// model")
    assert app.get("/api/jobs/barn_cam/files/model_data.h").text == "// model"
    assert app.get("/api/jobs/barn_cam/files/spec.json").status_code == 404  # not downloadable


def test_invalid_spec(app):
    assert app.post("/api/jobs", data={"spec": '{"name": "x"}'}).status_code == 400
    assert app.post("/api/jobs", data={"spec": "not json"}).status_code == 400


def test_index(app):
    assert "EyeForge" in app.get("/").text


def test_status_survives_stale_and_invalid_specs(app, tmp_path):
    import os
    job = tmp_path / "jobs" / "old_cam"
    (job / "logs").mkdir(parents=True)
    bad = {**GOOD, "scene": {"distractors": ["bicycle"]}}  # accepted by an older version, not by this one
    (job / "spec.json").write_text(json.dumps(bad))
    s = app.get("/api/jobs/old_cam")
    assert s.status_code == 200 and s.json()["stages"] == [] and "distractors" in s.json()["log"][-1]
    # Relaunched: the new request is newer than the previous run's spec.json, so it wins.
    (job / "request.json").write_text(json.dumps(GOOD))
    os.utime(job / "spec.json", (1, 1))
    assert app.get("/api/jobs/old_cam").json()["stages"]


def test_review_corrections_relaunch_and_media_is_fenced(app, tmp_path):
    from PIL import Image
    job = tmp_path / "jobs" / "barn_cam"
    (job / "label" / "cow").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "request.json").write_text(json.dumps(GOOD))
    Image.new("RGB", (800, 600)).save(job / "label" / "cow" / "a.jpg")
    (job / "label" / "verdicts.json").write_text(json.dumps({"a.jpg": {"class": "cow", "assigned": "cow"}}))
    r = app.get("/api/jobs/barn_cam/review").json()
    assert r["photos"][0]["image"] == "label/cow/a.jpg"
    thumb = app.get("/api/jobs/barn_cam/media/label/cow/a.jpg")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    for bad in ("request.json", "../barn_cam/request.json", "label/../../../etc/passwd"):
        assert app.get(f"/api/jobs/barn_cam/media/{bad}").status_code == 404
    r = app.post("/api/jobs/barn_cam/corrections", json={"labels": {"a.jpg": "empty"}})
    assert r.json() == {"rerun_from": "label"} and app.launched[-1][3] == "label"
    # The relaunched job is still running (FakeProc never ends): no second relaunch on top of it.
    assert app.post("/api/jobs/barn_cam/corrections", json={}).status_code == 409


def test_chat_photos_travel_with_the_job(app, tmp_path):
    sid = app.post("/api/interview").json()["session"]
    files = [("photos", ("a.jpg", b"jpg", "image/jpeg")), ("photos", ("b.jpg", b"jpg", "image/jpeg"))]
    r = app.post(f"/api/interview/{sid}/photos", data={"purpose": "training"}, files=files)
    assert r.status_code == 200 and r.json()["photos"] == 2 and r.json()["kind"] == "say"
    upload = tmp_path / "uploads" / sid / "training"
    spec = {**GOOD, "user_data": {"raw": str(upload)}}
    assert app.post("/api/jobs", data={"spec": json.dumps(spec)}).status_code == 200
    job = tmp_path / "jobs" / "barn_cam"
    raw = json.loads((job / "request.json").read_text())["user_data"]["raw"]
    assert not Path(raw).is_absolute() and (job / "input" / raw / "a.jpg").exists()
    assert app.get(f"/api/interview/{sid}/thumb", params={"file": str(upload / "a.jpg")}).status_code == 404
