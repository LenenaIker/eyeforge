"""Web API with a scripted interview and a fake launcher: no LLM, no pipeline run."""
import json

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

    def launcher(home, job, request, backend, max_usd):
        launched.append((job.name, request, backend))
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
    assert "Sim2Chip" in app.get("/").text
