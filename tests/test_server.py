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
    # The same spec again is a new job: it must not overwrite barn_cam.
    assert app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()["job"] == "barn_cam_2"

    s = app.get("/api/jobs/barn_cam").json()
    states = {st["name"]: st["state"] for st in s["stages"]}
    assert states["fetch_assets"] == "running" and states["render"] == "pending" and "label" in states
    assert app.get("/api/jobs").json() == {"jobs": ["barn_cam", "barn_cam_2"]}
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


def test_count_review_points_and_big_thumbnail(app, tmp_path):
    from PIL import Image
    spec = {**GOOD, "task": "count", "user_data": {"raw": "photos"}}
    job = tmp_path / "jobs" / "barn_cam"
    (job / "label" / "images").mkdir(parents=True)
    (job / "logs").mkdir()
    (job / "request.json").write_text(json.dumps(spec))
    Image.new("RGB", (1600, 1200)).save(job / "label" / "images" / "a.jpg")
    (job / "label" / "verdicts.json").write_text(json.dumps(
        {"a.jpg": {"objects": [{"x": 0.5, "y": 0.5, "class": "cow", "box": [0.4, 0.4, 0.6, 0.6]}],
                   "assigned": "images"}}))
    r = app.get("/api/jobs/barn_cam/review").json()
    assert r["task"] == "count" and r["photos"][0]["points"] == [[0.5, 0.5, "cow"]]
    small = app.get("/api/jobs/barn_cam/media/label/images/a.jpg")
    big = app.get("/api/jobs/barn_cam/media/label/images/a.jpg?size=1024")
    assert len(big.content) > len(small.content)
    r = app.post("/api/jobs/barn_cam/corrections", json={"points": {"a.jpg": [[0.1, 0.2, "cow"], [0.5, 0.5, "cow"]]}})
    assert r.json() == {"rerun_from": "label"}
    saved = json.loads((job / "input" / "corrections.json").read_text())
    assert saved["points"] == {"a.jpg": [[0.1, 0.2, "cow"], [0.5, 0.5, "cow"]]}


def test_public_server_needs_the_key_and_uploaded_photos_only(tmp_path):
    def launcher(home, job, request, backend, max_usd, start=None):
        (job.logs / "pipeline.log").write_text("job x\n")
        return FakeProc()

    app = create_app(tmp_path, backend="nebius", launcher=launcher, public=True, key="s3cret", max_runs=1)
    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/?key=s3cret").status_code == 200  # sets the cookie for the next requests
    assert client.get("/api/jobs").status_code == 200
    # A server path in the spec is refused: it would be read and uploaded to the job bucket.
    spec = {**GOOD, "user_data": {"raw": str(Path.home())}}
    r = client.post("/api/jobs", data={"spec": json.dumps(spec)})
    assert r.status_code == 400 and "uploaded" in r.json()["detail"]
    files = [("photos", ("a.jpg", b"x", "image/jpeg"))]
    assert client.post("/api/jobs", data={"spec": json.dumps(GOOD)}, files=files).status_code == 200
    r = client.post("/api/jobs", data={"spec": json.dumps(GOOD)}, files=files)
    assert r.status_code == 429  # one run at a time


def test_uploads_only_interview_never_reads_server_folders(tmp_path):
    cats = interview.CategoryIndex(counts={"cow": 62})
    s = interview.Session(Scripted([{"describe_scene": {"folder": str(tmp_path), "classes": "cows"}},
                                    {"say": "ok"}]), "m", cats, uploads=tmp_path / "uploads" / "sid",
                          scene_profiler=lambda folder, classes: pytest.fail("read a server folder"),
                          uploads_only=True)
    assert "no access to folders" in s.messages[0]["content"]
    assert s.reply("my photos are in that folder").text == "ok"


def test_cloud_job_left_running_shows_and_can_be_cancelled(app, tmp_path, monkeypatch):
    import time

    from sim2chip.backends import nebius
    cancelled = []
    monkeypatch.setattr(nebius.NebiusCLI, "cancel", lambda self, job_id: cancelled.append(job_id))
    job = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()["job"]
    logs = tmp_path / "jobs" / job / "logs"
    app.launched.clear()
    # A server restart forgets its processes; the marker the run left says a cloud job may still bill.
    (logs / "cloud_job.json").write_text(json.dumps({"id": "aijob-1", "started": time.time(), "hours": 4}))
    monkeypatch.setattr(FakeProc, "poll", lambda self: 0)
    assert app.get(f"/api/jobs/{job}").json()["running"]
    assert app.post(f"/api/jobs/{job}/corrections", json={"labels": {"x": "cow"}}).status_code == 409
    assert app.post(f"/api/jobs/{job}/cancel").json()["cloud_job"] == "aijob-1"
    assert cancelled == ["aijob-1"] and not app.get(f"/api/jobs/{job}").json()["running"]


def test_daily_cloud_budget(tmp_path):
    import time
    logs = tmp_path / "jobs" / "old" / "logs"
    logs.mkdir(parents=True)
    (logs / "cloud_usage.jsonl").write_text(json.dumps({"t": time.time(), "usd_estimate": 9.0}) + "\n")
    client = TestClient(create_app(tmp_path, backend="nebius", launcher=lambda *a, **k: FakeProc(), daily_usd=10))
    r = client.post("/api/jobs", data={"spec": json.dumps(GOOD)})
    assert r.status_code == 429 and "budget" in r.json()["detail"]


def test_the_same_setup_sent_twice_runs_once(app):
    first = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()
    again = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()
    assert again["job"] == first["job"] and again["already_running"] and len(app.launched) == 1


def test_timing_estimates_the_time_left_from_earlier_runs(app, tmp_path):
    import time
    old = tmp_path / "jobs" / "old" / "logs"
    old.mkdir(parents=True)
    t = time.time() - 3600
    steps = [("upload job", 10, "local"), ("nebius job: provisioning", 60, "nebius"), ("label", 100, "nebius"),
             ("fetch_assets", 200, "nebius"), ("render", 300, "local")]  # a local render is not a cloud one
    (old / "timeline.jsonl").write_text("".join(json.dumps({"t": t, "step": s, "seconds": sec, "where": w}) + "\n"
                                                for s, sec, w in steps))
    job = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()["job"]
    s = app.get(f"/api/jobs/{job}").json()
    assert s["timing"]["elapsed_s"] < 5
    # Nothing ran yet: upload, the GPU wait and fetch_assets (running, just started) are left; render never
    # ran on Nebius, so it adds nothing.
    assert 265 <= s["timing"]["eta_s"] <= 270
    logs = tmp_path / "jobs" / job / "logs"
    (logs / "timeline.jsonl").write_text(json.dumps({"t": time.time(), "step": "upload job", "seconds": 8,
                                                     "where": "local"}) + "\n")
    assert 255 <= app.get(f"/api/jobs/{job}").json()["timing"]["eta_s"] <= 260


def test_public_visitors_only_see_their_own_jobs(tmp_path):
    def launcher(home, job, request, backend, max_usd, start=None):
        (job.logs / "pipeline.log").write_text("job x\n")
        return FakeProc()

    app = create_app(tmp_path, backend="nebius", launcher=launcher, public=True, max_runs=5)
    alice, bob = TestClient(app), TestClient(app)
    job = alice.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()["job"]
    assert alice.get("/api/jobs").json() == {"jobs": [job]} and alice.get(f"/api/jobs/{job}").status_code == 200
    assert bob.get("/api/jobs").json() == {"jobs": []}
    for path in (f"/api/jobs/{job}", f"/api/jobs/{job}/review", f"/api/jobs/{job}/files/report.json"):
        assert bob.get(path).status_code == 404
    assert bob.post(f"/api/jobs/{job}/cancel").status_code == 404
    # The same setup from another visitor is their own run, not a link to someone else's.
    assert bob.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()["job"] != job
    cfg = bob.get("/api/config").json()
    assert cfg["public"] and not cfg["busy"] and cfg["max_runs"] == 5


def test_app_routes_serve_the_build_or_the_old_page(tmp_path, monkeypatch):
    from sim2chip import server
    client = TestClient(create_app(tmp_path, launcher=lambda *a, **k: FakeProc()))
    monkeypatch.setattr(server, "DIST", tmp_path / "missing")
    assert "EyeForge" in client.get("/app/job/x").text  # no build: the old page
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(server, "DIST", dist)
    assert client.get("/").text == "<div id=root></div>"
    assert client.get("/app/job/x").text == "<div id=root></div>"  # the app routes itself
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/nothing").status_code == 404
    assert client.get("/../../etc/passwd").text == "<div id=root></div>"
