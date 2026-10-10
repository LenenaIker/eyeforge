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

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def built_app(tmp_path, monkeypatch):
    """A stand-in for the web app build (web/dist), which a checkout without node does not have."""
    from sim2chip import server
    dist = tmp_path / "_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id=root></div>")
    monkeypatch.setattr(server, "DIST", dist)


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


def labeled_job(tmp_path, spec, n=2, task="classify", classes=("cow", "empty")):
    """A job made by the server from a library dataset of n photos and a labeling of it, as adopt_library does:
    (job folder, library, labeling id, the job's names of the photos)."""
    from PIL import Image

    from sim2chip.library import Library
    lib = Library(tmp_path, tmp_path / "cache")
    up = tmp_path / "_up"
    up.mkdir(exist_ok=True)
    files = []
    for i in range(n):
        Image.new("RGB", (800, 600), (i * 40, 0, 0)).save(up / f"{i}.jpg")
        files.append(up / f"{i}.jpg")
    ds = lib.new_dataset("local", "Barn", files)
    lid = lib.create_labeling(ds, "Cows", task, [{"key": c} for c in classes])
    job = tmp_path / "jobs" / "barn_cam"
    (job / "logs").mkdir(parents=True)
    folder = lib.export(lid, job / "input")
    (job / "request.json").write_text(json.dumps({**spec, "user_data": {"raw": folder, "labeling": lid}}))
    names = sorted(json.loads((job / "input" / "labeling.json").read_text())["photos"])
    return job, lib, lid, names


def test_interview_over_http(app):
    sid = app.post("/api/interview").json()["session"]
    t = app.post(f"/api/interview/{sid}", json={"text": "cows"}).json()
    assert (t["kind"], t["text"]) == ("say", "Where is the camera?")
    t = app.post(f"/api/interview/{sid}?act=turn2", json={"text": "on a post"}).json()
    assert t["kind"] == "confirm" and t["spec"]["name"] == "Barn Cam" and t["intro"] == ""
    # The turn's steps come with the answer and, under its id, from the activity poll (not under another id).
    assert [s["text"] for s in t["activity"]][:2] == ["Thinking", "Searching the 3D model catalog: cow"]
    assert t["activity"][-1]["text"] == "Checking the proposed setup" and t["activity"][-1]["state"] == "done"
    polled = app.get(f"/api/interview/{sid}/activity", params={"act": "turn2"}).json()
    assert [s["text"] for s in polled["steps"]] == [s["text"] for s in t["activity"]] and polled["now"] > 0
    assert app.get(f"/api/interview/{sid}/activity", params={"act": "old"}).json()["steps"] == []
    t = app.post(f"/api/interview/{sid}", json={"text": "sí, adelante"}).json()
    assert t["kind"] == "done"
    assert app.post(f"/api/interview/{sid}", json={"text": "x"}).status_code == 404


def test_job_with_photos_runs_and_reports(app, tmp_path):
    files = [("photos", ("a.jpg", b"jpg", "image/jpeg")), ("photos", ("notes.txt", b"hi", "text/plain"))]
    r = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}, files=files)
    assert r.status_code == 200 and r.json() == {"job": "barn_cam", "photos": 1}
    job = tmp_path / "jobs" / "barn_cam"
    assert (job / "input" / "photos" / "a.jpg").exists() and not (job / "input" / "photos" / "notes.txt").exists()
    # The photos are a dataset of the library, labeled in a new labeling the job takes a copy of.
    ud = json.loads((job / "request.json").read_text())["user_data"]
    assert ud["raw"] == f"labeling_{ud['labeling']}" and (job / "input" / "labeling.json").exists()
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
    assert app.get("/").text == "<div id=root></div>"


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
    job, lib, lid, names = labeled_job(tmp_path, GOOD)
    (job / "label" / "cow").mkdir(parents=True)
    Image.new("RGB", (800, 600)).save(job / "label" / "cow" / "a.jpg")
    (job / "label" / "verdicts.json").write_text(json.dumps({names[0]: {"class": "cow", "assigned": "cow",
                                                                        "copy": "a.jpg"}}))
    (job / "export").mkdir()
    (job / "export" / "model_int8.tflite").write_bytes(b"v1")
    (job / "export" / "report.json").write_text(json.dumps({"metric": "accuracy", "val_score": 0.5, "seed": 0}))
    r = app.get("/api/jobs/barn_cam/review").json()
    assert r["photos"][0]["image"] == "label/cow/a.jpg" and not r["photos"][0]["corrected"]
    thumb = app.get("/api/jobs/barn_cam/media/label/cow/a.jpg")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    for bad in ("request.json", "../barn_cam/request.json", "label/../../../etc/passwd"):
        assert app.get(f"/api/jobs/barn_cam/media/{bad}").status_code == 404
    assert app.post("/api/jobs/barn_cam/corrections", json={"labels": {names[0]: "horse"}}).status_code == 400
    r = app.post("/api/jobs/barn_cam/corrections", json={"labels": {names[0]: "empty"}})
    assert r.json() == {"rerun_from": "label"} and app.launched[-1][3] == "label"
    # The correction is the photo's truth in the labeling, and the job's copy of it says so.
    photo = next(p for p in lib.labeling_photos(lid) if f"{p['id']}.jpg" == names[0])
    assert photo["reviewed"] and photo["marks"][0]["class"] == "empty" and photo["marks"][0]["source"] == "user"
    assert json.loads((job / "input" / "labeling.json").read_text())["photos"][names[0]]["reviewed"]
    assert app.get("/api/jobs/barn_cam/review").json()["photos"][0]["label"] == "empty"
    # The model it replaces is kept as v1 and can still be downloaded.
    s = app.get("/api/jobs/barn_cam").json()
    assert [v["n"] for v in s["versions"]] == [1] and s["version"]["n"] == 2
    assert s["version"]["reason"] == "corrected labels"
    assert app.get("/api/jobs/barn_cam/versions/1/files/model_int8.tflite").content == b"v1"
    assert app.get("/api/jobs/barn_cam/versions/1/files/spec.json").status_code == 404
    # The relaunched job is still running (FakeProc never ends): no second relaunch on top of it.
    assert app.post("/api/jobs/barn_cam/corrections", json={}).status_code == 409


def test_add_photos_to_a_trained_job(app, tmp_path):
    import io

    from PIL import Image
    job, lib, lid, names = labeled_job(tmp_path, {**GOOD, "task": "count"}, task="count", classes=("cow",))
    (job / "export").mkdir()
    (job / "export" / "model_int8.tflite").write_bytes(b"v1")
    (job / "export" / "report.json").write_text(json.dumps({"metric": "f1", "real_test": {"mean": 0.2}}))

    def jpg(color):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), color).save(buf, "JPEG")
        return buf.getvalue()
    files = [("photos", ("a.jpg", jpg("white"), "image/jpeg")), ("photos", ("b.jpg", jpg("yellow"), "image/jpeg")),
             ("photos", ("notes.txt", b"no", "text/plain"))]
    said = {"a.jpg": {"points": []}, "b.jpg": {"points": [[0.5, 0.25, "cow"], [2, 0, "cow"]]}, "zzz.jpg": {"label": "x"}}
    r = app.post("/api/jobs/barn_cam/photos", data={"note": "misses  dry\npatches", "labels": json.dumps(said)},
                 files=files)
    assert r.json() == {"rerun_from": "label", "photos": 2} and app.launched[-1][3] == "label"
    # The photos join the labeling's dataset; what the user said in the live test is their review.
    photos = lib.labeling_photos(lid)
    assert len(photos) == 4 and sum(p["reviewed"] for p in photos) == 2
    assert sorted(m["x"] for p in photos for m in p["marks"]) == [0.5]
    assert len(json.loads((job / "input" / "labeling.json").read_text())["photos"]) == 4
    s = app.get("/api/jobs/barn_cam").json()
    assert s["version"]["reason"] == "added 2 photos: misses dry patches" and s["versions"][0]["headline"] == 0.2
    assert app.post("/api/jobs/barn_cam/photos", files=files[:1]).status_code == 409  # still running


def test_add_photos_leaves_linked_folders_alone(app, tmp_path):
    job = tmp_path / "jobs" / "linked"
    (job / "logs").mkdir(parents=True)
    (job / "input").mkdir()
    (tmp_path / "mine").mkdir()
    (job / "input" / "photos").symlink_to(tmp_path / "mine")
    (job / "spec.json").write_text(json.dumps({**GOOD, "user_data": {"raw": "photos"}}))
    r = app.post("/api/jobs/linked/photos", files=[("photos", ("a.jpg", b"x", "image/jpeg"))])
    assert r.status_code == 400 and not list((tmp_path / "mine").iterdir())
    r = app.post("/api/jobs/linked/photos", files=[("photos", ("a.txt", b"x", "text/plain"))])
    assert r.status_code == 400


def test_revise_a_trained_job(app, tmp_path):
    from sim2chip import spec as spec_mod
    from sim2chip.pipeline import Pipeline
    from sim2chip.workspace import Cache, Job
    job = Job(tmp_path / "jobs" / "barn_cam")
    (job.logs).mkdir(parents=True)
    job.spec_path.write_text(json.dumps(GOOD))
    pipe = Pipeline(spec_mod.from_dict(GOOD), job, Cache(tmp_path / "cache"))
    for name, fp in pipe.fingerprints().items():
        job.mark_done(name, fp)
    (job.root / "export" / "model_int8.tflite").write_bytes(b"v1")
    (job.root / "export" / "report.json").write_text(json.dumps({"metric": "accuracy", "val_score": 0.4}))

    r = app.post("/api/interview", params={"job": "barn_cam"}).json()
    assert r["job"] == "barn_cam" and "change" in r["text"]
    assert app.post("/api/interview", params={"job": "nope"}).status_code == 404

    jetson = json.dumps({**GOOD, "hardware": "jetson_nano"})
    r = app.post("/api/jobs/barn_cam/revise", data={"spec": jetson, "dry": "true"}).json()
    assert r["changed"] == ["hardware"] and "render" not in r["stages"] and "train" in r["stages"]
    assert not app.launched and not (job.root / "versions").exists()  # dry: nothing happens
    r = app.post("/api/jobs/barn_cam/revise", data={"spec": jetson}).json()
    assert app.launched[-1][0] == "barn_cam" and app.launched[-1][3] is None and r["rerun_from"] == r["stages"][0]
    s = app.get("/api/jobs/barn_cam").json()
    assert s["version"]["reason"] == "changed hardware" and s["versions"][0]["n"] == 1
    assert s["spec"]["hardware"] == "jetson_nano"
    assert app.post("/api/jobs/barn_cam/revise", data={"spec": jetson}).status_code == 409  # running


def test_chat_photos_travel_with_the_job(app, tmp_path):
    sid = app.post("/api/interview").json()["session"]
    files = [("photos", ("a.jpg", b"jpg-a", "image/jpeg")), ("photos", ("b.jpg", b"jpg-b", "image/jpeg"))]
    r = app.post(f"/api/interview/{sid}/photos", data={"purpose": "training"}, files=files)
    assert r.status_code == 200 and r.json()["photos"] == 2 and r.json()["kind"] == "say"
    (chat,) = app.get("/api/library").json()["datasets"]  # the chat's own dataset
    upload = tmp_path / "uploads" / sid / "training"
    spec = {**GOOD, "user_data": {"raw": str(upload)}}
    assert app.post("/api/jobs", data={"spec": json.dumps(spec), "session": sid}).status_code == 200
    job = tmp_path / "jobs" / "barn_cam"
    ud = json.loads((job / "request.json").read_text())["user_data"]
    assert not Path(ud["raw"]).is_absolute() and len(list((job / "input" / ud["raw"]).iterdir())) == 2
    (d,) = app.get("/api/library").json()["datasets"]
    assert d["id"] == chat["id"] and d["photos"] == 2 and d["labelings"][0]["id"] == ud["labeling"]
    assert app.get(f"/api/interview/{sid}/thumb", params={"file": str(upload / "a.jpg")}).status_code == 404


def test_photos_in_batches_make_one_turn_and_one_rerun(app, tmp_path):
    """A proxy caps the request size: the app sends big uploads in batches and only the last one acts."""
    sid = app.post("/api/interview").json()["session"]
    batch = lambda *names: [("photos", (n, n.encode(), "image/jpeg")) for n in names]  # noqa: E731
    r = app.post(f"/api/interview/{sid}/photos", data={"more": "true"}, files=batch("a.jpg", "b.jpg"))
    assert r.json() == {"stored": 2}
    r = app.post(f"/api/interview/{sid}/photos", data={"total": "3"}, files=batch("c.jpg"))
    assert r.json()["photos"] == 3 and r.json()["kind"] == "say"
    assert len(list((tmp_path / "uploads" / sid / "training").iterdir())) == 3

    job, lib, lid, names = labeled_job(tmp_path, GOOD)
    r = app.post("/api/jobs/barn_cam/photos", data={"more": "true"}, files=batch("x.jpg", "y.jpg"))
    assert r.json() == {"stored": 2} and not app.launched
    r = app.post("/api/jobs/barn_cam/photos", data={"total": "3", "note": "dusk"}, files=batch("z.jpg"))
    assert r.json() == {"rerun_from": "label", "photos": 3} and len(app.launched) == 1
    assert len(lib.labeling_photos(lid)) == 5


def test_count_review_points_and_big_thumbnail(app, tmp_path):
    from PIL import Image
    job, lib, lid, names = labeled_job(tmp_path, {**GOOD, "task": "count"}, task="count", classes=("cow",))
    (job / "label" / "images").mkdir(parents=True)
    Image.new("RGB", (1600, 1200)).save(job / "label" / "images" / "a.jpg")
    (job / "label" / "verdicts.json").write_text(json.dumps(
        {names[0]: {"objects": [{"x": 0.5, "y": 0.5, "class": "cow", "box": [0.4, 0.4, 0.6, 0.6]}],
                    "assigned": "images", "copy": "a.jpg"}}))
    r = app.get("/api/jobs/barn_cam/review").json()
    assert r["task"] == "count" and r["photos"][0]["points"] == [[0.5, 0.5, "cow"]]
    small = app.get("/api/jobs/barn_cam/media/label/images/a.jpg")
    big = app.get("/api/jobs/barn_cam/media/label/images/a.jpg?size=1024")
    assert len(big.content) > len(small.content)
    # Test photos' marks also carry the box size ([x, y, class, w, h]): only the point is kept.
    r = app.post("/api/jobs/barn_cam/corrections",
                 json={"points": {names[0]: [[0.1, 0.2, "cow"], [0.5, 0.5, "cow", 0.2, 0.2]]}})
    assert r.json() == {"rerun_from": "label"}
    marks = next(p for p in lib.labeling_photos(lid) if f"{p['id']}.jpg" == names[0])["marks"]
    assert [(m["x"], m["y"], m["w"]) for m in marks] == [(0.1, 0.2, None), (0.5, 0.5, None)]


def test_public_server_needs_the_key_and_uploaded_photos_only(tmp_path):
    def launcher(home, job, request, backend, max_usd, start=None):
        (job.logs / "pipeline.log").write_text("job x\n")
        return FakeProc()

    app = create_app(tmp_path, backend="nebius", launcher=launcher, public=True, key="s3cret", max_runs=1)
    client = TestClient(app)
    assert client.get("/").status_code in (200, 503)  # the landing is open (503: no web build in tests)
    assert client.get("/api/config").json()["locked"] is True
    assert client.get("/api/jobs").status_code == 401
    assert client.post("/api/key", json={"key": "wrong"}).status_code == 401
    assert client.get("/api/jobs").status_code == 401
    assert client.post("/api/key", json={"key": "s3cret"}).status_code == 200  # sets the cookie
    assert client.get("/api/jobs").status_code == 200
    assert client.get("/api/config").json()["locked"] is False
    other = TestClient(app)
    assert other.get("/?key=s3cret").status_code in (200, 503)  # an access link sets the cookie too
    assert other.get("/api/jobs").status_code == 200
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


def test_app_routes_serve_the_build(tmp_path, monkeypatch):
    from sim2chip import server
    client = TestClient(create_app(tmp_path, launcher=lambda *a, **k: FakeProc()))
    monkeypatch.setattr(server, "DIST", tmp_path / "missing")
    r = client.get("/app/job/x")
    assert r.status_code == 503 and "npm run build" in r.text  # no build: say how to make one
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


def test_calibrate_the_teacher_before_the_first_run(app, tmp_path, monkeypatch):
    """A new job with raw photos: the teacher marks a few, the user corrects them, the next round tunes the
    teacher to them and marks new ones, and only "train" launches the run."""
    import io
    import time

    from PIL import Image

    from sim2chip import teacher
    from sim2chip.job_service import CALIBRATION_PHOTOS

    asked = []

    def embed(items, texts, text_out, log=print, timeline=None):  # SigLIP 2 without a GPU: every photo along its own axis
        for path, out in items:
            asked.append(path)
            i = (int(path.stem) - 1) % 20  # the job's photos are named by their library id, 1 to 20
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"embedding": [1.0 if j == i else 0.0 for j in range(22)]}))
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(json.dumps({"embeddings": [[1.0 if j == 20 + k else 0.0 for j in range(22)]
                                                       for k in range(len(texts))]}))
    monkeypatch.setattr(teacher, "embed", embed)
    monkeypatch.setattr(teacher, "prompts", lambda spec, client=None: {k: k for k in spec.labels})
    monkeypatch.setattr(teacher, "available", lambda: True)

    def photo(i):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (i * 9, 0, 0)).save(buf, "JPEG")
        return ("photos", (f"p{i:02d}.jpg", buf.getvalue(), "image/jpeg"))

    def settle(job):
        for _ in range(200):
            st = app.get(f"/api/jobs/{job}").json()
            if not st["running"]:
                return st
            time.sleep(0.02)
        raise AssertionError("calibration did not finish")

    r = app.post("/api/jobs", data={"spec": json.dumps(GOOD), "calibrate": "true"},
                 files=[photo(i) for i in range(20)]).json()
    assert r["calibrating"] and not app.launched
    st = settle(r["job"])
    assert st["calibration"] == {"state": "ready"}
    photos = app.get(f"/api/jobs/{r['job']}/review").json()["photos"]
    # Every photo is embedded (cheap, and needed to pick varied ones); the user sees only a few.
    assert len(photos) == CALIBRATION_PHOTOS and len(asked) == 20
    # Corrections: another round over those photos and new ones, tuned to them. Still no run.
    fixes = {photos[0]["key"]: "cow", photos[1]["key"]: "empty"}
    assert app.post(f"/api/jobs/{r['job']}/corrections", json={"labels": fixes}).json() == {
        "rerun_from": "calibration"}
    settle(r["job"])
    review = app.get(f"/api/jobs/{r['job']}/review").json()
    assert len(review["photos"]) == 2 * CALIBRATION_PHOTOS and len(asked) == 20
    assert "2 photos you corrected" in review["teacher_rules"][0] and not app.launched
    # Train: the run starts from the calibrated answers.
    assert app.post(f"/api/jobs/{r['job']}/train", json={}).json() == {"rerun_from": None}
    assert app.launched and app.launched[-1][3] is None
    assert app.get(f"/api/jobs/{r['job']}").json()["calibration"] is None
    assert app.post(f"/api/jobs/{r['job']}/train", json={}).status_code == 409
    # The check still open on another device: its fixes start nothing once training started.
    app.launched.clear()
    stale = app.post(f"/api/jobs/{r['job']}/corrections", json={"labels": fixes, "calibrating": True})
    assert stale.status_code == 409 and "another device" in stale.json()["detail"] and not app.launched


def test_the_interview_is_credited_to_the_job_it_creates(tmp_path):
    """The usage panel shows the interview's model calls in the job its spec became, once."""
    class Logged(Scripted):
        usage_log = None

        def chat(self, messages, model, purpose, **kw):
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_log, "a") as f:
                f.write(json.dumps({"model": "nvidia/nemotron-3-super-120b-a12b", "purpose": purpose,
                                    "prompt_tokens": 100, "completion_tokens": 10}) + "\n")
            return super().chat(messages, model, purpose, **kw)

    def interviewer():
        return interview.Session(Logged([{"say": "Where is the camera?"}]), "m", interview.CategoryIndex(counts={}))

    app = TestClient(create_app(tmp_path, backend="nebius", interviewer=interviewer,
                                launcher=lambda *a, **k: FakeProc()))
    sid = app.post("/api/interview").json()["session"]
    app.post(f"/api/interview/{sid}", json={"text": "cows"})
    job = app.post("/api/jobs", data={"spec": json.dumps(GOOD), "session": sid}).json()["job"]
    [m] = app.get(f"/api/jobs/{job}").json()["usage"]["models"]
    assert (m["steps"], m["calls"], m["input_tokens"], m["nvidia"]) == (["interview"], 1, 100, True)
    other = app.post("/api/jobs", data={"spec": json.dumps({**GOOD, "background_class": "nothing"}), "session": sid}).json()["job"]
    assert app.get(f"/api/jobs/{other}").json()["usage"]["models"] == []  # counted once
    assert app.post("/api/jobs", data={"spec": json.dumps(GOOD), "session": "../../x"}).status_code == 200


def test_photos_in_the_interview_warm_up_a_gpu_and_rounds_with_answers_skip_it(tmp_path, monkeypatch):
    """The warm GPU is asked for at the first photo upload, kept while the user checks the teacher, and a round
    whose answers are all cached here runs without any GPU (no cloud job, no teacher)."""
    import io
    import time

    from PIL import Image

    from sim2chip import teacher
    from sim2chip.stages.label import Label

    class Warm:
        def __init__(self):
            self.calls = []

        def start(self, reason="", who=None):
            self.calls.append(("start", reason))

        def touch(self, seconds=0, who=None):
            self.calls.append(("touch",))

    gpu = Warm()

    def interviewer():
        return interview.Session(Scripted([{"say": "Where is the camera?"}]), "m",
                                 interview.CategoryIndex(counts={"cow": 62}))
    client = TestClient(create_app(tmp_path, backend="nebius", interviewer=interviewer, warm=gpu,
                                   launcher=lambda *a, **k: FakeProc()))
    sid = client.post("/api/interview").json()["session"]
    client.post(f"/api/interview/{sid}/photos", data={"purpose": "training"},
                files=[("photos", ("a.jpg", b"jpg", "image/jpeg"))])
    assert gpu.calls == [("start", "photos uploaded")]

    monkeypatch.setattr(teacher, "available", lambda: False)  # this server has no GPU
    monkeypatch.setattr(Label, "gpu_needed", lambda self, ctx: False)  # but every answer is cached

    def embed(items, texts, text_out, log=print, timeline=None):
        for _path, out in items:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"embedding": [1.0, 0.0, 0.0]}))
        text_out.parent.mkdir(parents=True, exist_ok=True)
        text_out.write_text(json.dumps({"embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]][:len(texts)]}))
    monkeypatch.setattr(teacher, "embed", embed)
    monkeypatch.setattr(teacher, "prompts", lambda spec, client=None: {k: k for k in spec.labels})
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "JPEG")
    r = client.post("/api/jobs", data={"spec": json.dumps(GOOD), "calibrate": "true"},
                    files=[("photos", ("p.jpg", buf.getvalue(), "image/jpeg"))]).json()
    for _ in range(200):
        st = client.get(f"/api/jobs/{r['job']}").json()
        if not st["running"]:
            break
        time.sleep(0.02)
    assert st["calibration"] == {"state": "ready"}
    # The check needs no GPU: nothing keeps it up for the user (it stops after the prefetch).
    assert ("start", "calibration") in gpu.calls and ("touch",) not in gpu.calls
    assert not (tmp_path / "jobs" / r["job"] / "logs" / "pipeline.log").exists()  # no cloud run was launched


def test_a_failed_job_can_be_tried_again(app, tmp_path):
    job = tmp_path / "jobs" / "barn_cam"
    (job / "logs").mkdir(parents=True)
    (job / "spec.json").write_text(json.dumps({**GOOD, "user_data": {"raw": "photos"}}))
    r = app.post("/api/jobs/barn_cam/retry")
    assert r.status_code == 200 and app.launched[-1][0] == "barn_cam"
    assert app.post("/api/jobs/barn_cam/retry").status_code == 409  # running again


def test_a_chat_goes_on_after_the_server_restarts(tmp_path):
    """A deploy restarts the server: the chat comes back from its transcript, its proposal still waiting."""
    def interviewer():
        cats = interview.CategoryIndex(counts={"cow": 62})
        return interview.Session(Scripted([{"say": "Where is the camera?"}, {"spec": GOOD}]), "m", cats)

    before = TestClient(create_app(tmp_path, backend="nebius", interviewer=interviewer))
    sid = before.post("/api/interview").json()["session"]
    before.post(f"/api/interview/{sid}", json={"text": "cows"})
    assert before.post(f"/api/interview/{sid}", json={"text": "on a post"}).json()["kind"] == "confirm"
    after = TestClient(create_app(tmp_path, backend="nebius", interviewer=interviewer))  # a new server process
    t = after.post(f"/api/interview/{sid}", json={"text": "yes"}).json()
    assert t["kind"] == "done" and t["spec"]["name"] == "Barn Cam"
    assert after.post("/api/interview/0123456789abcdef0123456789abcdef", json={"text": "x"}).status_code == 404


def street(tmp_path, n=8):
    """A library dataset of n photos of the default owner: (library, dataset id, photo ids)."""
    from PIL import Image

    from sim2chip import library
    lib = library.Library(tmp_path, tmp_path / "cache")
    files = []
    for i in range(n):
        f = tmp_path / "scratch" / f"p{i:02d}.jpg"
        f.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 64), (i * 9, 0, 0)).save(f, "JPEG")
        files.append(f)
    ds = lib.new_dataset("local", "street", files)
    return lib, ds, [p["id"] for p in lib.photos(ds)]



def test_a_labeling_reviewed_for_a_round_needs_no_check(app, tmp_path):
    """The check comes first while the user has reviewed fewer photos of the labeling than one round shows."""
    from sim2chip.job_service import CALIBRATION_PHOTOS
    lib, ds, ids = street(tmp_path, 20)
    lid = lib.create_labeling(ds, "Cows", "classify", [{"key": "cow"}, {"key": "empty"}])
    spec = {**GOOD, "user_data": {"raw": f"library:{ds}", "labeling": lid}}
    for pid in ids[:CALIBRATION_PHOTOS - 1]:
        lib.review_photo(lid, pid, [{"class": "cow"}])
    r = app.post("/api/jobs", data={"spec": json.dumps(spec), "calibrate": "true"}).json()
    assert r["calibrating"] and not app.launched
    lib.review_photo(lid, ids[-1], [{"class": "empty"}])
    r = app.post("/api/jobs", data={"spec": json.dumps({**spec, "name": "Barn Cam 2"}), "calibrate": "true"}).json()
    assert not r.get("calibrating") and app.launched


def test_labeling_a_dataset_puts_the_class_ids_in_english(app, tmp_path, monkeypatch):
    """The names typed on the Datasets page become English keys, and for counting the teacher's English names;
    teach starts the teacher on the new labeling."""
    from sim2chip import teacher
    lib, ds, ids = street(tmp_path, 1)
    monkeypatch.setattr(teacher, "english_phrases", lambda meaning, client=None: {"c0": ["blue bin", "blue container"]})
    r = app.post(f"/api/library/datasets/{ds}/labelings",
                 json={"task": "count", "classes": ["contenedor azul"], "teach": True}).json()
    (c,) = lib.classes(r["labeling"])
    assert (c["key"], c["description"], c["phrases"]) == ("blue_bin", "contenedor azul", ["blue bin", "blue container"])
    cls = json.loads((tmp_path / "jobs" / r["job"] / "request.json").read_text())["classes"]
    assert cls == [{**cls[0], "id": "blue_bin", "description": "contenedor azul", "names": ["blue bin", "blue container"]}]
    assert app.post(f"/api/library/datasets/{ds}/labelings", json={"task": "classify", "classes": ["a"]}).status_code == 400


def test_a_job_uses_the_labeling_it_names_or_one_with_the_same_classes(app, tmp_path):
    """Same classes: the same labeling, with its English names for the teacher. Other classes with a labeling
    named: that labeling changed to them. One dataset per model; a dataset id is never a 3D model."""
    lib, ds, ids = street(tmp_path, 2)
    count = {**GOOD, "task": "count", "classes": [{**GOOD["classes"][0], "names": ["cattle"]}],
             "user_data": {"raw": f"library:{ds}"}}
    lid = lib.create_labeling(ds, "Cows", "count", [{"key": "cow", "phrases": ["cow"]}])
    r = app.post("/api/jobs", data={"spec": json.dumps(count)}).json()
    req = json.loads((tmp_path / "jobs" / r["job"] / "request.json").read_text())
    assert req["user_data"]["labeling"] == lid and req["classes"][0]["names"] == ["cow"]
    horse = {**GOOD["classes"][0], "id": "horse", "description": "horses", "names": ["horse"]}
    more = {**count, "name": "Barn 2", "classes": [*count["classes"], horse], "user_data": {**count["user_data"], "labeling": lid}}
    r = app.post("/api/jobs", data={"spec": json.dumps(more)}).json()
    req = json.loads((tmp_path / "jobs" / r["job"] / "request.json").read_text())
    assert req["user_data"]["labeling"] == lid and [c["key"] for c in lib.classes(lid)] == ["cow", "horse"]
    other, _, _ = street(tmp_path / "o", 1)
    two = {**count, "name": "Barn 3", "user_data": {"raw": [f"library:{ds}", f"library:{other.datasets('local')[0]['id']}"]}}
    assert app.post("/api/jobs", data={"spec": json.dumps(two)}).status_code == 400
    as_model = {**count, "name": "Barn 4", "classes": [{**count["classes"][0], "assets": {"source": "library", "models": [f"library:{ds}"]},
                                                         "size_m": {"measure": "length", "range": [1.0, 2.0]}}]}
    r = app.post("/api/jobs", data={"spec": json.dumps(as_model)})
    assets = json.loads((tmp_path / "jobs" / r.json()["job"] / "request.json").read_text())["classes"][0]["assets"]
    assert r.status_code == 200 and assets["source"] == "search" and assets["queries"] == ["cattle"] and not assets["models"]


def test_a_finished_turn_can_be_picked_up_after_the_request_was_cut(app):
    """Cloudflare cuts a chat request after 100 s while the turn goes on: the web gets its answer from activity."""
    sid = app.post("/api/interview").json()["session"]
    turn = app.post(f"/api/interview/{sid}?act=t1", json={"text": "cows"}).json()
    got = app.get(f"/api/interview/{sid}/activity?act=t1").json()
    assert got["turn"] == turn and turn["text"] == "Where is the camera?"
    assert "turn" not in app.get(f"/api/interview/{sid}/activity?act=t2").json()  # another turn: still running


def test_a_model_can_be_renamed_and_deleted(app, tmp_path, monkeypatch):
    r = app.post("/api/jobs", data={"spec": json.dumps(GOOD)}).json()
    job = r["job"]
    assert app.patch(f"/api/jobs/{job}", json={"title": "Cows at night"}).json()["ok"]
    assert app.get(f"/api/jobs/{job}").json()["title"] == "Cows at night"
    assert next(m for m in app.get("/api/library").json()["models"] if m["job"] == job)["title"] == "Cows at night"
    assert app.delete(f"/api/jobs/{job}").status_code == 409  # still running
    monkeypatch.setattr(FakeProc, "poll", lambda self: 0)
    assert app.delete(f"/api/jobs/{job}").json()["ok"]
    assert not (tmp_path / "jobs" / job).exists() and app.get(f"/api/jobs/{job}").status_code == 404


def test_labelings_can_be_managed_on_the_datasets_page(app, tmp_path, monkeypatch):
    from sim2chip import teacher
    lib, ds, ids = street(tmp_path, 2)
    monkeypatch.setattr(teacher, "english_phrases", lambda meaning, client=None: {"c0": ["green bin"]})
    lid = lib.create_labeling(ds, "Bins", "count", [{"key": "blue_bin", "phrases": ["blue bin"]}])
    d = app.get(f"/api/library/datasets/{ds}").json()
    assert d["photo_ids"] == ids and d["labelings"][0]["id"] == lid
    assert app.patch(f"/api/library/labelings/{lid}", json={"name": "Street bins"}).json()["ok"]
    got = app.get(f"/api/library/labelings/{lid}").json()
    assert got["name"] == "Street bins" and [p["id"] for p in got["photos"]] == ids and "file" not in got["photos"][0]
    assert app.post(f"/api/library/labelings/{lid}/classes", json={"name": "contenedor verde"}).json() == {"labeling": lid}
    blue, green = lib.classes(lid)
    assert green["key"] == "green_bin" and green["phrases"] == ["green bin"]
    assert app.patch(f"/api/library/labelings/{lid}/classes/{blue['id']}", json={"key": "Navy bin"}).json() == {"labeling": lid}
    assert lib.classes(lid)[0]["key"] == "navy_bin"
    ok = app.put(f"/api/library/labelings/{lid}/photos/{ids[0]}", json={"marks": [{"class": "navy_bin", "x": 0.5, "y": 0.5}]})
    assert ok.json()["ok"] and app.get(f"/api/library/labelings/{lid}").json()["reviewed"] == 1
    bad = app.put(f"/api/library/labelings/{lid}/photos/{ids[1]}",
                  json={"marks": [{"class": "navy_bin", "x": 0.5, "y": 0.5}, {"class": "green_bin", "x": 0.5, "y": 0.5}]})
    assert bad.status_code == 400 and "several classes" in bad.json()["detail"]
    assert app.delete(f"/api/library/labelings/{lid}/photos/{ids[0]}").json()["ok"]
    copy = app.post(f"/api/library/labelings/{lid}/copy", json={}).json()["labeling"]
    assert app.delete(f"/api/library/labelings/{lid}/classes/{green['id']}").json() == {"labeling": lid}
    assert app.delete(f"/api/library/labelings/{lid}").json()["ok"]
    assert app.get(f"/api/library/labelings/{lid}").status_code == 404 and lib.labeling(copy)["name"] == "Street bins (2)"
    assert app.delete(f"/api/library/photos/{ids[1]}").json()["ok"] and len(lib.photos(ds)) == 1
    assert app.get(f"/api/library/photos/{ids[0]}").status_code == 200
    assert app.get(f"/api/library/labelings/{copy}", headers={}).status_code == 200


def test_an_open_chat_comes_back_in_another_tab_until_a_new_one(app):
    sid = app.post("/api/interview").json()["session"]
    view = {"msgs": [{"id": 0, "from": "me", "text": "cows"}], "text": "half typed"}
    assert app.put(f"/api/interview/{sid}/view", json={"key": "eyeforge.chat.new", "view": view}).json()["ok"]
    assert app.get("/api/interview/open", params={"key": "eyeforge.chat.new"}).json() == {"session": sid, "view": view}
    assert app.get("/api/interview/open", params={"key": "eyeforge.chat.kit.x"}).json() == {}
    app.delete("/api/interview/open", params={"key": "eyeforge.chat.new"})
    assert app.get("/api/interview/open", params={"key": "eyeforge.chat.new"}).json() == {}
    assert app.put("/api/interview/0123456789abcdef0123456789abcdef/view", json={"key": "k", "view": {}}).status_code == 404
