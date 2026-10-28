"""The web app's work without HTTP: runs, the public server's limit and the cloud budget."""
import json
import time

from sim2chip.job_service import JobService
from sim2chip.workspace import open_job


class Proc:
    def __init__(self):
        self.code, self.stopped = None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.stopped = True


def test_runs_count_against_the_public_limit_until_they_end(tmp_path):
    procs = []

    def launcher(home, job, request, backend, max_usd, start=None):
        procs.append(Proc())
        return procs[-1]

    jobs = JobService(tmp_path, launcher=launcher, public=True, max_runs=1)
    a, b = open_job(tmp_path, "a"), open_job(tmp_path, "b")
    assert not jobs.busy() and not jobs.running(a)
    jobs.start(a, a.root / "request.json")
    assert jobs.busy() and jobs.running(a) and not jobs.running(b)
    procs[0].code = 0
    assert not jobs.busy() and not jobs.running(a)
    assert not JobService(tmp_path, launcher=launcher, max_runs=0).busy()  # a private server has no limit


def test_cancel_stops_the_process_and_says_so_in_the_log(tmp_path):
    jobs = JobService(tmp_path, launcher=lambda *a, **k: Proc())
    job = open_job(tmp_path, "a")
    jobs.start(job, job.root / "request.json")
    assert jobs.cancel(job) is None and jobs.runs["a"].stopped
    assert "cancelled by the user" in (job.logs / "pipeline.log").read_text()


def test_the_cloud_budget_counts_only_on_nebius(tmp_path):
    job = open_job(tmp_path, "a")
    (job.logs / "cloud_usage.jsonl").write_text(json.dumps({"t": time.time(), "usd_estimate": 9.0}) + "\n")
    assert JobService(tmp_path, backend="local", daily_usd=10).over_budget() is None
    assert "budget is used up" in JobService(tmp_path, backend="nebius", daily_usd=10).over_budget()
    assert JobService(tmp_path, backend="nebius", daily_usd=100).over_budget() is None


def test_stopping_the_teacher_is_no_failure(tmp_path):
    """Stop while the teacher marks the check (eyeforge.app, 2026-10-28: no way to stop it): stopped, not failed,
    also when its process then ends in an error."""
    from sim2chip.job_service import CALIBRATION, LABELING, _stopped, calibration_photos
    jobs = JobService(tmp_path)
    job = open_job(tmp_path, "barn")
    (job.root / CALIBRATION).write_text(json.dumps({"state": "labeling"}))
    (job.root / LABELING).write_text(json.dumps({"dataset": "d", "state": "done"}))
    jobs.cancel(job)
    assert _stopped(job.root / CALIBRATION)
    assert json.loads((job.root / LABELING).read_text())["state"] == "done"  # over already: left as it was
    # A round shows at least 15 photos, 10 % of a large dataset (at most 40), all of a small one.
    assert [calibration_photos(n) for n in (8, 80, 150, 300, 2000)] == [8, 15, 15, 30, 40]


def test_the_teacher_starts_on_the_chats_photos_before_any_job(tmp_path, monkeypatch):
    """prefetch_draft: the photos of an upload go to the warm GPU at once (embeddings only), the proposed spec's
    with its classes (SAM 3), in a draft folder of the owner's that is no job of theirs."""
    from sim2chip.backends import nebius
    from sim2chip.spec import from_dict
    from sim2chip.tasks import finds_objects

    class Warm:
        def start(self, reason="", who=None):
            pass
    sent = []

    class Backend:
        def __init__(self, max_usd=None):
            pass

        def warm_for(self, cache, runtime):
            return {"wid": "w"}

        def prefetch_warm(self, job, cache, spec, sample, log=print):
            sent.append((sorted(p.name for p in (job.input / "photos").iterdir()), finds_objects(spec), sample,
                         spec.user_data.raw))
            return True
    monkeypatch.setattr(nebius, "NebiusBackend", Backend)
    photos = []
    for i in range(3):
        f = tmp_path / "up" / f"{i}.jpg"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(bytes([i]) * 10)
        photos.append(f)
    jobs = JobService(tmp_path, backend="nebius", warm=Warm())
    jobs.prefetch_draft("google:1", photos[:2])
    spec = from_dict({"name": "bins", "classes": [{"id": "container"}], "user_data": {"raw": ["library:x", "y"]}})
    for _ in range(200):
        if not jobs.prefetching:
            break
        time.sleep(0.01)
    jobs.prefetch_draft("google:1", photos, spec)
    for _ in range(200):
        if len(sent) == 2:
            break
        time.sleep(0.01)
    assert [(len(n), objects, sample, raw) for n, objects, sample, raw in sent] == [
        (2, False, 2, "photos"), (3, True, 3, "photos")]
    assert not (tmp_path / "jobs").exists()  # no job of the user's
