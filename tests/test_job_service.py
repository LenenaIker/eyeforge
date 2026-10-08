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
