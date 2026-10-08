"""Our modules run in another Python: the request on stdin, the lines worth showing to the log, the tail on failure."""
import sys

import pytest

from sim2chip import subproc, teacher

ECHO = """import json, sys
req = json.loads(sys.stdin.read())
print("noise from a library")
print("SAM 3 on 2 photos")
print("  photo 1 of 2")
sys.exit(req.get("exit", 0))
"""


@pytest.fixture
def echo(tmp_path, monkeypatch):
    (tmp_path / "echo_worker.py").write_text(ECHO)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "echo_worker"


def test_run_module_logs_everything_and_returns_the_tail(echo, tmp_path):
    seen = []
    code, tail = subproc.run_module([sys.executable], echo, request={"exit": 3}, on_line=seen.append,
                                    log_file=tmp_path / "out.log", tail=2)
    assert code == 3 and tail == ["SAM 3 on 2 photos", "  photo 1 of 2"]
    assert len(seen) == 3 and (tmp_path / "out.log").read_text().startswith("noise from a library")


def test_python_env_puts_the_repo_first_and_can_leave_the_rest_out(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert subproc.python_env()["PYTHONPATH"].split(":") == [str(subproc.REPO), "/elsewhere"]
    assert subproc.python_env(inherit_path=False)["PYTHONPATH"] == str(subproc.REPO)
    assert subproc.python_env({"HF_TOKEN": "x"})["HF_TOKEN"] == "x"


def test_teacher_shows_only_its_progress_and_fails_with_the_tail(echo, monkeypatch):
    monkeypatch.setattr(teacher, "python", lambda: [sys.executable])
    shown = []
    teacher._worker(echo, "SAM 3", {}, log=shown.append)
    assert shown == ["  SAM 3 on 2 photos", "    photo 1 of 2"]
    with pytest.raises(teacher.TeacherError, match="exit 2"):
        teacher._worker(echo, "SAM 3", {"exit": 2}, log=shown.append)


def test_local_backend_runs_a_stage_in_its_python_and_shows_the_failure(tmp_path, capsys):
    from sim2chip.backends.local import LocalBackend
    from sim2chip.spec import from_dict
    from sim2chip.stages import BY_NAME
    from sim2chip.workspace import Cache, open_job

    job = open_job(tmp_path, "job")
    spec = {"name": "t", "classes": [{"id": "cow"}]}
    job.spec_path.write_text(from_dict(spec).to_json())
    LocalBackend().run_stage(BY_NAME["export"], job, Cache(tmp_path / "cache"), "fp")  # nothing trained yet
    out = capsys.readouterr().out
    assert "export failed (exit 1)" in out and "summary.json" in out
    assert "Traceback" in (job.logs / "export.log").read_text() and job.done("export") is None
