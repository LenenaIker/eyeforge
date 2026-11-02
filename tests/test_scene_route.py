"""The scene route's guards (experiments/worldstereo/wsroute/steps.py, sim2chip/scene_job.py, sim2chip/locks.py): a
paid step is done only with every leg back, an empty next-best-view plan pays nothing, a step that ends without its
output stops the route, missing interpreters stop it before the first step, and a scene whose job is gone fails."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "worldstereo"))
from wsroute import steps  # noqa: E402
from wsroute.paths import Case  # noqa: E402

from sim2chip import locks, scene_job  # noqa: E402


def case_in(tmp_path) -> Case:
    case = Case("t")
    case.dir = tmp_path / "t"
    return case


def leg(case: Case, name: str, generated: bool):
    (case.scene / name).mkdir(parents=True)
    (case.scene / f"{name}_render.jpg").write_bytes(b"x")  # next to the legs, as walks.prep leaves them
    if generated:
        case.leg_video(name).parent.mkdir(parents=True)
        case.leg_video(name).write_bytes(b"x")


def test_walk_legs_are_done_only_with_every_leg_back(tmp_path):
    case = case_in(tmp_path)
    walks, planned = steps.legs_done(case, "nba_", False), steps.legs_done(case, "nba_", True)
    assert not walks()
    leg(case, "reconstruct_0/traj0", True)
    leg(case, "reconstruct_0/traj1", False)  # a job cut short
    assert not walks()
    case.leg_video("reconstruct_0/traj1").parent.mkdir(parents=True)
    case.leg_video("reconstruct_0/traj1").write_bytes(b"x")
    assert walks() and not planned()
    leg(case, "nba_00/traj0", True)
    assert walks() and planned()


def test_an_empty_plan_needs_no_staging_and_no_paid_run(tmp_path):
    case = case_in(tmp_path)
    assert not steps.nothing_planned(case, "nba_")
    case.nbv("G2_base").mkdir(parents=True)
    (case.nbv("G2_base") / "plan.json").write_text(json.dumps({"legs": []}))
    assert steps.nothing_planned(case, "nba_")
    (case.nbv("G2_base") / "plan.json").write_text(json.dumps({"legs": [{"name": "nba_00"}]}))
    assert not steps.nothing_planned(case, "nba_")


def test_a_step_without_its_output_stops_the_route(tmp_path, monkeypatch):
    ran = []
    fake = [steps.Step("a", "here", [], lambda: False), steps.Step("b", "here", [], lambda: False)]
    monkeypatch.setattr(steps, "recipe", lambda cfg: fake)
    monkeypatch.setattr(steps, "preflight", lambda todo: None)
    monkeypatch.setattr(steps.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    monkeypatch.setattr(steps, "command", lambda step: [step.name])
    cfg = type("Cfg", (), {"case": "t", "photo": "", "pano": "p.jpg"})()
    with pytest.raises(SystemExit, match="a ended without its output"):
        steps.run(cfg, "b")
    assert ran == [["a"]]


def test_missing_interpreters_stop_the_route_before_it_starts(monkeypatch):
    monkeypatch.setattr(steps, "env", lambda name: None)
    todo = [steps.Step("masks", "sam3", [], lambda: False), steps.Step("walk_legs", "cloud", [], lambda: False)]
    with pytest.raises(SystemExit) as e:
        steps.preflight(todo)
    assert "SIM2CHIP_SAM3_PYTHON (masks)" in str(e.value) and "SIM2CHIP_WS_PROJECT (walk_legs)" in str(e.value)


def test_cloud_command_quotes_its_arguments():
    cmd = steps.command(steps.Step("walk_legs", "cloud", ["cloud", "--case", "a b"], lambda: False))
    assert "'a b'" in cmd[-1]


def test_a_lock_waits_for_the_other_holder(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2CHIP_LOCKS", str(tmp_path))
    order, said = [], []

    def other():
        with locks.hold("x", "other"):
            order.append("other in")
            time.sleep(0.3)
            order.append("other out")
    t = threading.Thread(target=other)
    t.start()
    time.sleep(0.1)
    with locks.hold("x", "me", log=said.append):
        order.append("me in")
    t.join()
    assert order == ["other in", "other out", "me in"]
    assert said and "other" in said[0]


def test_a_photo_without_focal_says_why(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    (tmp_path / "b.heic").write_bytes(b"not an image")
    assert "no focal length" in scene_job.no_focal(tmp_path / "a.jpg")
    assert "cannot be opened" in scene_job.no_focal(tmp_path / "b.heic")


def test_a_scene_whose_job_is_gone_is_stale(tmp_path):
    f = scene_job.progress_file(tmp_path, "s")
    f.parent.mkdir(parents=True)
    assert scene_job.stale(tmp_path, "s")  # queued in a server that stopped: no progress yet
    with subprocess.Popen(["sleep", "30"]) as other:  # the CLI's job: still running
        f.write_text(json.dumps({"status": "building", "pid": other.pid}))
        assert not scene_job.stale(tmp_path, "s")
        other.kill()
    f.write_text(json.dumps({"status": "building", "pid": 2 ** 22 + 12345}))
    assert scene_job.stale(tmp_path, "s")
    f.write_text(json.dumps({"status": "building", "pid": os.getpid()}))  # in a container the old server had our pid
    assert scene_job.stale(tmp_path, "s")
    f.write_text(json.dumps({"status": "failed", "pid": os.getpid()}))  # an older run's file
    assert scene_job.stale(tmp_path, "s")


def test_a_vm_left_on_is_only_a_gone_jobs(tmp_path):
    mark = scene_job.vm_mark(tmp_path)
    mark.parent.mkdir(parents=True)
    assert not scene_job.vm_left_on(tmp_path)  # no job has marked it
    mark.write_text(str(2 ** 22 + 12345))
    assert scene_job.vm_left_on(tmp_path)
    mark.write_text(str(os.getpid()))  # the server before, in a container
    assert scene_job.vm_left_on(tmp_path)
    with subprocess.Popen(["sleep", "30"]) as other:  # the CLI's job on the VM: not to be cut
        mark.write_text(str(other.pid))
        assert not scene_job.vm_left_on(tmp_path)
        other.kill()
    mark.write_text("")  # killed while writing it
    assert scene_job.vm_left_on(tmp_path)


def test_vm_off_asks_again_until_nebius_agrees(tmp_path, monkeypatch):
    mark = scene_job.vm_mark(tmp_path)
    mark.parent.mkdir(parents=True)
    mark.write_text("1")
    answers = [subprocess.CompletedProcess([], 1, "", "nebius is down"), OSError("no python"),
               subprocess.CompletedProcess([], 0, "stopped", "")]

    def fake_run(*a, **kw):
        assert mark.exists()  # kept until the VM is known to be off
        got = answers.pop(0)
        if isinstance(got, Exception):
            raise got
        return got
    monkeypatch.setattr(scene_job.subprocess, "run", fake_run)
    scene_job.vm_off(tmp_path, wait=0)
    assert answers == [] and not mark.exists()


@pytest.fixture
def vm_module(monkeypatch):
    """wsroute/vm.py on a fake Nebius: the states `get` answers in turn, the stops asked for."""
    from wsroute import vm
    stops = []
    monkeypatch.setattr(vm.time, "sleep", lambda s: None)
    monkeypatch.setattr(vm, "neb", lambda *a, check=True: stops.append(a) or "")

    def answers(*states):
        left = list(states)

        def get(kind, name):
            state = left.pop(0)
            return state and {"metadata": {"id": "vm1"}, "status": {"state": state}}
        monkeypatch.setattr(vm, "get", get)
        return left
    return vm, stops, answers


def test_stop_is_done_only_when_nebius_says_so(vm_module):
    vm, stops, answers = vm_module
    answers("RUNNING", "STOPPING")
    vm.stop()
    assert len(stops) == 1
    stops.clear()
    answers("STOPPED")
    vm.stop()
    assert stops == []  # already off


def test_stop_tries_again_and_fails_loudly(vm_module):
    vm, stops, answers = vm_module
    answers(None, "RUNNING", "RUNNING", "RUNNING", "STOPPED")  # Nebius did not answer; the first stop did nothing
    vm.stop()
    assert len(stops) == 2
    answers(None, None, None)
    with pytest.raises(SystemExit, match="could not stop"):
        vm.stop()
