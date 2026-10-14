"""The GPU lock on photo2scene's heavy steps (sim2chip/photo2scene/__init__.py): two scene builds share one GPU and
take turns on the steps that need 10+ GB of its memory."""
import contextlib
import multiprocessing as mp
import time

from sim2chip import photo2scene as p2s


def hold(lock_dir, start, out):
    import os
    os.environ["SIM2CHIP_GPU_LOCK_DIR"] = str(lock_dir)
    start.wait()
    with p2s.gpu_turn(lambda m: None):
        out.put(("in", time.time()))
        time.sleep(0.5)
        out.put(("out", time.time()))


def test_two_processes_take_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    ctx = mp.get_context("fork")
    start, out = ctx.Event(), ctx.Queue()
    procs = [ctx.Process(target=hold, args=(tmp_path, start, out)) for _ in range(2)]
    for p in procs:
        p.start()
    start.set()
    for p in procs:
        p.join(10)
    events = sorted((out.get(timeout=1) for _ in range(4)), key=lambda e: e[1])
    assert [e[0] for e in events] == ["in", "out", "in", "out"]


def test_one_lock_per_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2CHIP_GPU_LOCK_DIR", str(tmp_path))
    for gpu in ("0", "1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
        with p2s.gpu_turn():
            pass
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sim2chip_gpu_0.lock", "sim2chip_gpu_1.lock"]


def test_only_heavy_steps_wait(tmp_path, monkeypatch):
    turns = []
    monkeypatch.setattr(p2s, "gpu_turn", lambda log, step: turns.append(step) or contextlib.nullcontext())
    monkeypatch.setattr(p2s.config, "env", lambda k: "/bin/true" if k in p2s.PYTHONS.values() else None)
    flow = p2s.Flow(tmp_path, log=lambda m: None)
    for step, script in (("S.1_plan", "bg_plan.py"), ("S.2_erase", "erase_qwen.py"), ("1.5a_sam3d_0", "flow_sam3d.py")):
        flow.run(step, "sam3", script, [])
    assert turns == ["S.2_erase", "1.5a_sam3d_0"]
