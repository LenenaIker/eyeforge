"""The GPU's use in a cloud job (gpu_usage.py): samples of the job script summed up per step of the run."""
import json

from sim2chip import gpu_usage
from sim2chip.backends import nebius
from sim2chip.workspace import Job


def test_a_run_says_how_much_the_gpu_worked_per_step(tmp_path):
    samples = tmp_path / "gpu.log"
    # 100 s: idle while setting up (0-40), busy training (40-100); a line without a reading is skipped.
    samples.write_text("".join(f"{t} {0 if t < 1040 else 90} {500 if t < 1040 else 9000}\n" for t in range(1000, 1100, 2))
                       + "1100 \n")
    job = Job(tmp_path / "jobs" / "j")
    job.logs.mkdir(parents=True)
    recs = [{"t": 1040, "step": "cloud: set up environment", "seconds": 40, "where": "nebius"},
            {"t": 1099, "step": "train", "seconds": 59, "where": "nebius"},
            {"t": 1099, "step": "download results", "seconds": 5, "where": "local"}]
    (job.logs / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = gpu_usage.write(job.logs, 1000, 1100, samples)
    assert out["steps"]["cloud: set up environment"]["util"] < 5
    assert out["steps"]["train"] == {"util": 90, "busy": 1.0, "mem_mb": 9000, "samples": 30}
    assert "download results" not in out["steps"]
    assert out["all"]["busy"] == 0.6
    lines = []
    nebius.print_timeline(job, log=lines.append)
    assert any("train (nebius, GPU 90%)" in line for line in lines)
    assert any("busy 60% of the time, 8.8 GB at most" in line for line in lines)


def test_no_samples_no_summary(tmp_path):
    assert gpu_usage.write(tmp_path, 0, 1, tmp_path / "none.log") is None
    assert not (tmp_path / "gpu_usage.json").exists()


def test_the_job_scripts_sample_the_gpu_first_thing():
    for script in (nebius.worker_script("j", [], runtime=nebius.worker_runtime(False)), nebius.warm_script(False)):
        lines = script.splitlines()
        assert lines.index(gpu_usage.SAMPLER) < next(i for i, l in enumerate(lines) if "S2C_ENVS" in l or "kit" in l.lower())


def test_the_cpu_and_ram_are_summed_up_too(tmp_path):
    samples = tmp_path / "gpu.log"
    # CPU full while the GPU waits (unpacking, 0-20 s), then the GPU busy; the first line has no CPU reading yet,
    # one has no GPU reading.
    samples.write_text("1000 0 500 - - 2000\n"
                       + "".join(f"{t} {0 if t < 1020 else 80} 900 {95 if t < 1020 else 30} 4 {3000 + t - 1000}\n"
                                 for t in range(1002, 1040, 2))
                       + "1040 - - 50 0 3100\n")
    s = gpu_usage.read(samples)
    assert len(s) == 20 and s[0][3] is None
    w = gpu_usage.window(s, 1000, 1040)
    assert w["cpu"] == 61 and w["cpu_full"] == 0.47 and w["iowait"] == 4 and w["ram_mb"] == 3038
    assert gpu_usage.window(s, 1000, 1000).keys() == {"util", "busy", "mem_mb", "samples"}
