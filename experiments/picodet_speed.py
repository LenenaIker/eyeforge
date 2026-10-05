"""Where PicoDet-S training time goes, and what cuts it without losing F1 (2026-10-05).

Runs the train stage's detector recipe on a finished job's marks (ingest/train for fitting and validation, as
_detector splits them; ingest/test to score), one seed at a time or several at once, with the worker's
constants overridden per variant, and times each phase from the worker's output.

    python experiments/picodet_speed.py JOB VARIANT [--seeds 0 1 2] [--parallel]

VARIANT is a name in VARIANTS. Results append to runs/bench/picodet_speed.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim2chip import counting, detector  # noqa: E402

# Worker overrides: module constants of sim2chip.detector_worker, plus "steps" and "workers" (worker_num).
VARIANTS = {
    "base": {},
    "s600": {"steps": 600},
    "s300": {"steps": 300},
    "epoch50": {"MIN_STEPS_PER_EPOCH": 50},
    "epoch50_s600": {"MIN_STEPS_PER_EPOCH": 50, "steps": 600},
    "epoch50_s300": {"MIN_STEPS_PER_EPOCH": 50, "steps": 300},
    "epoch100": {"MIN_STEPS_PER_EPOCH": 100},
    "e50_s600_min3": {"MIN_STEPS_PER_EPOCH": 50, "steps": 600, "MIN_EPOCHS": 3},
    "e100_s600": {"MIN_STEPS_PER_EPOCH": 100, "steps": 600, "MIN_EPOCHS": 3},
    "e200_s600": {"MIN_STEPS_PER_EPOCH": 200, "steps": 600, "MIN_EPOCHS": 3},
    "epoch30_s300": {"MIN_STEPS_PER_EPOCH": 30, "steps": 300},
    "epoch20_s200": {"MIN_STEPS_PER_EPOCH": 20, "steps": 200},
    "epoch50_w2": {"MIN_STEPS_PER_EPOCH": 50, "workers": 2},
    "epoch50_w4": {"MIN_STEPS_PER_EPOCH": 50, "workers": 4},
}

PATCH = """
import sys, json
import sim2chip.detector_worker as w
o = json.loads(sys.argv[1])
for k, v in o.items():
    if k.isupper():
        setattr(w, k, v)
if "workers" in o:
    base = w.config
    def config(*a, **k):
        p = base(*a, **k)
        p.write_text(p.read_text() + f"worker_num: {o['workers']}\\n")
        return p
    w.config = config
run = w.run
def timed(cmd, cwd, label):
    run(cmd, cwd, label)
    print(f"done {label}", flush=True)
w.run = timed
w.main()
"""


def items_of(job: Path):
    labels = ["empty"] + [c["id"] for c in json.loads((job / "spec.json").read_text())["classes"]]
    real = [(job / v["source"], v["points"]) for _, v in
            sorted(json.loads((job / "ingest/train/points.json").read_text()).items())]
    random.Random(0).shuffle(real)
    held = max(1, round(len(real) * 0.2)) if len(real) >= 5 else 0
    val, fit = real[:held], real[held:]
    test = [(job / v["source"], v["points"]) for _, v in
            sorted(json.loads((job / "ingest/test/points.json").read_text()).items())]
    side = detector.box_side([p for _, p in real])
    items = lambda pairs: [{"path": str(f), "boxes": detector.boxes_of(p, labels, side)} for f, p in pairs]  # noqa: E731
    return labels, items(fit), items(val), val, test


def one_seed(job: Path, name: str, seed: int, size: int, work: Path) -> dict:
    labels, fit, val, val_pairs, test = items_of(job)
    o = dict(VARIANTS[name])
    steps = o.pop("steps", detector.STEPS)
    py, det = detector.setup()
    req = {"paddledet": det, "out": str(work), "size": size, "classes": labels[1:], "steps": steps,
           "train": fit, "val": val, "seed": seed}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
    marks, t0 = {}, time.time()
    with subprocess.Popen([py, "-c", PATCH, json.dumps(o)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, env=env) as proc:
        proc.stdin.write(json.dumps(req))
        proc.stdin.close()
        for line in proc.stdout:
            now = round(time.time() - t0, 1)
            if line.startswith("picodet on"):
                marks["shrunk"] = now
                print(f"[{name} s{seed}] {line.strip()}", flush=True)
            elif line.startswith("done "):
                marks[line.split()[1]] = now
            elif "failed" in line:
                print(line, flush=True)
    if proc.returncode:
        raise SystemExit(f"{name} seed {seed} failed")
    t1 = time.time()
    blob = detector.to_tflite(work / "model.onnx", size)
    marks["tflite"] = round(time.time() - t1, 1)
    grid = counting.grid_size(size)
    raw = detector.predict(blob, detector.load([f for f, _ in val_pairs], size))
    truth = [[tuple(q) for q in p] for _, p in val_pairs]
    curve = {t: counting.score([detector.decode(r, labels, t) for r in raw], truth, grid) for t in detector.THRESHOLDS}
    th = detector.pick_threshold({t: v["f1"] for t, v in curve.items()})
    raw = detector.predict(blob, detector.load([f for f, _ in test], size))
    score = counting.score([detector.decode(r, labels, th) for r in raw], [[tuple(q) for q in p] for _, p in test], grid)
    return {"variant": name, "seed": seed, "worker_s": round(t1 - t0, 1), "marks": marks, "threshold": th,
            "val_f1": curve[th]["f1"], "test_f1": score["f1"], "test_mae": score["count_mae"],
            "precision": score["precision"], "recall": score["recall"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
        run = lambda s: one_seed(a.job, a.variant, s, a.size, Path(tmp) / f"seed_{s}")  # noqa: E731
        if a.parallel:
            with ThreadPoolExecutor(len(a.seeds)) as pool:
                res = list(pool.map(run, a.seeds))
        else:
            res = [run(s) for s in a.seeds]
    total = round(time.time() - t0, 1)
    out = ROOT / "runs/bench/picodet_speed.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in res:
            r.update(job=a.job.name, total_s=total, parallel=a.parallel, n_seeds=len(a.seeds))
            f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
    print(f"total {total} s", flush=True)


if __name__ == "__main__":
    main()
