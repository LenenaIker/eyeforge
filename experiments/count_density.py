"""Density counting on FSC-147 geese against the points model, both scored on the count per photo (dataset truth).

The density head outputs, per output cell, how many objects it holds; the count is the sum. Same trunk and size
as the points model (MobileNetV2 cut at stride 8). Targets: every point a Gaussian of SIGMA cells summing to 1.
Augmentation is done offline in numpy so the points move with the pixels exactly (flip, crop, color, blur).

  python experiments/count_density.py JOB [SEED ...]     # also scores the job's own points models (train/seed_*)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from count_arch import load  # noqa: E402
from fsc147_count import CLASS, job_dir, truth_of  # noqa: E402
from sim2chip import counting, hardware  # noqa: E402
from sim2chip import student as tu  # noqa: E402

SIZE = 224
GRID = SIZE // counting.STRIDE
SIGMA = float(os.environ.get("SIGMA", 1.0))  # cells
SCALE = 100.0  # density x SCALE in training: per-cell values are small
COPIES = int(os.environ.get("COPIES", 10))
FLOOR = float(os.environ.get("FLOOR", 0.02))  # cells below this many objects count as none (noise)
HYBRID = 15  # density count above which the hybrid trusts the density
HYBRID_POINTS = int(os.environ.get("HYBRID_POINTS", 10))  # points count above which hybrid-by-points uses density
BUCKETS = ((0, 15), (16, 30), (31, 10_000))


def density(points) -> np.ndarray:
    """(GRID, GRID) map summing to the number of points inside the image."""
    yy, xx = np.mgrid[0:GRID, 0:GRID] + 0.5
    d = np.zeros((GRID, GRID), np.float32)
    for x, y, *_ in points:
        g = np.exp(-((xx - x * GRID) ** 2 + (yy - y * GRID) ** 2) / (2 * SIGMA ** 2))
        d += g / g.sum()
    return d


def augment(img, points, rng):
    """img (SIZE, SIZE, 3) uint8, points in 0..1 -> augmented copy and its points."""
    pts = np.array([p[:2] for p in points], np.float32).reshape(-1, 2)
    if rng.random() < 0.5:
        img, pts[:, 0] = img[:, ::-1], 1 - pts[:, 0]
    c = int(rng.uniform(0.8, 1.0) * SIZE)
    oy, ox = rng.integers(0, SIZE - c + 1, 2)
    img = img[oy:oy + c, ox:ox + c]
    pts = (pts * SIZE - [ox, oy]) / c
    pts = pts[(pts >= 0).all(1) & (pts < 1).all(1)]
    x = tf.image.resize(img.astype(np.float32) / 255, [SIZE, SIZE])
    x = tf.image.adjust_brightness(x, rng.uniform(-0.25, 0.25))
    x = tf.image.adjust_contrast(x, rng.uniform(0.6, 1.4))
    x = tf.image.adjust_saturation(x, rng.uniform(0.3, 1.3))
    small = int(rng.uniform(0.5, 1.0) * SIZE)
    x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [SIZE, SIZE])
    x = tf.clip_by_value(x + rng.normal(0, rng.uniform(0, 0.03), x.shape), 0, 1)
    return (x.numpy() * 255).astype(np.uint8), [(a, b) for a, b in pts]


def build(hw):
    base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), alpha=hw.alpha, include_top=False,
                                             weights="imagenet")
    trunk = tf.keras.Model(base.input, base.get_layer("block_6_expand_relu").output)
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = trunk(tf.keras.applications.mobilenet_v2.preprocess_input(inp))
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    out = tf.keras.layers.Conv2D(1, 1, activation="relu")(x)
    return tf.keras.Model(inp, out), trunk


def loss(y, p):
    """Per-cell squared error plus the error of the whole photo's count (both in SCALE units)."""
    cell = tf.reduce_mean(tf.square(y - p))
    total = tf.reduce_mean(tf.abs(tf.reduce_sum(y, [1, 2, 3]) - tf.reduce_sum(p, [1, 2, 3]))) / SCALE
    return cell + 0.01 * total


def train(x, pts, seed):
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    nv = max(1, len(x) // 10)
    vi, ti = idx[:nv], idx[nv:]
    xa, da = [], []
    for i in ti:
        xa.append(x[i])
        da.append(density(pts[i]))
        for _ in range(COPIES):
            a, p = augment(x[i], pts[i], rng)
            xa.append(a)
            da.append(density(p))
    xa, da = np.stack(xa).astype(np.float32), np.stack(da)[..., None] * SCALE
    xv, dv = x[vi].astype(np.float32), np.stack([density(pts[i]) for i in vi])[..., None] * SCALE
    model, trunk = build(hardware.get("generic"))
    trunk.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), loss)
    model.fit(xa, da, validation_data=(xv, dv), batch_size=32, epochs=5, verbose=0)
    trunk.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-4), loss)
    model.fit(xa, da, validation_data=(xv, dv), batch_size=32, epochs=int(os.environ.get("EPOCHS", 25)), verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return tu.to_int8(model, x[ti])


def report(name, counts, truth):
    counts, truth = np.asarray(counts, float), np.asarray(truth, float)
    err = np.abs(counts - truth)
    parts = [f"mae {err.mean():5.2f}", f"rel {np.mean(err / truth):.2f}"]
    for lo, hi in BUCKETS:
        s = (truth >= lo) & (truth <= hi)
        parts.append(f"{lo}-{hi if hi < 10_000 else '+'}: {err[s].mean():5.1f} (n={s.sum()})")
    print(f"RESULT {name:28s} " + " | ".join(parts), flush=True)
    return float(err.mean())


def main():
    name, *seeds = sys.argv[1:]
    seeds = [int(s) for s in seeds] or [0, 1, 2]
    job = job_dir(name)
    labels = ["empty", CLASS]
    x, _, _, pts = load(job, "train", labels, SIZE)
    xt, _, names, _ = load(job, "test", labels, SIZE)
    info = json.loads((job / "ingest" / "test" / "points.json").read_text())
    tt = [truth_of()[info[n]["key"]] for n in names]
    true_n = [len(t) for t in tt]

    # The job's points models: counted by their points, and by summing the object probability of every cell.
    xt224 = load(job, "test", labels, 224)[0]
    summary = json.loads((job / "train" / "summary.json").read_text())
    by_points = []
    for d in sorted((job / "train").glob("seed_*")):
        grids = tu.predict_int8_grid((d / "model_int8.tflite").read_bytes(), xt224)
        th = summary[d.name.split("_")[1]].get("threshold", counting.THRESHOLD)
        by_points.append([len(counting.decode(g, labels, th)) for g in grids])
        report(f"points {d.name}", by_points[-1], true_n)
        report(f"points-sum {d.name}", [float((1 - np.asarray(g)[..., 0]).sum()) for g in grids], true_n)

    maes = []
    for seed in seeds:
        blob = train(x, pts, seed)
        grids = tu.predict_int8_grid(blob, xt)
        dens = [float(np.asarray(g).sum()) / SCALE for g in grids]
        maes.append(report(f"density seed {seed} ({len(blob) // 1024} KB)", dens, true_n))
        floor = [float(np.where(np.asarray(g) / SCALE >= FLOOR, np.asarray(g), 0).sum()) / SCALE for g in grids]
        report(f"density floor {FLOOR} seed {seed}", floor, true_n)
        # Few objects: the points model's count (it was better there); a crowd: the density's.
        pts_n = by_points[seed % len(by_points)]
        report(f"hybrid seed {seed}", [d if d > HYBRID else p for d, p in zip(dens, pts_n)], true_n)
        # The same switch on the points model's count: it undercounts crowds but rarely sees one where none is.
        report(f"hybrid-by-points seed {seed}", [d if p > HYBRID_POINTS else p for d, p in zip(dens, pts_n)], true_n)
    print(f"SUMMARY density sigma {SIGMA}: mae {np.mean(maes):.2f} +/- {np.std(maes):.2f}")


if __name__ == "__main__":
    main()
