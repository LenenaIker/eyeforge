"""Count student architectures on FSC-147 geese: train on a job's ingested train photos, score the int8 model on
its test photos against the dataset's own points (spike/fsc147_count.py).

  SIZE=320 python spike/count_arch.py JOB VARIANT [SEED ...]     # VARIANT: s8 (current), fpn, fpn_wide
Photos are read from the job's label copies (full size), so SIZE is a real resolution, not an upscale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "spike"))

import dataclasses  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from count_decode_lib import peaks  # noqa: E402
from fsc147_count import CLASS, job_dir, truth_of  # noqa: E402
from sim2chip import counting, hardware  # noqa: E402
from sim2chip.stages import tflite_utils as tu  # noqa: E402


def build(variant: str):
    def build_count_model(n_classes: int, hw):
        size = hw.input_size
        base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                                 weights="imagenet")
        if variant == "s8":
            outs = base.get_layer("block_6_expand_relu").output
        else:
            outs = [base.get_layer("block_6_expand_relu").output, base.get_layer("block_13_expand_relu").output]
        trunk = tf.keras.Model(base.input, outs)
        inp = tf.keras.Input((size, size, 3))
        f = trunk(tf.keras.applications.mobilenet_v2.preprocess_input(inp))
        if variant == "s8":
            x = f
        else:
            width = 64 if variant == "fpn" else 128
            fine = tf.keras.layers.Conv2D(width, 1, activation="relu")(f[0])
            deep = tf.keras.layers.Conv2D(width, 1, activation="relu")(f[1])
            deep = tf.keras.layers.UpSampling2D(2, interpolation="nearest")(deep)
            x = tf.keras.layers.Add()([fine, deep])
            x = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
        out = tf.keras.layers.Conv2D(n_classes, 1, activation="softmax")(x)
        return tf.keras.Model(inp, out), trunk
    return build_count_model


def load(job, side, labels, size):
    """tu.load_points, but from the full size label copies the ingest made its 224 px images from."""
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp())
    table = json.loads((job / "ingest" / side / "points.json").read_text())
    (d / "images").mkdir()
    for n, v in table.items():
        os.symlink(job / v["source"], d / "images" / n)
    shutil.copy(job / "ingest" / side / "points.json", d / "points.json")
    return tu.load_points(d, labels, size)


def main():
    name, variant, *seeds = sys.argv[1:]
    seeds = [int(s) for s in seeds] or [0, 1, 2]
    tu.build_count_model = build(variant)
    job = job_dir(name)
    labels = ["empty", CLASS]
    sharp = float(os.environ.get("SHARP", 0.45))
    hw = dataclasses.replace(hardware.get("generic"), input_size=int(os.environ.get("SIZE", 224)),
                             sharpness=(sharp, 1.0))
    tu.OBJECT_WEIGHT = float(os.environ.get("OW", tu.OBJECT_WEIGHT))
    x, m, _, p = load(job, "train", labels, hw.input_size)
    keep = int(os.environ.get("NTRAIN", len(x)))  # learning curve: the first NTRAIN photos of a fixed shuffle
    sel = np.random.default_rng(0).permutation(len(x))[:keep]
    x, m, p = x[sel], m[sel], [p[i] for i in sel]
    xt, _, names, _ = load(job, "test", labels, hw.input_size)
    info = json.loads((job / "ingest" / "test" / "points.json").read_text())
    truth = truth_of()
    tt = [truth[info[n]["key"]] for n in names]
    grid = counting.grid_size(hw.input_size)
    res = {}
    for seed in seeds:
        blob, _, hist = tu.train_count(x, m, p, labels, hw, int(os.environ.get("EPOCHS", 25)), seed)
        grids = tu.predict_int8_grid(blob, xt)
        th = hist["threshold"]
        at = counting.score([counting.decode(g, labels, th) for g in grids], tt, grid)
        curve = {t: counting.score([counting.decode(g, labels, t) for g in grids], tt, grid)["f1"]
                 for t in counting.THRESHOLDS}
        pk = {t: counting.score([peaks(np.asarray(g), labels, t) for g in grids], tt, grid) for t in counting.THRESHOLDS}
        res[seed] = {"threshold": th, **at, "oracle": max(curve.values()), "at_0.5": curve[0.5],
                     "peak_at_th": pk[th]["f1"], "peak_mae_at_th": pk[th]["count_mae"],
                     "peak_oracle": max(v["f1"] for v in pk.values()), "kb": round(len(blob) / 1024, 1)}
        print(f"RESULT {variant} seed {seed}: {res[seed]}", flush=True)
    f1 = [r["f1"] for r in res.values()]
    print(f"SUMMARY {name} {variant}: F1 {np.mean(f1):.3f} +/- {np.std(f1):.3f}, mae "
          f"{np.mean([r['count_mae'] for r in res.values()]):.2f}, oracle "
          f"{np.mean([r['oracle'] for r in res.values()]):.3f}, at 0.5 {np.mean([r['at_0.5'] for r in res.values()]):.3f} | peak F1 "
          f"{np.mean([r['peak_at_th'] for r in res.values()]):.3f}, mae {np.mean([r['peak_mae_at_th'] for r in res.values()]):.2f}, "
          f"oracle {np.mean([r['peak_oracle'] for r in res.values()]):.3f}")


if __name__ == "__main__":
    main()
