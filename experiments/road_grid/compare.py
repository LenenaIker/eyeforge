"""3x3 road grid: is each ninth of the photo walkable or drivable (road) or not (noroad)? Two ways to train it:

  A "crops": each cell cut out and classified alone by a 1-output network, 9 runs per photo.
  B "grid":  the whole photo through one network with 9 outputs (one per cell), 1 run per photo.

    python experiments/road_grid/compare.py PHOTOS MASKS OUT [--rule 0.5]

Labels: SAM 3 masks of everything walkable (sam3_masks.py, union of its phrases); a cell is road when at least
`rule` of it is. Photos are squashed to a square like the camera models see them. Each variant is scored with 5
folds split by photo (never cells of one photo on both sides), int8 like on the device, with two recipes: "head"
(frozen pretrained backbone, last layer only, as train.py does under 100 photos a class) and "finetune" (the
whole network). Baselines: "prior" (each cell's most common label in training: what position alone gives).
Speed: int8 time per photo on this PC's CPU, one thread (9 runs for A), only to compare them with each other.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from sim2chip import student  # noqa: E402
from sim2chip.hardware import PROFILES  # noqa: E402

FOLDS = 5
AUG_COPIES = 10   # head recipe: augmented copies of each training image, as student.AUG_COPIES
STEPS = (100, 400)  # finetune recipe: steps with the backbone frozen, then with all of it


def load(photos: Path, masks: Path, rule: float, side: int):
    """Photos squashed to side x side, the 9 road fractions of each (row by row) and its labels."""
    phrases = sorted(d for d in masks.iterdir() if d.is_dir())
    xs, fracs = [], []
    for p in sorted(photos.glob("*.jpg")):
        m = None
        for d in phrases:
            a = np.asarray(Image.open(d / f"{p.stem}.png")) > 0
            m = a if m is None else m | a
        h, w = m.shape
        fracs.append([m[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3].mean()
                      for r in range(3) for c in range(3)])
        with Image.open(p) as im:
            xs.append(np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((side, side), Image.BILINEAR)))
    fracs = np.array(fracs, np.float32)
    return np.stack(xs), fracs, (fracs >= rule).astype(np.float32)


def cells(x: np.ndarray) -> np.ndarray:
    """(n, 3s, 3s, 3) -> (n * 9, s, s, 3), row by row."""
    n, side = len(x), x.shape[1] // 3
    return x.reshape(n, 3, side, 3, side, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n * 9, side, side, 3)


def shrink(x: np.ndarray, size: int) -> np.ndarray:
    if x.shape[1] == size:
        return x
    return tf.image.resize(x.astype(np.float32), [size, size], antialias=True).numpy().round().astype(np.uint8)


def make_augment(sharpness: tuple[float, float], grid: bool):
    """Mirror (for the grid its labels mirror too), color, the camera's blur, noise and JPEG: no shifts or turns,
    which would move objects between cells."""
    lo, hi = sharpness

    def augment(x, y):
        size = tf.shape(x)[0]
        x = tf.cast(x, tf.float32) / 255.0
        flip = tf.random.uniform([]) < 0.5
        x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
        if grid:
            y = tf.cond(flip, lambda: tf.reshape(tf.reverse(tf.reshape(y, [3, 3]), [1]), [9]), lambda: y)
        x = tf.image.random_brightness(x, 0.25)
        x = tf.image.random_contrast(x, 0.6, 1.4)
        x = tf.image.random_saturation(x, 0.3, 1.3)
        x = tf.image.random_hue(x, 0.05)
        small = tf.cast(tf.random.uniform([], lo, hi) * tf.cast(size, tf.float32), tf.int32)
        x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [size, size])
        x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.04))
        x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 1), 20, 90)
        return x * 255.0, y
    return augment


def build(hw, size: int, grid: bool):
    """Backbone, then for A: average pooling and 1 output; for B: the feature map pooled to 3x3 (each output still
    sees all of it: position and context) and 9 outputs. Returns (model, backbone, features model)."""
    inp = tf.keras.Input((size, size, 3))
    if hw.backbone == "mobilenetv2":  # as student.build_model
        base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                                 weights="imagenet")
        x = base(tf.keras.applications.mobilenet_v2.preprocess_input(inp), training=False)
    else:
        base = tf.keras.applications.EfficientNetV2B0(input_shape=(size, size, 3), include_top=False,
                                                      weights="imagenet")
        x = base(inp, training=False)
    if grid:
        m = x.shape[1]
        s = m // 3
        x = tf.keras.layers.AveragePooling2D(m - 2 * s, s)(x)
        x = tf.keras.layers.Flatten()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    feat = tf.keras.Model(inp, x)
    out = tf.keras.layers.Dense(9 if grid else 1, activation="sigmoid")(tf.keras.layers.Dropout(0.3)(x))
    return tf.keras.Model(inp, out), base, feat


def fit_head(f, y, seed: int):
    """Logistic regression per output on centred features, one scale for all (student._fit_head's reasons)."""
    tf.keras.utils.set_random_seed(seed)
    mu, sd = f.mean(0), np.full(f.shape[1], f.std() + 1e-6, np.float32)
    m = tf.keras.Sequential([tf.keras.Input((f.shape[1],)),
                             tf.keras.layers.Dense(y.shape[1], activation="sigmoid",
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-2))])
    m.compile(tf.keras.optimizers.Adam(1e-2), "binary_crossentropy")
    m.fit((f - mu) / sd, y, epochs=150, batch_size=256, verbose=0)
    return mu, sd, *m.layers[-1].get_weights()


def train(recipe: str, hw, size: int, grid: bool, x, y, seed: int):
    """A float model that takes RGB 0-255 and gives road probabilities."""
    tf.keras.utils.set_random_seed(seed)
    aug = make_augment(hw.sharpness, grid)
    model, base, feat = build(hw, size, grid)
    if recipe == "head":
        clean = feat.predict(x.astype(np.float32), batch_size=32, verbose=0)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).map(aug, tf.data.AUTOTUNE).batch(32)
        copies = [feat.predict(ds.map(lambda a, _: a), verbose=0) for _ in range(AUG_COPIES)]
        mu, sd, k, b = fit_head(np.concatenate([clean] + copies), np.tile(y, (AUG_COPIES + 1, 1)), seed)
        z = tf.keras.layers.Normalization(mean=mu, variance=sd ** 2)(feat.output)
        dense = tf.keras.layers.Dense(y.shape[1], activation="sigmoid")
        head = tf.keras.Model(feat.input, dense(z))
        dense.set_weights([k, b])
        return head
    batch = 32 if size <= 288 else 8
    ds = (tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed).repeat()
          .map(aug, tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE))
    for trainable, steps, lr in ((False, STEPS[0], 1e-3), (True, STEPS[1], 1e-4)):
        base.trainable = trainable
        model.compile(tf.keras.optimizers.Adam(lr), "binary_crossentropy")
        model.fit(ds, steps_per_epoch=steps, epochs=1, verbose=0)
    return model


def run_int8(blob: bytes, x: np.ndarray) -> np.ndarray:
    it = tf.lite.Interpreter(model_content=blob, num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = []
    for img in x:
        it.set_tensor(inp["index"], img[None].astype(np.float32))
        it.invoke()
        res.append(it.get_tensor(out["index"])[0])
    return np.array(res)


def speed(blob: bytes, size: int, runs: int) -> float:
    """Median ms of `runs` inferences, one CPU thread (the whole photo: A runs 9)."""
    it = tf.lite.Interpreter(model_content=blob, num_threads=1)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.default_rng(0).uniform(0, 255, (1, size, size, 3)).astype(np.float32)
    ts = []
    for i in range(12):
        t = time.perf_counter()
        for _ in range(runs):
            it.set_tensor(inp["index"], x)
            it.invoke()
        if i >= 2:
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def scores(pred: np.ndarray, y: np.ndarray, prior: np.ndarray, fracs: np.ndarray) -> dict:
    """pred, y, prior: (photos, 9) of 0/1. "where_prior_fails": accuracy on the cells position alone gets wrong,
    the ones that need looking. "clear_cells": on cells under 20% or over 80% road (no borderline ones)."""
    hard, clear = prior != y, (fracs < 0.2) | (fracs > 0.8)
    tp, fp, fn = [float(((pred == a) & (y == b)).sum()) for a, b in ((1, 1), (1, 0), (0, 1))]
    return {"accuracy": round(float((pred == y).mean()), 4),
            "road_f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "rows": [round(float((pred[:, r * 3:r * 3 + 3] == y[:, r * 3:r * 3 + 3]).mean()), 4) for r in range(3)],
            "photos_all_right": round(float((pred == y).all(1).mean()), 4),
            "where_prior_fails": round(float((pred[hard] == y[hard]).mean()), 4),
            "clear_cells": round(float((pred[clear] == y[clear]).mean()), 4)}


VARIANTS = [  # (profile, way, input px of the network)
    ("esp32cam", "crops", 96), ("esp32cam", "grid", 96), ("esp32cam", "grid", 288),
    ("generic", "crops", 224), ("generic", "grid", 224), ("generic", "grid", 672),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rule", type=float, default=0.5)
    ap.add_argument("--recipes", default="head,finetune")
    ap.add_argument("--only", default="", help="comma list of profile:way:size")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
    big, fracs, y = load(args.photos, args.masks, args.rule, 672)
    n = len(big)
    print(f"{n} photos, road cells {y.mean():.1%}, by row {[round(float(y[:, r*3:r*3+3].mean()), 2) for r in range(3)]}",
          flush=True)
    fold = np.random.default_rng(0).permutation(n) % FOLDS
    (args.out / "labels.json").write_text(json.dumps(
        {"rule": args.rule, "photos": [p.name for p in sorted(args.photos.glob("*.jpg"))],
         "fractions": fracs.round(3).tolist(), "fold": fold.tolist()}))
    res_path = args.out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    prior = np.zeros_like(y)
    for k in range(FOLDS):
        prior[fold == k] = (y[fold != k].mean(0) >= 0.5).astype(np.float32)
    results["prior"] = scores(prior, y, prior, fracs)
    print("prior", results["prior"], flush=True)

    only = set(filter(None, args.only.split(",")))
    for name, way, size in VARIANTS:
        if only and f"{name}:{way}:{size}" not in only:
            continue
        hw = PROFILES[name]
        grid = way == "grid"
        cell = size if not grid else None
        x = shrink(big, size * 3 if not grid else size)
        for recipe in args.recipes.split(","):
            key = f"{name}/{way}/{size}/{recipe}"
            if key in results:
                print(key, "done", flush=True)
                continue
            t0, pred, blob = time.time(), np.zeros_like(y), None
            for k in range(FOLDS):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                if grid:
                    xtr, ytr, xte = x[tr], y[tr], x[te]
                else:
                    xtr, ytr, xte = cells(x[tr]), y[tr].reshape(-1, 1), cells(x[te])
                model = train(recipe, hw, size, grid, xtr, ytr, seed=k)
                blob = student.to_int8(model, xtr, int8_io=False)
                p = run_int8(blob, xte)
                pred[te] = (p.reshape(len(te), 9) >= 0.5).astype(np.float32)
                tf.keras.backend.clear_session()
            (args.out / f"{name}_{way}_{size}_{recipe}.tflite").write_bytes(blob)
            results[key] = {**scores(pred, y, prior, fracs), "kb": round(len(blob) / 1024),
                            "ms_per_photo_pc_1thread": round(speed(blob, cell or size, 1 if grid else 9), 1),
                            "train_min": round((time.time() - t0) / 60, 1)}
            print(key, results[key], flush=True)
            res_path.write_text(json.dumps(results, indent=1))
    res_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
