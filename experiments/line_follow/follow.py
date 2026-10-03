"""Path following as regression: where does the path ahead go, left or right of where the camera looks?

    python experiments/line_follow/follow.py PHOTOS MASKS OUT [--recipes head,finetune] [--only esp32cam:96]
    python experiments/line_follow/follow.py PHOTOS - OUT --points molmo2.json [--centre 0.5] [--gate gate.json]

Labels without a human (the plan in TODO, "SAM 3 + geometry"): the walkable masks of road_grid/sam3_masks.py,
the main path is the connected part of the mask that covers the most of the bottom middle of the photo, and its
far point is the mean x of that part in its top rows (where the path disappears). The target is that x
relative to the centre of the view, -1 at the left edge, +1 at the right.
With --points the far point is the teacher's chosen one since 2026-10-03, Molmo2 (trails_point.py --scene walk):
its x, moved by 0.5 - centre (the calibration: where the far point sits when the path goes straight on).
Molmo2 points even where there is no path, so --gate drops the photos a VLM answered 0 for (path_gate_vlm.py;
GLM-5.3-Flash chosen by the user, 2026-10-03).

Turning, without new photos: the photos are mostly taken looking along the path, so the far point sits near the
middle and "always straight" would score well. Each photo gives several square views at different horizontal
offsets (a camera turned a little left or right, as PilotNet's shifted cameras), mirrored too, so the target
spreads over [-1, 1]. Folds split by photo, int8 like on the device, two recipes as in road_grid/compare.py.
Scores: mean absolute error of the target, and the steering call (left / straight / right, straight within
+-STRAIGHT) against the baselines "always straight" and the training mean.
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
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from sim2chip import student  # noqa: E402
from sim2chip.hardware import PROFILES  # noqa: E402

FOLDS = 5
OFFSETS = 7        # views per photo, evenly spread over the photo's width
STRAIGHT = 0.2     # |target| under this is "straight"
TOP_BAND = 0.08    # far point: mean x of the path in its top rows, this share of the photo's height
AUG_COPIES = 10
STEPS = (100, 400)


def label_parts(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected parts of a boolean mask (4-neighbours), numbered from 1: flood fill, fine at a few hundred px."""
    from collections import deque
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    h, w = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if lab[sy, sx]:
            continue
        n += 1
        lab[sy, sx] = n
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    q.append((ny, nx))
    return lab, n


def runs(row: np.ndarray) -> int:
    """Separate stretches of True in a 1D array."""
    return int(np.sum(np.diff(np.concatenate([[0], row.astype(np.int8)])) == 1))


def far_point(mask: np.ndarray) -> dict | None:
    """The main path's far point (x, y in 0-1 of the photo) and how wide the path is there, or None."""
    step = max(1, mask.shape[1] // 252)
    mask = mask[::step, ::step]
    h, w = mask.shape
    lab, n = label_parts(mask)
    if not n:
        return None
    bottom = lab[int(h * 0.85):, w // 3:2 * w // 3]
    counts = np.bincount(bottom.ravel(), minlength=n + 1)
    counts[0] = 0
    if counts.max() < bottom.size * 0.1:  # nothing walkable right ahead
        return None
    path = lab == counts.argmax()
    rows = np.where(path.any(1))[0]
    top = int(np.percentile(rows, 2))
    band = path[top:top + max(2, int(h * TOP_BAND))]
    ys, xs = np.nonzero(band)
    # several separate runs in the band: a fork, or the path hidden in parts; kept, but flagged
    return {"x": float(xs.mean() / w), "y": float((top + ys.mean()) / h), "width": float(np.ptp(xs) / w),
            "runs": runs(band.any(0))}


def load_labels(photos: Path, masks: Path) -> list[dict]:
    phrases = sorted(d for d in masks.iterdir() if d.is_dir())
    out = []
    for p in sorted(photos.glob("*.jpg")):
        m = None
        for d in phrases:
            a = np.asarray(Image.open(d / f"{p.stem}.png")) > 0
            m = a if m is None else m | a
        fp = far_point(m)
        out.append({"photo": p.name, **(fp or {}), "ok": fp is not None})
    return out


def label_sheet(photos: Path, labels: list[dict], dst: Path) -> None:
    """Every photo with its far point and the line to it from the bottom middle, to check the labels by eye."""
    tiles = []
    for lab in labels:
        with Image.open(photos / lab["photo"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((320, 320))
        d = ImageDraw.Draw(im)
        if lab["ok"]:
            x, y = lab["x"] * im.width, lab["y"] * im.height
            color = (255, 60, 60) if lab["runs"] > 1 else (255, 255, 0)
            d.line([(im.width / 2, im.height), (x, y)], fill=color, width=3)
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=color, width=3)
        else:
            d.text((8, 8), "no path ahead", fill=(255, 0, 0))
        tiles.append(im)
    cols = 6
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, -(-len(tiles) // cols) * th), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(dst, quality=85)


def views(photos: Path, labels: list[dict], size: int):
    """For each labeled photo, OFFSETS square views across its width and their mirrors: (n, k, s, s, 3) and the
    targets (n, k), x of the far point from the view's centre in half view widths, clipped to [-1, 1]."""
    xs, ys = [], []
    for lab in labels:
        with Image.open(photos / lab["photo"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        side = min(w, h)
        px, vx, vy = lab["x"] * w, [], []
        for cx in np.linspace(side / 2, w - side / 2, OFFSETS):
            crop = im.crop((int(cx - side / 2), h - side, int(cx - side / 2) + side, h)).resize((size, size),
                                                                                               Image.BILINEAR)
            a = np.asarray(crop)
            t = float(np.clip((px - cx) / (side / 2), -1, 1))
            vx += [a, a[:, ::-1]]
            vy += [t, -t]
        xs.append(np.stack(vx))
        ys.append(vy)
    return np.stack(xs), np.array(ys, np.float32)


def augment(x, y, sharpness):
    """Color, the camera's blur, noise and JPEG; no flips or shifts (they change the target)."""
    lo, hi = sharpness
    size = tf.shape(x)[0]
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_brightness(x, 0.25)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    x = tf.image.random_saturation(x, 0.3, 1.3)
    x = tf.image.random_hue(x, 0.05)
    small = tf.cast(tf.random.uniform([], lo, hi) * tf.cast(size, tf.float32), tf.int32)
    x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [size, size])
    x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.04))
    x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 1), 20, 90)
    return x * 255.0, y


def build(hw, size: int):
    inp = tf.keras.Input((size, size, 3))
    if hw.backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                                 weights="imagenet")
        x = base(tf.keras.applications.mobilenet_v2.preprocess_input(inp), training=False)
    else:
        base = tf.keras.applications.EfficientNetV2B0(input_shape=(size, size, 3), include_top=False,
                                                      weights="imagenet")
        x = base(inp, training=False)
    # The far point's side is a position: keep a coarse left-to-right layout instead of averaging it away.
    m = x.shape[1]
    x = tf.keras.layers.AveragePooling2D((m, max(1, m // 3)), (m, max(1, m // 3)))(x)
    x = tf.keras.layers.Flatten()(x)
    feat = tf.keras.Model(inp, x)
    out = tf.keras.layers.Dense(1)(tf.keras.layers.Dropout(0.3)(x))
    return tf.keras.Model(inp, out), base, feat


def train(recipe: str, hw, size: int, x, y, seed: int):
    tf.keras.utils.set_random_seed(seed)
    model, base, feat = build(hw, size)
    aug = lambda a, b: augment(a, b, hw.sharpness)  # noqa: E731
    if recipe == "head":
        clean = feat.predict(x.astype(np.float32), batch_size=32, verbose=0)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).map(aug, tf.data.AUTOTUNE).batch(32)
        copies = [feat.predict(ds.map(lambda a, _: a), verbose=0) for _ in range(AUG_COPIES)]
        f = np.concatenate([clean] + copies)
        t = np.tile(y, AUG_COPIES + 1)
        # ridge regression on centred features, one scale for all (student._fit_head's int8 reason)
        mu, sd = f.mean(0), np.full(f.shape[1], f.std() + 1e-6, np.float32)
        z = (f - mu) / sd
        k = np.linalg.solve(z.T @ z + 10.0 * len(z) * 1e-3 * np.eye(z.shape[1]), z.T @ (t - t.mean()))
        norm = tf.keras.layers.Normalization(mean=mu, variance=sd ** 2)(feat.output)
        dense = tf.keras.layers.Dense(1)
        head = tf.keras.Model(feat.input, dense(norm))
        dense.set_weights([k[:, None].astype(np.float32), np.array([t.mean()], np.float32)])
        return head
    ds = (tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed).repeat()
          .map(aug, tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))
    for trainable, steps, lr in ((False, STEPS[0], 1e-3), (True, STEPS[1], 1e-4)):
        base.trainable = trainable
        model.compile(tf.keras.optimizers.Adam(lr), tf.keras.losses.Huber(0.3))
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
        res.append(float(it.get_tensor(out["index"])[0][0]))
    return np.array(res)


def side(v: np.ndarray) -> np.ndarray:
    return np.where(v < -STRAIGHT, -1, np.where(v > STRAIGHT, 1, 0))


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    """pred, y: (views,). Steering: the left / straight / right call; "wrong_side": left for right or back."""
    sp, sy = side(pred), side(y)
    return {"mae": round(float(np.abs(pred - y).mean()), 4),
            "steer_acc": round(float((sp == sy).mean()), 4),
            "wrong_side": round(float((sp * sy == -1).mean()), 4),
            "corr": round(float(np.corrcoef(pred, y)[0, 1]), 4) if pred.std() > 0 else 0.0}


VARIANTS = [("esp32cam", 96), ("generic", 224)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--recipes", default="head,finetune")
    ap.add_argument("--only", default="", help="comma list of profile:size")
    ap.add_argument("--points", type=Path, help="Molmo2 points (trails_point.py output) instead of SAM 3 masks")
    ap.add_argument("--centre", type=float, default=0.5, help="x of the far point when the path goes straight")
    ap.add_argument("--gate", type=Path, help="path_gate_vlm.py answers: photos answered 0 are left out")
    ap.add_argument("--gate-model", default="zai-org/GLM-5.3-Flash")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    if args.points:
        pts = json.loads(args.points.read_text())["x"]
        labels = [{"photo": p.name, "x": pts[p.name] + 0.5 - args.centre, "y": 0.45, "runs": 1, "ok": True}
                  if pts.get(p.name) is not None else {"photo": p.name, "ok": False}
                  for p in sorted(args.photos.glob("*.jpg"))]
    else:
        labels = load_labels(args.photos, args.masks)
    if args.gate:
        gate = json.loads(args.gate.read_text())[args.gate_model]
        labels = [lab if gate.get(lab["photo"]) != 0 else {"photo": lab["photo"], "ok": False, "gate": 0}
                  for lab in labels]
    (args.out / "labels.json").write_text(json.dumps(labels, indent=1))
    label_sheet(args.photos, labels, args.out / "labels.jpg")
    good = [lab for lab in labels if lab["ok"]]
    n = len(good)
    print(f"{n} of {len(labels)} photos with a path ahead, {sum(lab['runs'] > 1 for lab in good)} with a fork or gap "
          f"at the far end; far point x mean {np.mean([lab['x'] for lab in good]):.2f} "
          f"sd {np.std([lab['x'] for lab in good]):.2f}", flush=True)
    fold = np.random.default_rng(0).permutation(n) % FOLDS
    res_path = args.out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    only = set(filter(None, args.only.split(",")))
    for name, size in VARIANTS:
        if only and f"{name}:{size}" not in only:
            continue
        hw = PROFILES[name]
        x, y = views(args.photos, good, size)
        k_views = y.shape[1]
        if "baselines" not in results:
            zero, mean = np.zeros_like(y), np.zeros_like(y)
            for k in range(FOLDS):
                mean[fold == k] = y[fold != k].mean()
            results["baselines"] = {"straight": scores(zero.ravel(), y.ravel()),
                                    "train_mean": scores(mean.ravel(), y.ravel()),
                                    "share_straight": round(float((side(y) == 0).mean()), 4)}
            print("baselines", results["baselines"], flush=True)
        for recipe in args.recipes.split(","):
            key = f"{name}/{size}/{recipe}"
            if key in results:
                print(key, "done", flush=True)
                continue
            t0, pred, blob = time.time(), np.zeros_like(y), None
            for k in range(FOLDS):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                xtr = x[tr].reshape(-1, size, size, 3)
                model = train(recipe, hw, size, xtr, y[tr].ravel(), seed=k)
                blob = student.to_int8(model, xtr, int8_io=False)
                pred[te] = run_int8(blob, x[te].reshape(-1, size, size, 3)).reshape(len(te), k_views)
                tf.keras.backend.clear_session()
            (args.out / f"{name}_{size}_{recipe}.tflite").write_bytes(blob)
            np.save(args.out / f"{name}_{size}_{recipe}_pred.npy", pred)
            results[key] = {**scores(pred.ravel(), y.ravel()), "kb": round(len(blob) / 1024),
                            "train_min": round((time.time() - t0) / 60, 1)}
            print(key, results[key], flush=True)
            res_path.write_text(json.dumps(results, indent=1))
    res_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
