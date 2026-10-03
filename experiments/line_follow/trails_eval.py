"""Teacher and student on the IDSIA forest trails test (Giusti et al. 2016): is the trail to the left, straight
ahead or to the right? The paper's test numbers (3 classes): their DNN 85.2%, humans 86.5% and 82.0%, saliency
+ SVM 52.3%, Santana et al. (segment the trail, centroid of the largest part) 36.5%.

    python experiments/line_follow/trails_eval.py teacher FRAMES MASKS [--fit TRAIN_FRAMES TRAIN_MASKS]
    python experiments/line_follow/trails_eval.py student TRAIN_FRAMES TEST_FRAMES OUT [--labels TEACHER.json]

FRAMES: a folder from trails_sample.py (links and labels.json: -1 TL, 0 GS, +1 TR). MASKS: sam3_masks.py output.

Teacher: the SAM 3 trail mask, then one of three rules gives an x in the frame; x left of 0.5 - k is TL, right of
0.5 + k is TR, else GS; no trail found counts as wrong. k is chosen on --fit frames when given, and also on the
test itself (what Santana's numbers did) to show the best the rule can do.
  ahead:   the part under the camera (bottom middle) and its far point (follow.py's rule for the road photos)
  largest: the largest part and its far point
  centroid: the largest part's centroid (Santana et al.'s rule)

Student: the device models trained on the train frames' labels (the dataset's, or a teacher's with --labels),
mirrored copies with TL and TR swapped as in the paper; recipes as follow.py, 3 classes, int8.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few dataset frames are cut short

sys.path.insert(0, str(Path(__file__).parent))
from follow import label_parts  # noqa: E402

RULES = ("ahead", "largest", "centroid")


def trail_mask(masks: Path, name: str) -> np.ndarray | None:
    m = None
    for d in sorted(x for x in masks.iterdir() if x.is_dir()):
        f = d / name.replace(".jpg", ".png")
        if f.exists():
            a = np.asarray(Image.open(f)) > 0
            m = a if m is None else m | a
    return m


def rule_x(mask: np.ndarray, rule: str) -> float | None:
    step = max(1, mask.shape[1] // 252)
    mask = mask[::step, ::step]
    h, w = mask.shape
    lab, n = label_parts(mask)
    if not n:
        return None
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    if rule == "ahead":
        bottom = lab[int(h * 0.85):, w // 3:2 * w // 3]
        counts = np.bincount(bottom.ravel(), minlength=n + 1)
        counts[0] = 0
        if counts.max() < bottom.size * 0.1:
            return None
        part = lab == counts.argmax()
    else:
        if sizes.max() < mask.size * 0.005:
            return None
        part = lab == sizes.argmax()
    ys, xs = np.nonzero(part)
    if rule == "centroid":
        return float(xs.mean() / w)
    top = int(np.percentile(ys, 2))
    band = (ys >= top) & (ys < top + max(2, int(h * 0.08)))
    return float(xs[band].mean() / w)


def classify(x: np.ndarray, k: float) -> np.ndarray:
    """x may hold nan (no trail): those get 9, never right."""
    out = np.where(x < 0.5 - k, -1, np.where(x > 0.5 + k, 1, 0))
    return np.where(np.isnan(x), 9, out)


def best_k(x: np.ndarray, y: np.ndarray) -> float:
    ks = np.arange(0.0, 0.5, 0.01)
    return float(ks[int(np.argmax([(classify(x, k) == y).mean() for k in ks]))])


def teacher_xs(frames: Path, masks: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    labels = json.loads((frames / "labels.json").read_text())
    names = sorted(labels)
    y = np.array([labels[n] for n in names])
    xs = {r: [] for r in RULES}
    for n in names:
        m = trail_mask(masks, n)
        for r in RULES:
            v = rule_x(m, r) if m is not None else None
            xs[r].append(np.nan if v is None else v)
    return names, y, {r: np.array(v) for r, v in xs.items()}


def teacher(a) -> None:
    names, y, xs = teacher_xs(a.frames, a.masks)
    fit = teacher_xs(a.fit_frames, a.fit_masks) if a.fit_frames else None
    out = {}
    for r in RULES:
        x = xs[r]
        k_test = best_k(x, y)
        row = {"found": round(float((~np.isnan(x)).mean()), 3), "k_on_test": k_test,
               "acc_k_on_test": round(float((classify(x, k_test) == y).mean()), 3)}
        if fit:
            k_fit = best_k(fit[2][r], fit[1])
            pred = classify(x, k_fit)
            row |= {"k_on_train": k_fit, "acc": round(float((pred == y).mean()), 3),
                    "per_class": {c: round(float((pred[y == c] == c).mean()), 3) for c in (-1, 0, 1)},
                    "gs_vs_not": round(float(((pred == 0) == (y == 0)).mean()), 3)}
            if a.save and r == a.save_rule:
                fx = fit[2][r]
                lab = {n: int(v) for n, v in zip(fit[0], classify(fx, k_fit)) if v != 9}
                Path(a.save).write_text(json.dumps(lab))
                row["saved_train_labels"] = len(lab)
        out[r] = row
        print(r, row, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def load_frames(frames: Path, names: list[str], size: int) -> np.ndarray:
    return np.stack([np.asarray(Image.open(frames / n).convert("RGB").resize((size, size), Image.BILINEAR))
                     for n in names])


def student(a) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from follow import augment, build
    from sim2chip import student as st
    from sim2chip.hardware import PROFILES

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tr_lab = json.loads(Path(a.labels).read_text()) if a.labels else json.loads((a.train / "labels.json").read_text())
    te_lab = json.loads((a.test / "labels.json").read_text())
    tr_names, te_names = sorted(tr_lab), sorted(te_lab)
    ytr = np.array([tr_lab[n] for n in tr_names]) + 1  # 0 TL, 1 GS, 2 TR
    yte = np.array([te_lab[n] for n in te_names]) + 1
    a.out.mkdir(parents=True, exist_ok=True)
    res_path = a.out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name, size in (("esp32cam", 96), ("generic", 224)):
        hw = PROFILES[name]
        xtr, xte = load_frames(a.train, tr_names, size), load_frames(a.test, te_names, size)
        xtr = np.concatenate([xtr, xtr[:, :, ::-1]])
        ytr2 = np.concatenate([ytr, 2 - ytr])
        onehot = np.eye(3, dtype=np.float32)[ytr2]
        for recipe in a.recipes.split(","):
            key = f"{a.tag}/{name}/{size}/{recipe}"
            if key in results:
                continue
            tf.keras.utils.set_random_seed(0)
            model, base, feat = build(hw, size)
            x = feat.output
            out = tf.keras.layers.Dense(3, activation="softmax")(tf.keras.layers.Dropout(0.3)(x))
            model = tf.keras.Model(feat.input, out)
            aug = lambda p, q: augment(p, q, hw.sharpness)  # noqa: E731
            ds = (tf.data.Dataset.from_tensor_slices((xtr, onehot)).shuffle(len(xtr), seed=0).repeat()
                  .map(aug, tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))
            if recipe == "head":
                base.trainable = False
                model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy")
                model.fit(ds, steps_per_epoch=300, epochs=1, verbose=0)
            else:
                for trainable, steps, lr in ((False, 150, 1e-3), (True, 600, 1e-4)):
                    base.trainable = trainable
                    model.compile(tf.keras.optimizers.Adam(lr), "categorical_crossentropy")
                    model.fit(ds, steps_per_epoch=steps, epochs=1, verbose=0)
            blob = st.to_int8(model, xtr[::4], int8_io=False)
            it = tf.lite.Interpreter(model_content=blob, num_threads=8)
            it.allocate_tensors()
            i, o = it.get_input_details()[0], it.get_output_details()[0]
            pred = []
            for img in xte:
                it.set_tensor(i["index"], img[None].astype(np.float32))
                it.invoke()
                pred.append(int(np.argmax(it.get_tensor(o["index"])[0])))
            pred = np.array(pred)
            results[key] = {"acc": round(float((pred == yte).mean()), 3),
                            "per_class": {c: round(float((pred[yte == c + 1] == c + 1).mean()), 3) for c in (-1, 0, 1)},
                            "gs_vs_not": round(float(((pred == 1) == (yte == 1)).mean()), 3),
                            "train_frames": len(tr_names)}
            print(key, results[key], flush=True)
            res_path.write_text(json.dumps(results, indent=1))
            tf.keras.backend.clear_session()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("frames", type=Path)
    t.add_argument("masks", type=Path)
    t.add_argument("--fit", nargs=2, type=Path, metavar=("TRAIN_FRAMES", "TRAIN_MASKS"))
    t.add_argument("--out")
    t.add_argument("--save", help="write the train frames' labels by --save-rule here, for the student")
    t.add_argument("--save-rule", default="largest")
    s = sub.add_parser("student")
    s.add_argument("train", type=Path)
    s.add_argument("test", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--labels", help="teacher labels json (default: the dataset's)")
    s.add_argument("--tag", default="truth")
    s.add_argument("--recipes", default="head,finetune")
    a = ap.parse_args()
    if a.cmd == "teacher":
        a.fit_frames, a.fit_masks = a.fit if a.fit else (None, None)
        teacher(a)
    else:
        student(a)


if __name__ == "__main__":
    main()
