"""Classification benchmark of small backbones for microcontrollers (ranges R0-R1 of docs/hardware_makers.md) and
for phones: the few-photos recipe (frozen backbone, last layer only, student.train_head) with each backbone, scored
as the int8 model that goes to the device, on photos it never saw. Phone backbones are also scored in float, to
tell the backbone apart from its int8 conversion.

  python experiments/classify_backbones.py [lawn|miotcd|all] [REPEATS] [micro|phone]

Datasets: the user's lawn photos (data/real/grass_labels.json, 3 classes, half for training) and 4 MIO-TCD
classes (data/real/train, evaluation only, 15 photos per class for training, 100 per class for the test).
Every backbone sees the same splits, so the comparison is paired. Results go to runs/bench/classify_backbones.json.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from sim2chip import student  # noqa: E402
from sim2chip.hardware import PROFILES  # noqa: E402

# (name, family, alpha, input size, minimalistic)
CONFIGS = [
    ("mnv2-0.35-96", "mnv2", 0.35, 96, False),   # today's esp32 profile
    ("mnv2-0.35-128", "mnv2", 0.35, 128, False),
    ("mnv2-0.35-160", "mnv2", 0.35, 160, False),
    ("mnv2-0.5-96", "mnv2", 0.5, 96, False),
    ("mnv2-0.5-128", "mnv2", 0.5, 128, False),
    ("mnv2-1.0-96", "mnv2", 1.0, 96, False),
    ("mnv2-1.0-160", "mnv2", 1.0, 160, False),
    ("mnv3s-min-1.0-96", "mnv3s", 1.0, 96, True),
    ("mnv3s-min-1.0-128", "mnv3s", 1.0, 128, True),
    ("mnv3s-1.0-96", "mnv3s", 1.0, 96, False),
]
# Phones (the "generic" profile today is mnv2-1.0-224)
PHONE_CONFIGS = [
    ("mnv2-1.0-224", "mnv2", 1.0, 224, False),
    ("mnv3l-1.0-224", "mnv3l", 1.0, 224, False),
    ("effv2b0-224", "effv2b0", 1.0, 224, False),
    ("convnext-t-224", "convnext-t", 1.0, 224, False),
]
MIO_CLASSES = ["car", "bus", "motorcycle", "pedestrian"]
MIO_TRAIN, MIO_TEST = 15, 100
OUT = ROOT / "runs" / "bench" / "classify_backbones.json"


def build_model(n_classes: int, hw):
    """student.build_model with the backbone of hw.backbone."""
    size = hw.input_size
    shape = (size, size, 3)
    inp = tf.keras.Input(shape)
    x = inp  # every backbone but MobileNetV2 takes RGB 0-255 and normalizes inside the model
    if hw.backbone == "mnv2":
        base = tf.keras.applications.MobileNetV2(input_shape=shape, alpha=hw.alpha, include_top=False,
                                                 weights="imagenet")
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    elif hw.backbone.startswith("mnv3s"):
        base = tf.keras.applications.MobileNetV3Small(input_shape=shape, alpha=hw.alpha, include_top=False,
                                                      weights="imagenet", minimalistic=hw.backbone.endswith("min"))
    elif hw.backbone == "mnv3l":
        base = tf.keras.applications.MobileNetV3Large(input_shape=shape, alpha=hw.alpha, include_top=False,
                                                      weights="imagenet")
    elif hw.backbone == "effv2b0":
        base = tf.keras.applications.EfficientNetV2B0(input_shape=shape, include_top=False, weights="imagenet")
    elif hw.backbone == "convnext-t":
        base = tf.keras.applications.ConvNeXtTiny(input_shape=shape, include_top=False, weights="imagenet")
    else:
        raise ValueError(hw.backbone)
    x = tf.keras.layers.GlobalAveragePooling2D()(base(x))
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


student.build_model = build_model
_to_int8 = student.to_int8
LAST = {}


def to_int8(model, rep_x, int8_io: bool = True) -> bytes:
    """student.to_int8 that keeps the float model, to score it too."""
    LAST["model"] = model
    return _to_int8(model, rep_x, int8_io)


student.to_int8 = to_int8


def predict_int8(blob: bytes, x, ref: bool = True) -> np.ndarray:
    """student.predict_int8, by default without the XNNPACK delegate, which cannot prepare full MobileNetV3's int8
    hard-swish (the browser's LiteRT.js runs on XNNPACK too). Reference kernels are too slow for the big phone
    backbones, so those use the default ones (ref=False)."""
    kind = tf.lite.experimental.OpResolverType
    it = tf.lite.Interpreter(model_content=blob,
                             experimental_op_resolver_type=kind.BUILTIN_REF if ref else kind.AUTO)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    preds = []
    for img in x:
        it.set_tensor(inp["index"], np.clip(np.round(img.astype(np.float32) / s + z), -128, 127).astype(np.int8)[None])
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(preds)


def load(paths: list[Path], size: int) -> np.ndarray:
    return np.stack([np.asarray(tf.keras.utils.load_img(p, target_size=(size, size)), dtype=np.uint8)
                     for p in paths])


def lawn_splits(repeats: int):
    truth = json.loads((ROOT / "data/real/grass_labels.json").read_text())
    labels = sorted(set(truth.values()))
    names = sorted(truth)
    y = np.array([labels.index(truth[n]) for n in names])
    paths = [ROOT / "data/real/grass" / n for n in names]
    for r in range(repeats):
        rng, tr = np.random.default_rng(r), np.zeros(len(y), bool)
        for c in range(len(labels)):
            idx = rng.permutation(np.where(y == c)[0])
            tr[idx[:len(idx) // 2]] = True
        yield labels, paths, y, np.where(tr)[0], np.where(~tr)[0]


def mio_splits(repeats: int):
    labels = MIO_CLASSES
    pool = {c: sorted((ROOT / "data/real/train" / c).iterdir()) for c in labels}
    for r in range(repeats):
        rng, paths, y, tr, te = np.random.default_rng(100 + r), [], [], [], []
        for ci, c in enumerate(labels):
            pick = rng.choice(len(pool[c]), MIO_TRAIN + MIO_TEST, replace=False)
            for j, k in enumerate(pick):
                (tr if j < MIO_TRAIN else te).append(len(paths))
                paths.append(pool[c][k])
                y.append(ci)
        yield labels, paths, np.array(y), np.array(tr), np.array(te)


def arena_kb(blob: bytes) -> float:
    """Rough peak activation memory of the int8 model: the largest sum of one op's non-constant inputs and outputs
    (TFLite Micro's planner lands near this for a chain of layers)."""
    it = tf.lite.Interpreter(model_content=blob)
    t = {d["index"]: d for d in it.get_tensor_details()}
    ops = it._get_ops_details()
    produced = {o for op in ops for o in op["outputs"]} | {d["index"] for d in it.get_input_details()}
    nbytes = lambda i: int(np.prod(t[i]["shape"])) * np.dtype(t[i]["dtype"]).itemsize  # noqa: E731
    return max(sum(nbytes(i) for i in list(op["inputs"]) + list(op["outputs"]) if i in produced and i >= 0)
               for op in ops) / 1024


def run(dataset: str, repeats: int, results: dict, configs: list):
    splits = list((lawn_splits if dataset == "lawn" else mio_splits)(repeats))
    phone = configs is PHONE_CONFIGS
    for name, family, alpha, size, minimal in configs:
        key = f"{dataset}/{name}"
        if key in results:
            continue
        hw = replace(PROFILES["esp32cam"], input_size=size, alpha=alpha,
                     backbone=family + ("-min" if minimal else ""))
        accs, faccs, t0 = [], [], time.time()
        for r, (labels, paths, y, tr, te) in enumerate(splits):
            x = load(paths, size)
            blob, _, _ = student.train_head(x[tr], y[tr], labels, hw, seed=r)
            accs.append(float(np.mean(predict_int8(blob, x[te], ref=not phone or family == "mnv3l") == y[te])))
            if phone:
                p = LAST["model"].predict(x[te].astype(np.float32), batch_size=64, verbose=0)
                faccs.append(float(np.mean(np.argmax(p, 1) == y[te])))
            tf.keras.backend.clear_session()
        results[key] = {"acc": accs, "mean": float(np.mean(accs)), "min": float(np.min(accs)),
                        "model_kb": len(blob) / 1024, "arena_kb": arena_kb(blob), "seconds": time.time() - t0}
        extra = ""
        if phone:
            results[key] |= {"float_acc": faccs, "float_mean": float(np.mean(faccs))}
            extra = f" float {np.mean(faccs):.3f}"
        print(f"{key:28s} mean {np.mean(accs):.3f} min {np.min(accs):.3f}{extra} model {len(blob) / 1024:7.0f} KB "
              f"arena {results[key]['arena_kb']:6.0f} KB", flush=True)
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    configs = PHONE_CONFIGS if len(sys.argv) > 3 and sys.argv[3] == "phone" else CONFIGS
    for d in (["lawn", "miotcd"] if which == "all" else [which]):
        run(d, repeats, results, configs)
