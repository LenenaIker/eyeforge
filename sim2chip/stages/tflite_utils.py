"""TensorFlow helpers shared by train and evaluate (imported only inside the TF runtime)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from ..hardware import Hardware


def list_dir(d: Path, labels: list[str]) -> list[tuple[Path, int]]:
    """The images of a <class>/*.png folder with their label index, in the order load_dir returns them."""
    return [(f, i) for i, c in enumerate(labels) for f in sorted((d / c).glob("*.png"))]


def load_dir(d: Path, labels: list[str], size: int):
    items = list_dir(d, labels)
    if not items:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0,), int)
    xs = [np.asarray(tf.keras.utils.load_img(f, target_size=(size, size)), dtype=np.uint8) for f, _ in items]
    return np.stack(xs), np.array([i for _, i in items])


def make_augment(hw: Hardware):
    size = hw.input_size
    lo, hi = hw.sharpness

    def augment(x, y):
        """Pull clean renders towards a cheap camera: color, framing, blur, noise and JPEG artifacts."""
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.25)
        x = tf.image.random_contrast(x, 0.6, 1.4)
        x = tf.image.random_saturation(x, 0.3, 1.3)
        x = tf.image.random_hue(x, 0.05)
        crop = tf.cast(tf.random.uniform([], 0.8, 1.0) * size, tf.int32)  # off-center region of interest
        x = tf.image.resize(tf.image.random_crop(x, [crop, crop, 3]), [size, size])
        # Equalize effective resolution across classes: otherwise small (blurry) renders of one class
        # teach the model "blurry = that class". Degrade everything to the camera's sharpness range.
        small = tf.cast(tf.random.uniform([], lo, hi) * size, tf.int32)
        x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [size, size])
        x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.04))
        x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 1), 20, 90)
        return x * 255.0, y
    return augment


def build_model(n_classes: int, hw: Hardware):
    if hw.backbone != "mobilenetv2":
        raise ValueError(f"backbone {hw.backbone} not supported")
    size = hw.input_size
    base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                             weights="imagenet")
    inp = tf.keras.Input((size, size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(base(x))
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def to_int8(model, rep_x, int8_io: bool = True) -> bytes:
    """int8 weights and activations. int8_io=False keeps float32 input (RGB 0-255) and output
    (probabilities), which is what phone apps and generic runtimes expect."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in rep_x[:300])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def train(x, y, labels: list[str], hw: Hardware, epochs: int, seed: int):
    tf.keras.utils.set_random_seed(seed)
    n = len(labels)
    idx = np.random.default_rng(seed).permutation(len(x))
    nv = len(x) // 10
    xv, yv, xtr, ytr = x[idx[:nv]], y[idx[:nv]], x[idx[nv:]], y[idx[nv:]]
    weights = len(ytr) / (n * np.maximum(np.bincount(ytr, minlength=n), 1))
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(xtr)).map(make_augment(hw),
                                                                                 tf.data.AUTOTUNE)
          .batch(64).prefetch(tf.data.AUTOTUNE))
    dv = tf.data.Dataset.from_tensor_slices((xv.astype(np.float32), yv)).batch(128)
    model, base = build_model(n, hw)
    cw = dict(enumerate(weights))
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds, validation_data=dv, epochs=5, class_weight=cw, verbose=2)
    base.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(ds, validation_data=dv, epochs=epochs, class_weight=cw, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    return to_int8(model, xtr), to_int8(model, xtr, int8_io=False), hist


def predict_int8(blob: bytes, x) -> np.ndarray:
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    preds = []
    for img in x:
        q = np.clip(np.round(img.astype(np.float32) / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q[None])
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(preds)


# Count task: a grid of class probabilities (background first), one cell per counting.STRIDE input pixels.
# Few cells hold an object, so their cross-entropy weighs OBJECT_WEIGHT times more. FOMO uses 100; on the
# pots test that made the model fire everywhere (19 found per photo for 5 real), 20 fixed it and 5 missed most.
OBJECT_WEIGHT = 20.0


def load_points(d: Path, labels: list[str], size: int):
    """Images of d/images with their points (d/points.json). Returns x, point maps (class index on each
    object's box core, or its single pixel without a box size; 0 elsewhere), the file names and the points."""
    from ..counting import TRAIN_CORE
    import json

    table = json.loads((d / "points.json").read_text())
    names = sorted(table)
    xs, maps, pts = [], [], []
    for n in names:
        p = table[n]["points"] if isinstance(table[n], dict) else table[n]
        xs.append(np.asarray(tf.keras.utils.load_img(d / "images" / n, target_size=(size, size)), dtype=np.uint8))
        m = np.zeros((size, size), np.uint8)
        for q in p:
            x, y, c = q[:3]
            cy, cx = min(int(y * size), size - 1), min(int(x * size), size - 1)
            m[cy, cx] = labels.index(c)
            if len(q) >= 5 and TRAIN_CORE > 0:
                hw, hh = q[3] * TRAIN_CORE * size / 2, q[4] * TRAIN_CORE * size / 2
                m[max(0, int(cy - hh)):int(cy + hh) + 1, max(0, int(cx - hw)):int(cx + hw) + 1] = labels.index(c)
        maps.append(m)
        pts.append([tuple(q) for q in p])
    if not xs:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0, size, size), np.uint8), [], []
    return np.stack(xs), np.stack(maps), names, pts


def count_target(p, n_classes: int):
    """Point map (size, size, 1) -> one-hot grid (size/STRIDE, size/STRIDE, n_classes), background first."""
    from ..counting import STRIDE
    onehot = tf.one_hot(tf.cast(p[..., 0], tf.int32), n_classes)
    cells = tf.nn.max_pool2d(onehot[None, ..., 1:], STRIDE, STRIDE, "VALID")[0]
    y = tf.concat([1.0 - tf.reduce_max(cells, -1, keepdims=True), cells], -1)
    return y / tf.reduce_sum(y, -1, keepdims=True)


def make_count_augment(hw: Hardware, n_classes: int):
    size = hw.input_size
    lo = max(hw.sharpness[0], 0.5)

    def augment(x, p):
        """Same camera degradations as classification, with flips and crops applied to the points too."""
        x = tf.cast(x, tf.float32) / 255.0
        p = tf.cast(p[..., None], tf.float32)
        if tf.random.uniform([]) < 0.5:
            x, p = tf.image.flip_left_right(x), tf.image.flip_left_right(p)
        crop = tf.cast(tf.random.uniform([], 0.8, 1.0) * size, tf.int32)
        oy = tf.random.uniform([], 0, size - crop + 1, tf.int32)
        ox = tf.random.uniform([], 0, size - crop + 1, tf.int32)
        x = tf.image.resize(x[oy:oy + crop, ox:ox + crop], [size, size])
        p = tf.image.resize(p[oy:oy + crop, ox:ox + crop], [size, size], method="nearest")  # upscale: no point lost
        x = tf.image.random_brightness(x, 0.25)
        x = tf.image.random_contrast(x, 0.6, 1.4)
        x = tf.image.random_saturation(x, 0.3, 1.3)
        x = tf.image.random_hue(x, 0.05)
        # Milder than for classification: small objects are what this task is about.
        small = tf.cast(tf.random.uniform([], lo, 1.0) * size, tf.int32)
        x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [size, size])
        x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.03))
        x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 1), 30, 95)
        return x * 255.0, count_target(p, n_classes)
    return augment


def build_count_model(n_classes: int, hw: Hardware):
    """MobileNetV2 cut at stride 8 (block_6_expand) plus a 1x1 head: small, int8 friendly, fits a microcontroller
    at 96 px and keeps enough resolution to separate objects that stand close."""
    if hw.backbone != "mobilenetv2":
        raise ValueError(f"backbone {hw.backbone} not supported")
    size = hw.input_size
    base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                             weights="imagenet")
    trunk = tf.keras.Model(base.input, base.get_layer("block_6_expand_relu").output)
    inp = tf.keras.Input((size, size, 3))
    x = trunk(tf.keras.applications.mobilenet_v2.preprocess_input(inp))
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    out = tf.keras.layers.Conv2D(n_classes, 1, activation="softmax")(x)
    return tf.keras.Model(inp, out), trunk


def weighted_ce(y, p):
    w = 1.0 + (OBJECT_WEIGHT - 1.0) * (1.0 - y[..., 0])
    return tf.reduce_mean(w * -tf.reduce_sum(y * tf.math.log(tf.clip_by_value(p, 1e-7, 1.0)), -1))


def train_count(x, pmap, pts, labels: list[str], hw: Hardware, epochs: int, seed: int):
    from .. import counting

    tf.keras.utils.set_random_seed(seed)
    n, size = len(labels), hw.input_size
    idx = np.random.default_rng(seed).permutation(len(x))
    nv = max(1, len(x) // 10)
    vi, ti = idx[:nv], idx[nv:]
    target = lambda xx, pp: (tf.cast(xx, tf.float32), count_target(tf.cast(pp[..., None], tf.float32), n))  # noqa: E731
    ds = (tf.data.Dataset.from_tensor_slices((x[ti], pmap[ti])).shuffle(len(ti))
          .map(make_count_augment(hw, n), tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))
    dv = tf.data.Dataset.from_tensor_slices((x[vi], pmap[vi])).map(target).batch(64)
    model, trunk = build_count_model(n, hw)
    trunk.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), weighted_ce)
    model.fit(ds, validation_data=dv, epochs=5, verbose=2)
    trunk.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-4), weighted_ce)
    h = model.fit(ds, validation_data=dv, epochs=epochs, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    probs = model.predict(x[vi].astype(np.float32), batch_size=64, verbose=0)
    hist["threshold"], hist["val_count"] = counting.best_threshold(probs, [pts[i] for i in vi], labels,
                                                                   counting.grid_size(size))
    return to_int8(model, x[ti]), to_int8(model, x[ti], int8_io=False), hist


def predict_int8_grid(blob: bytes, x) -> list[np.ndarray]:
    """Dequantized output grids of an int8 count model."""
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, z = inp["quantization"]
    so, zo = out["quantization"]
    grids = []
    for img in x:
        q = np.clip(np.round(img.astype(np.float32) / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q[None])
        it.invoke()
        grids.append((it.get_tensor(out["index"])[0].astype(np.float32) - zo) * so)
    return grids
