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


# Geometry, TrivialAugment style (Mueller and Hutter, 2021): each image gets one operation drawn at random, at a
# random strength up to these limits, so nothing is tuned per job. Which turns and mirrors are allowed comes from
# spec.training.orientation (spec.ORIENTATIONS). The limits are RandAugment's, a bit milder for tilt: small
# students underfit with the strong augmentation big models like.
ROTATE_DEG = 30.0  # "upright" and "fixed"; "any" adds quarter turns, so 45 covers every angle
SHEAR = 0.3        # slant: a vertical line leans up to about 17 degrees
TILT = 0.2         # perspective: one edge of the image up to 20% nearer than the opposite one


def _geometry(size: int, orientation: str | None):
    """8-parameter projective transform (output pixel -> input pixel) of one random operation about the centre."""
    u = lambda m: tf.random.uniform([], -m, m)  # noqa: E731
    one, zero = tf.constant(1.0), tf.constant(0.0)

    def mat(rows):
        return tf.convert_to_tensor([[tf.convert_to_tensor(v, tf.float32) for v in r] for r in rows])

    def rotate():
        a = u((45.0 if orientation == "any" else ROTATE_DEG) * np.pi / 180)
        return mat([[tf.cos(a), -tf.sin(a), zero], [tf.sin(a), tf.cos(a), zero], [zero, zero, one]])

    ops = [lambda: tf.eye(3),
           rotate,
           lambda: mat([[one, u(SHEAR), zero], [zero, one, zero], [zero, zero, one]]),
           lambda: mat([[one, zero, zero], [u(SHEAR), one, zero], [zero, zero, one]]),
           lambda: mat([[one, zero, zero], [zero, one, zero], [u(TILT) * 2 / size, zero, one]]),
           lambda: mat([[one, zero, zero], [zero, one, zero], [zero, u(TILT) * 2 / size, one]])]
    a = tf.switch_case(tf.random.uniform([], 0, len(ops), tf.int32), ops)
    c = (size - 1) / 2
    m = tf.constant([[1, 0, c], [0, 1, c], [0, 0, 1]], tf.float32) @ a @ \
        tf.constant([[1, 0, -c], [0, 1, -c], [0, 0, 1]], tf.float32)
    return tf.reshape(m / m[2, 2], [-1])[:8]


def _warp(img, transform):
    """Reflect at the borders: black corners would become a clue, and every operation above keeps the centre."""
    return tf.raw_ops.ImageProjectiveTransformV3(images=img[None], transforms=transform[None],
                                                 output_shape=tf.shape(img)[:2], fill_value=0.0,
                                                 interpolation="BILINEAR", fill_mode="REFLECT")[0]


def _area_scale(transform, size: int):
    """How much a warp by transform (output pixel -> input pixel) stretches the input area at every output pixel:
    a density warped with it keeps its sum. For a homography H, |det J| = |det H| / w^3, w its third row."""
    h = tf.reshape(tf.concat([transform, [1.0]], 0), [3, 3])
    r = tf.range(size, dtype=tf.float32)
    w = h[2, 0] * r[None, :] + h[2, 1] * r[:, None] + h[2, 2]
    return (tf.abs(tf.linalg.det(h)) / tf.abs(w) ** 3)[..., None]


def _orient(images: list, orientation: str | None) -> list:
    """The same random mirror and quarter turn for every image of the list (a photo and its point map)."""
    if orientation != "fixed":
        flip = tf.random.uniform([]) < 0.5
        images = [tf.cond(flip, lambda t=t: tf.image.flip_left_right(t), lambda t=t: t) for t in images]
    if orientation == "any":  # with the mirror, every turn and flip of a square
        k = tf.random.uniform([], 0, 4, tf.int32)
        images = [tf.image.rot90(t, k) for t in images]
    return images


def make_augment(hw: Hardware, orientation: str | None = None):
    size = hw.input_size
    lo, hi = hw.sharpness

    def augment(x, y):
        """Turns and tilts the photo may really have, then pull it towards a cheap camera: color, framing,
        blur, noise and JPEG artifacts."""
        x = tf.cast(x, tf.float32) / 255.0
        x = _orient([x], orientation)[0]
        x = _warp(x, _geometry(size, orientation))
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


def train(x, y, labels: list[str], hw: Hardware, epochs: int, seed: int, orientation: str | None = None):
    tf.keras.utils.set_random_seed(seed)
    n = len(labels)
    idx = np.random.default_rng(seed).permutation(len(x))
    nv = len(x) // 10
    xv, yv, xtr, ytr = x[idx[:nv]], y[idx[:nv]], x[idx[nv:]], y[idx[nv:]]
    weights = len(ytr) / (n * np.maximum(np.bincount(ytr, minlength=n), 1))
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(xtr)).map(make_augment(hw, orientation),
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


# Few photos (train.py decides): the pretrained backbone stays frozen and only the last layer learns, on features
# of each photo plus AUG_COPIES augmented copies of it. Fine-tuning the whole network on 32 lawn photos, stopped by
# 3 validation photos, scored 56% on the test; this scored 80% on the same photos (2026-09-30).
AUG_COPIES = 10
CV_FOLDS = 5


def _fit_head(f, y, n: int, seed: int):
    """Softmax regression on centred features, classes weighted by their size. Returns the feature mean and scale
    and the Dense kernel and bias. One scale for every feature, not one each: per-feature scaling blew up features
    that barely vary, finer than int8 resolves, and int8 then lost 15 to 25 points on the ESP32 model."""
    tf.keras.utils.set_random_seed(seed)
    mu, sd = f.mean(0), np.full(f.shape[1], f.std() + 1e-6, np.float32)
    w = len(y) / (n * np.maximum(np.bincount(y, minlength=n), 1))
    m = tf.keras.Sequential([tf.keras.Input((f.shape[1],)),
                             tf.keras.layers.Dense(n, activation="softmax",
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-2))])
    m.compile(tf.keras.optimizers.Adam(1e-2), "sparse_categorical_crossentropy")
    m.fit((f - mu) / sd, y, epochs=150, batch_size=256, verbose=0, class_weight=dict(enumerate(w)))
    k, b = m.layers[-1].get_weights()
    return mu, sd, k, b


def _head_predict(f, head) -> np.ndarray:
    mu, sd, k, b = head
    return ((f - mu) / sd @ k + b).argmax(1)


def _folds(y, k: int, seed: int) -> np.ndarray:
    """Stratified fold of each photo."""
    rng, fold = np.random.default_rng(seed), np.zeros(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def train_head(x, y, labels: list[str], hw: Hardware, seed: int, orientation: str | None = None):
    """Last layer only. Its score is cross-validated on the training photos (each photo scored by a head that did
    not see it or its copies): with few photos a held-out slice would be 3 photos."""
    n = len(labels)
    full, _ = build_model(n, hw)
    gap = next(layer for layer in full.layers if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D))
    feat = tf.keras.Model(full.input, gap.output)
    clean = feat.predict(x.astype(np.float32), batch_size=64, verbose=0)
    tf.keras.utils.set_random_seed(seed)
    aug = tf.data.Dataset.from_tensor_slices((x, y)).map(make_augment(hw, orientation), tf.data.AUTOTUNE).batch(64)
    copies = [feat.predict(aug.map(lambda a, _: a), verbose=0) for _ in range(AUG_COPIES)]

    def fit(idx):
        f = np.concatenate([clean[idx]] + [c[idx] for c in copies])
        return _fit_head(f, np.tile(y[idx], AUG_COPIES + 1), n, seed)

    k = min(CV_FOLDS, int(np.bincount(y, minlength=n)[np.bincount(y, minlength=n) > 0].min()))
    hist = {"recipe": "head", "photos": int(len(x)), "aug_copies": AUG_COPIES}
    if k >= 2:
        fold, pred = _folds(y, k, seed), np.zeros(len(y), int)
        for i in range(k):
            pred[fold == i] = _head_predict(clean[fold == i], fit(np.where(fold != i)[0]))
        hist["cv_accuracy"] = round(float(np.mean(pred == y)), 4)
        hist["cv_per_class"] = {labels[c]: round(float(np.mean(pred[y == c] == c)), 3) for c in np.unique(y)}
    mu, sd, kern, bias = fit(np.arange(len(x)))
    z = tf.keras.layers.Normalization(mean=mu, variance=sd ** 2)(gap.output)
    dense = tf.keras.layers.Dense(n, activation="softmax")
    model = tf.keras.Model(full.input, dense(z))
    dense.set_weights([kern, bias])
    return to_int8(model, x), to_int8(model, x, int8_io=False), hist


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
# The count model also has a density output: per cell, how many objects it holds (each marked point a Gaussian of
# DENSITY_SIGMA cells that sums to 1); the count is its sum. Points say where, the density how many. On FSC-147
# (2026-10-01, 4 classes, dataset labels) the density's total beat the number of points: geese 8.7 -> 7.1, apples
# 50 -> 37, bread rolls 28.7 -> 17.4 off per photo, strawberries a tie (10.3 -> 11.0). Points merge crowds into
# fewer blobs and split big objects into several.
DENSITY_SIGMA = 1.0
DENSITY_SCALE = 100.0  # the density loss works on density x this: one cell holds a small share of an object
DENSITY_WEIGHT = 1.0   # of the density loss next to the points loss, for the shared trunk


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
    """Point map (size, size, 1) -> the two targets of the count model: the one-hot grid (size/STRIDE,
    size/STRIDE, n_classes), background first, and the density grid (size/STRIDE, size/STRIDE, n_classes - 1)."""
    objects = tf.one_hot(tf.cast(p[..., 0], tf.int32), n_classes)[..., 1:]
    return _grid(objects), _cells(_blur(objects))


def _blur(objects):
    """Per-class point masks (size, size, k) -> density images: every point a Gaussian of DENSITY_SIGMA cells
    that sums to 1 (less at the borders, where part of it falls outside). Smooth, so it survives resampling."""
    from ..counting import STRIDE
    sigma = DENSITY_SIGMA * STRIDE
    r = int(3 * sigma)
    g = tf.exp(-tf.square(tf.range(-r, r + 1, dtype=tf.float32)) / (2 * sigma ** 2))
    g = g / tf.reduce_sum(g)
    k = tf.shape(objects)[-1]
    x = tf.nn.depthwise_conv2d(objects[None], tf.tile(tf.reshape(g, [-1, 1, 1, 1]), [1, 1, k, 1]), [1] * 4, "SAME")
    x = tf.nn.depthwise_conv2d(x, tf.tile(tf.reshape(g, [1, -1, 1, 1]), [1, 1, k, 1]), [1] * 4, "SAME")
    return x[0]


def _cells(density):
    """Density image (size, size, k) -> objects per output cell."""
    from ..counting import STRIDE
    return tf.nn.avg_pool2d(density[None], STRIDE, STRIDE, "VALID")[0] * STRIDE ** 2


def _grid(objects):
    """Per-class object masks (size, size, n_classes - 1) -> the one-hot grid of count_target."""
    from ..counting import STRIDE
    cells = tf.nn.max_pool2d(objects[None], STRIDE, STRIDE, "VALID")[0]
    y = tf.concat([1.0 - tf.reduce_max(cells, -1, keepdims=True), cells], -1)
    return y / tf.reduce_sum(y, -1, keepdims=True)


def make_count_augment(hw: Hardware, n_classes: int, orientation: str | None = None):
    size = hw.input_size
    lo = max(hw.sharpness[0], 0.5)

    def augment(x, p):
        """Same camera degradations as classification, with turns, tilts and crops applied to the points too."""
        x = tf.cast(x, tf.float32) / 255.0
        p = tf.cast(p[..., None], tf.float32)
        d = _blur(tf.one_hot(tf.cast(p[..., 0], tf.int32), n_classes)[..., 1:])
        x, p, d = _orient([x, p, d], orientation)
        crop = tf.cast(tf.random.uniform([], 0.8, 1.0) * size, tf.int32)
        oy = tf.random.uniform([], 0, size - crop + 1, tf.int32)
        ox = tf.random.uniform([], 0, size - crop + 1, tf.int32)
        x = tf.image.resize(x[oy:oy + crop, ox:ox + crop], [size, size])
        p = tf.image.resize(p[oy:oy + crop, ox:ox + crop], [size, size], method="nearest")  # upscale: no point lost
        # The density keeps its sum: stretched by size / crop each way, its values shrink by the square of it.
        zoom = tf.cast(crop, tf.float32) / size
        d = tf.image.resize(d[oy:oy + crop, ox:ox + crop], [size, size]) * zoom ** 2
        # Masks, not class indices, go through the warp: bilinear sampling never skips a one-pixel point (nearest
        # can), and reflected objects at the borders get their points like the reflected pixels.
        g = _geometry(size, orientation)
        x = _warp(x, g)
        d = _warp(d, g) * _area_scale(g, size)
        objects = tf.cast(_warp(tf.one_hot(tf.cast(p[..., 0], tf.int32), n_classes)[..., 1:], g) > 0.01, tf.float32)
        x = tf.image.random_brightness(x, 0.25)
        x = tf.image.random_contrast(x, 0.6, 1.4)
        x = tf.image.random_saturation(x, 0.3, 1.3)
        x = tf.image.random_hue(x, 0.05)
        # Milder than for classification: small objects are what this task is about.
        small = tf.cast(tf.random.uniform([], lo, 1.0) * size, tf.int32)
        x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [size, size])
        x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.03))
        x = tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 1), 30, 95)
        return x * 255.0, (_grid(objects), _cells(d))
    return augment


def build_count_model(n_classes: int, hw: Hardware):
    """MobileNetV2 cut at stride 8 (block_6_expand) plus two 1x1 heads: small, int8 friendly, fits a microcontroller
    at 96 px and keeps enough resolution to separate objects that stand close. Outputs "points" (class
    probabilities per cell, background first) and "density" (objects per cell and class, background left out)."""
    if hw.backbone != "mobilenetv2":
        raise ValueError(f"backbone {hw.backbone} not supported")
    size = hw.input_size

    def cut(name):
        base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=hw.alpha, include_top=False,
                                                 weights="imagenet")
        return tf.keras.Model(base.input, base.get_layer("block_6_expand_relu").output, name=name)

    trunk = cut("trunk")
    inp = tf.keras.Input((size, size, 3))
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = trunk(pre)
    h = tf.keras.layers.Conv2D(32, 1, activation="relu", name="hidden")(x)
    points = tf.keras.layers.Conv2D(n_classes, 1, activation="softmax", name="points")(h)
    # The density has its own trunk. Sharing one cost the points 3 to 8 F1 points (geese 0.64 -> 0.56, apples
    # 0.44 -> 0.36); a shared trunk the density could not change counted strawberries off by 21 instead of 13
    # (FSC-147, 2026-10-01). Twice the size and time, still small: 265 KB at 224 px.
    density_trunk = cut("density_trunk")
    model = tf.keras.Model(inp, [points, _density_head(density_trunk(pre), n_classes)])
    model.trunks = [trunk, density_trunk]
    return model, trunk


HEAD_LAYERS = ("hidden", "points", "density_hidden", "density_raw")


def _density_head(x, n_classes: int):
    """Objects per cell and class. The layer learns DENSITY_SCALE times the density, then a fixed layer scales it
    down: learnt at its own small size (0 to 0.15 a cell), one Adam step pushed every cell below zero and the
    ReLU never came back (all four FSC-147 classes, 2026-10-01)."""
    h = tf.keras.layers.Conv2D(32, 1, activation="relu", name="density_hidden")(x)
    h = tf.keras.layers.Conv2D(n_classes - 1, 1, activation="relu", name="density_raw")(h)
    return tf.keras.layers.Rescaling(1 / DENSITY_SCALE, name="density")(h)


def density_loss(y, p):
    """Squared error per cell plus a little of the error of each photo's whole count."""
    y, p = y * DENSITY_SCALE, p * DENSITY_SCALE
    cell = tf.reduce_mean(tf.square(y - p))
    total = tf.reduce_mean(tf.abs(tf.reduce_sum(y, [1, 2, 3]) - tf.reduce_sum(p, [1, 2, 3]))) / DENSITY_SCALE
    return cell + 0.01 * total



# Augmented copies of every photo per epoch. With one, the density stayed short of the same model trained apart
# (strawberries off by 17 per photo against 11); with ten it matched it, and the points gained a little too
# (apples F1 0.40 -> 0.44). FSC-147, 2026-10-01.
COUNT_REPEAT = 10


def weighted_ce(y, p):
    w = 1.0 + (OBJECT_WEIGHT - 1.0) * (1.0 - y[..., 0])
    return tf.reduce_mean(w * -tf.reduce_sum(y * tf.math.log(tf.clip_by_value(p, 1e-7, 1.0)), -1))


COUNT_LOSSES = {"points": weighted_ce, "density": density_loss}
COUNT_LOSS_WEIGHTS = {"points": 1.0, "density": DENSITY_WEIGHT}


def train_count(x, pmap, pts, labels: list[str], hw: Hardware, epochs: int, seed: int,
                orientation: str | None = None):
    from .. import counting

    tf.keras.utils.set_random_seed(seed)
    n, size = len(labels), hw.input_size
    idx = np.random.default_rng(seed).permutation(len(x))
    nv = max(1, len(x) // 10)
    vi, ti = idx[:nv], idx[nv:]
    target = lambda xx, pp: (tf.cast(xx, tf.float32), count_target(tf.cast(pp[..., None], tf.float32), n))  # noqa: E731
    ds = (tf.data.Dataset.from_tensor_slices((x[ti], pmap[ti])).repeat(COUNT_REPEAT).shuffle(len(ti) * COUNT_REPEAT)
          .map(make_count_augment(hw, n, orientation), tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))
    dv = tf.data.Dataset.from_tensor_slices((x[vi], pmap[vi])).map(target).batch(64)
    model, _ = build_count_model(n, hw)
    for t in model.trunks:
        t.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), COUNT_LOSSES, loss_weights=COUNT_LOSS_WEIGHTS)
    model.fit(ds, validation_data=dv, epochs=5, verbose=2)
    for t in model.trunks:
        t.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-4), COUNT_LOSSES, loss_weights=COUNT_LOSS_WEIGHTS)
    h = model.fit(ds, validation_data=dv, epochs=epochs, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    probs, dens = model.predict(x[vi].astype(np.float32), batch_size=64, verbose=0)
    truths = [pts[i] for i in vi]
    hist["threshold"], hist["val_count"] = counting.best_threshold(probs, truths, labels, counting.grid_size(size))
    hist["val_count"] = counting.score([counting.decode(g, labels, hist["threshold"]) for g in probs], truths,
                                       counting.grid_size(size), [counting.density_counts(d, labels) for d in dens])
    return to_int8(model, x[ti]), to_int8(model, x[ti], int8_io=False), hist


# Few real photos (train.py decides): the trunk stays frozen and only the 1x1 head learns, on trunk features of the
# synthetic frames and of each real photo plus AUG_COPIES augmented copies. The threshold and the score come from
# cross-validation on the real photos. Fine-tuning picked them on a 10% slice of the mix, nearly all synthetic,
# which asked for 0.6 to 0.8 where the real photos wanted 0.4: pots test F1 0.12 to 0.20 against 0.29 to 0.30 at
# the right threshold (2026-09-30).
COUNT_HEAD_EPOCHS = 60


def _count_features(feat, x, m, n: int, augment=None):
    """Trunk features (float16, they are large) and the two target grids of x (points, density), augmented when
    augment is given."""
    if augment is None:
        ds = tf.data.Dataset.from_tensor_slices((x, m)).map(
            lambda a, p: (tf.cast(a, tf.float32), count_target(tf.cast(p[..., None], tf.float32), n)))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, m)).map(augment, tf.data.AUTOTUNE)
    fs, ys, ds_ = [], [], []
    for a, (y, d) in ds.batch(64):
        fs.append(feat(a, training=False).numpy().astype(np.float16))
        ys.append(y.numpy())
        ds_.append(d.numpy())
    return np.concatenate(fs), (np.concatenate(ys), np.concatenate(ds_))


def _fit_count_head(f, y, n: int, seed: int) -> dict:
    """The model's head layers trained on trunk features; y: (points targets, density targets). Returns the
    weights of each HEAD_LAYERS layer by name."""
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(f.shape[1:])
    h = tf.keras.layers.Conv2D(32, 1, activation="relu", name="hidden")(inp)
    points = tf.keras.layers.Conv2D(n, 1, activation="softmax", name="points")(h)
    m = tf.keras.Model(inp, [points, _density_head(inp, n)])
    m.compile(tf.keras.optimizers.Adam(1e-3), COUNT_LOSSES, loss_weights=COUNT_LOSS_WEIGHTS)
    m.fit(f, list(y), epochs=COUNT_HEAD_EPOCHS, batch_size=32, shuffle=True, verbose=0)
    return {name: m.get_layer(name).get_weights() for name in HEAD_LAYERS}


def _head_grids(f, w: dict) -> np.ndarray:
    """The points output of a head (weights by layer name) on trunk features, in numpy."""
    (k1, b1), (k2, b2) = w["hidden"], w["points"]
    z = np.maximum(f.astype(np.float32) @ k1[0, 0] + b1, 0) @ k2[0, 0] + b2
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def _head_density(f, w: dict) -> np.ndarray:
    """The density output of a head (weights by layer name) on trunk features, in numpy."""
    (k1, b1), (k2, b2) = w["density_hidden"], w["density_raw"]
    return np.maximum(np.maximum(f.astype(np.float32) @ k1[0, 0] + b1, 0) @ k2[0, 0] + b2, 0) / DENSITY_SCALE


def train_count_head(xs, ms, xr, mr, pr, labels: list[str], hw: Hardware, seed: int, real_weight: int,
                     orientation: str | None = None):
    """Head only, for count. xs, ms: synthetic frames and point maps (may be empty); xr, mr, pr: real photos, maps
    and points. The real photos weigh about 1/real_weight of the synthetic set, as in fine-tuning."""
    from .. import counting

    n = len(labels)
    tf.keras.utils.set_random_seed(seed)
    model, _ = build_count_model(n, hw)
    feat = tf.keras.Model(model.input, model.get_layer("hidden").input)
    fs, ys = _count_features(feat, xs, ms, n) if len(xs) else (None, None)
    clean, yc = _count_features(feat, xr, mr, n)
    augment = make_count_augment(hw, n, orientation)
    copies = [_count_features(feat, xr, mr, n, augment) for _ in range(AUG_COPIES)]
    rep = 1 if fs is None else max(1, len(fs) // (len(xr) * (AUG_COPIES + 1) * real_weight))
    synth = ([], []) if fs is None else ([fs], [ys])

    def fit(idx):
        f = np.concatenate(synth[0] + ([clean[idx]] + [c[0][idx] for c in copies]) * rep)
        y = tuple(np.concatenate([s[j] for s in synth[1]] + ([yc[j][idx]] + [c[1][j][idx] for c in copies]) * rep)
                  for j in range(2))
        return _fit_count_head(f, y, n, seed)

    k = min(CV_FOLDS, len(xr))
    fold = np.random.default_rng(seed).permutation(len(xr)) % k
    grids = np.zeros((len(xr), *yc[0].shape[1:]), np.float32)
    dens = np.zeros((len(xr), *yc[1].shape[1:]), np.float32)
    for i in range(k):
        w = fit(np.where(fold != i)[0])
        grids[fold == i] = _head_grids(clean[fold == i], w)
        dens[fold == i] = _head_density(clean[fold == i], w)
    curve = counting.threshold_curve(list(grids), pr, labels, counting.grid_size(hw.input_size),
                                     [counting.density_counts(d, labels) for d in dens])
    threshold = counting.steady_threshold({t: s["f1"] for t, s in curve.items()})
    cv = curve[threshold]
    for name, weights in fit(np.arange(len(xr))).items():
        model.get_layer(name).set_weights(weights)
    hist = {"recipe": "head", "photos": int(len(xr)), "synthetic": int(len(xs)), "aug_copies": AUG_COPIES,
            "threshold": threshold, "val_count": cv,
            "cv_curve": {str(t): {"f1": s["f1"], "count_mae": s["count_mae"]} for t, s in curve.items()}}
    rep_x = np.concatenate([xs[:200], xr]) if len(xs) else xr
    return to_int8(model, rep_x), to_int8(model, rep_x, int8_io=False), hist


def predict_int8_count(blob: bytes, x) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Dequantized outputs of an int8 count model: the points grids and the density grids (None for models from
    before the density output). The outputs are told apart by their channels: the points grid has one more
    (the background); converters do not keep their order."""
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, outs = it.get_input_details()[0], sorted(it.get_output_details(), key=lambda o: -o["shape"][-1])
    s, z = inp["quantization"]
    grids, dens = [], []
    for img in x:
        q = np.clip(np.round(img.astype(np.float32) / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q[None])
        it.invoke()
        got = [(it.get_tensor(o["index"])[0].astype(np.float32) - o["quantization"][1]) * o["quantization"][0]
               for o in outs]
        grids.append(got[0])
        dens.append(got[1] if len(got) > 1 else None)
    return grids, (dens if len(outs) > 1 else None)


def predict_int8_grid(blob: bytes, x) -> list[np.ndarray]:
    """Dequantized points grids of an int8 count model."""
    return predict_int8_count(blob, x)[0]
