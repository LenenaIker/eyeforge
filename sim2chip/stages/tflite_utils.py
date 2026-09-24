"""TensorFlow helpers shared by train and evaluate (imported only inside the TF runtime)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from ..hardware import Hardware


def load_dir(d: Path, labels: list[str], size: int):
    xs, ys = [], []
    for i, c in enumerate(labels):
        for f in sorted((d / c).glob("*.png")):
            xs.append(np.asarray(tf.keras.utils.load_img(f, target_size=(size, size)), dtype=np.uint8))
            ys.append(i)
    if not xs:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0,), int)
    return np.stack(xs), np.array(ys)


def make_augment(size: int):
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
        # teach the model "blurry = that class". Degrade everything to 20-64 px effective.
        small = tf.cast(tf.random.uniform([], 20, 64), tf.int32)
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


def to_int8(model, rep_x) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in rep_x[:300])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
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
    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(xtr)).map(make_augment(hw.input_size),
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
    return to_int8(model, xtr), hist


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
