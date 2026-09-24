"""Entrena el clasificador de ROI y lo evalúa contra recortes reales (MIO-TCD).

Modelo: MobileNetV2 alpha=0.35, 96x96 RGB, preentrenado en ImageNet, cuantizado a int8 para
TFLite Micro (ESP32). La métrica que importa es la del modelo int8 sobre datos reales.

Uso:
  python spike/train.py --name synth_only
  python spike/train.py --name hybrid --few-real      # + 50 fotos reales por clase
  python spike/train.py --name real_only --real-only  # referencia: solo las 50 reales por clase
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["empty", "small_vehicle", "large_vehicle", "motorcycle", "bicycle"]
SIZE = 96


def load_dir(d: Path, limit: int | None = None):
    xs, ys = [], []
    for i, c in enumerate(CLASSES):
        files = sorted((d / c).glob("*.png"))[:limit]
        for f in files:
            xs.append(np.asarray(tf.keras.utils.load_img(f, target_size=(SIZE, SIZE)), dtype=np.uint8))
            ys.append(i)
    return np.stack(xs), np.array(ys)


def augment(x, y):
    """Aumentaciones pensadas para acercar el render limpio a una cámara barata de tráfico."""
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.25)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    x = tf.image.random_saturation(x, 0.3, 1.3)
    x = tf.image.random_hue(x, 0.05)
    # Recorte/zoom aleatorio (ROI mal centrada)
    scale = tf.random.uniform([], 0.8, 1.0)
    crop = tf.cast(scale * SIZE, tf.int32)
    x = tf.image.random_crop(x, [crop, crop, 3])
    x = tf.image.resize(x, [SIZE, SIZE])
    # Resolución efectiva igualada entre clases: en el render las motos/bicis salen pequeñas y por tanto
    # borrosas, y el modelo aprende "borroso = dos ruedas". Se degrada todo a 20-64 px efectivos.
    small = tf.cast(tf.random.uniform([], 20, 64), tf.int32)
    x = tf.image.resize(tf.image.resize(x, [small, small], antialias=True), [SIZE, SIZE])
    x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([], 0, 0.04))
    x = tf.clip_by_value(x, 0, 1)
    x = tf.image.random_jpeg_quality(x, 20, 90)
    return x * 255.0, y


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), alpha=0.35, include_top=False,
                                             weights="imagenet")
    inp = tf.keras.Input((SIZE, SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def to_int8(model, rep_x) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in rep_x[:300])
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def eval_tflite(blob: bytes, x, y):
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
    preds = np.array(preds)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return float((preds == y).mean()), cm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--crops", type=Path, default=ROOT / "data" / "crops")
    p.add_argument("--synth-dir", type=Path, default=None, help="recortes sintéticos (por defecto <crops>/synth)")
    p.add_argument("--few-real", action="store_true")
    p.add_argument("--real-only", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    if args.real_only:
        x, y = load_dir(args.crops / "real" / "few")
    else:
        x, y = load_dir(args.synth_dir or args.crops / "synth")
        if args.few_real:
            xr, yr = load_dir(args.crops / "real" / "few")
            # Se repiten las reales para que pesen algo frente a miles de sintéticas
            rep = max(1, len(x) // (len(xr) * 4))
            x, y = np.concatenate([x] + [xr] * rep), np.concatenate([y] + [yr] * rep)
    xt, yt = load_dir(args.crops / "real" / "test")
    print(f"train {len(x)} {np.bincount(y, minlength=len(CLASSES)).tolist()}  test real {len(xt)}")

    # Validación: sintético (o lo que se entrene) apartado; el test real no se toca hasta el final
    idx = np.random.default_rng(args.seed).permutation(len(x))
    nv = len(x) // 10
    xv, yv, xtr, ytr = x[idx[:nv]], y[idx[:nv]], x[idx[nv:]], y[idx[nv:]]
    weights = len(ytr) / (len(CLASSES) * np.maximum(np.bincount(ytr, minlength=len(CLASSES)), 1))

    ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr)).shuffle(len(xtr)).map(augment, tf.data.AUTOTUNE)
          .batch(64).prefetch(tf.data.AUTOTUNE))
    dv = tf.data.Dataset.from_tensor_slices((xv.astype(np.float32), yv)).batch(128)

    model, base = build_model()
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds, validation_data=dv, epochs=5, class_weight=dict(enumerate(weights)), verbose=2)
    base.trainable = True
    model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds, validation_data=dv, epochs=args.epochs, class_weight=dict(enumerate(weights)), verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    blob = to_int8(model, xtr)
    (out / "model_int8.tflite").write_bytes(blob)
    acc_float = float((np.argmax(model.predict(xt.astype(np.float32), verbose=0), 1) == yt).mean())
    acc_int8, cm = eval_tflite(blob, xt, yt)
    per_class = {c: round(float(cm[i, i] / cm[i].sum()), 3) for i, c in enumerate(CLASSES)}
    res = {"name": args.name, "train_size": int(len(x)), "real_acc_float": round(acc_float, 3),
           "real_acc_int8": round(acc_int8, 3), "per_class_int8": per_class, "confusion_int8": cm.tolist(),
           "classes": CLASSES, "tflite_kb": round(len(blob) / 1024, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
