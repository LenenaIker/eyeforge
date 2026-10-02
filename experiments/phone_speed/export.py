"""Models for the phone speed test (experiments/phone_speed/index.html): each phone backbone of
experiments/classify_backbones.py with a 3-class head, as the formats the browser runtimes take. Only speed is
measured, so the head is untrained; the int8 models are calibrated on the lawn photos.

  PYTHONPATH=<dir with tf2onnx, onnx, onnxruntime> python experiments/phone_speed/export.py

Writes runs/bench/phone_speed/public/models/<backbone>-<format>.{tflite,onnx} and the LiteRT.js runtime files.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import onnx  # noqa: E402
import tensorflow as tf  # noqa: E402
import tf2onnx  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static  # noqa: E402
from onnxruntime.transformers.float16 import convert_float_to_float16  # noqa: E402

import classify_backbones as cb  # noqa: E402
from sim2chip.stages.export import fixed_batch  # noqa: E402

BACKBONES = ["mnv2", "effv2b0"]
SIZE = 224
PUBLIC = ROOT / "runs" / "bench" / "phone_speed" / "public"
OUT = PUBLIC / "models"


def tflite(model, fp16: bool) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


class Calib(CalibrationDataReader):
    def __init__(self, name: str, x):
        self.it = iter({name: img[None].astype(np.float32)} for img in x)

    def get_next(self):
        return next(self.it, None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = json.loads((ROOT / "data/real/grass_labels.json").read_text())
    x = cb.load([ROOT / "data/real/grass" / n for n in sorted(truth)][:64], SIZE)
    files = {}
    for b in BACKBONES:
        hw = replace(cb.PROFILES["esp32cam"], input_size=SIZE, alpha=1.0, backbone=b)
        model, _ = cb.build_model(3, hw)
        blobs = {"int8": cb.student.to_int8(model, x, int8_io=False), "fp16": tflite(model, True),
                 "fp32": tflite(model, False)}
        for k, blob in blobs.items():
            (OUT / f"{b}-{k}.tflite").write_bytes(fixed_batch(blob))  # as the app ships it
        fp32 = OUT / f"{b}-fp32.onnx"
        proto, _ = tf2onnx.convert.from_tflite(str(OUT / f"{b}-fp32.tflite"), opset=17, output_path=str(fp32))
        onnx.save(convert_float_to_float16(proto, keep_io_types=True), str(OUT / f"{b}-fp16.onnx"))
        quantize_static(str(fp32), str(OUT / f"{b}-int8.onnx"), Calib(proto.graph.input[0].name, x),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, per_channel=True)
        tf.keras.backend.clear_session()
    for f in sorted(OUT.iterdir()):
        files[f.name] = round(f.stat().st_size / 2 ** 20, 1)
        print(f"{f.name:24s} {files[f.name]:6.1f} MB")
    (PUBLIC / "models.json").write_text(json.dumps(files, indent=1))
    lit, dst = ROOT / "web/node_modules/@litertjs/core/wasm", PUBLIC / "litert"
    dst.mkdir(exist_ok=True)
    for f in lit.iterdir():
        shutil.copy(f, dst / f.name)


if __name__ == "__main__":
    main()
