"""YOLO reference for experiments/count_detector.py: fine-tune a COCO-pretrained YOLO on an exported dataset and
write its val and test detections for `count_detector.py score`. Runs in an environment with ultralytics.

  python experiments/count_detector_yolo.py DATASET MODEL SIZE SEED [int8|w8a32|fp32]
  (e.g. runs/bench/count_detector/geese_C yolo11n 320 0)
Writes DATASET/preds/<MODEL>-<SIZE>-<SEED>.json. With a format, the trained model is exported as that TFLite
(int8: weights and activations, calibrated on the train photos; w8a32: int8 weights only) and its detections
go to <MODEL>-<SIZE>-<SEED>-<format>.json instead.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

data, model, size, seed = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
fmt = sys.argv[5] if len(sys.argv) > 5 else None
name = f"{model}-{size}-{seed}"
run = data / "yolo" / name
if fmt:
    tflite = YOLO(str(run / "weights" / "best.pt")).export(format="tflite", imgsz=size, data=str(data / "data.yaml"),
                                                          **{"int8": {"int8": True}, "w8a32": {"quantize": "w8a32"},
                                                             "fp32": {}}[fmt])
    net, name = YOLO(str(Path(tflite)), task="detect"), f"{name}-{fmt}"
else:
    net = YOLO(f"{model}.pt")
    net.train(data=str(data / "data.yaml"), imgsz=size, epochs=100, batch=16, seed=seed, deterministic=False,
              project=str(data / "yolo"), name=name, exist_ok=True, verbose=False, plots=False)
out = {}
for side in ("val", "test"):
    out[side] = {}
    for r in net.predict(str(data / "images" / side), imgsz=size, conf=0.05, max_det=300, verbose=False, stream=True):
        out[side][Path(r.path).name] = [[*b, c] for b, c in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist())]
(data / "preds").mkdir(exist_ok=True)
(data / "preds" / f"{name}.json").write_text(json.dumps(out))
