"""Count with a pretrained detector (hardware profile counter "picodet_s"): PicoDet-S from PaddleDetection
(Apache 2.0), fine-tuned on the job's boxes. On FSC-147 it scored points F1 0.83-0.91 against 0.41-0.80 for the
grid head (docs/hardware_makers.md, 2026-10-02); it does not fit a microcontroller, which keep the grid head.

Training runs in its own Python (detector_worker.py): SIM2CHIP_PICODET_PYTHON, a Python with PaddlePaddle, and
SIM2CHIP_PADDLEDET, a PaddleDetection checkout (release 2.8.1). Its ONNX becomes a TFLite here (onnx2tf) that
takes RGB 0-255 like the classifiers, normalization inside, with float16 weights (full int8 lost 4 points of F1
on YOLO11n: boxes and scores share no good int8 scale). Outputs, before NMS: the boxes of every anchor
(1, anchors, 4) as x0, y0, x1, y1 in input pixels and their scores (1, classes, anchors), classes in labels
order without the background. decode() keeps the boxes over a threshold, removes overlaps per class and
returns their centres; web/src/live/boxes.ts does the same for the live test.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

NMS_IOU = 0.6        # PicoDet's own: a box overlapping a better one of its class more than this is the same object
MAX_OBJECTS = 1000   # per photo (FSC-147 crowds go to several hundred)
THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)  # candidates; validation picks one
PREFERRED = 0.4      # among thresholds that score the same, the closest to this (the best on FSC-147)
STEPS = 600          # about how many training steps, whatever the number of photos. On FSC-147 geese (2026-10-05,
                     # 3 seeds, test F1): 900 0.713, 600 0.714, 300 0.696, 200 0.686. SIM2CHIP_PICODET_STEPS
                     # overrides it (to compare budgets on a dataset)
DEFAULT_BOX = 0.1    # side of a point's box in 0..1 when no marked object in the job has a size
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class DetectorError(RuntimeError):
    pass


def setup() -> tuple[str, str] | None:
    """(PaddlePaddle Python, PaddleDetection checkout), or None when either is missing here."""
    from .config import env
    py, det = env("SIM2CHIP_PICODET_PYTHON"), env("SIM2CHIP_PADDLEDET")
    if py and det and Path(py).exists() and (Path(det) / "tools" / "train.py").exists():
        return py, det
    return None


def box_side(tables: list[list]) -> tuple[float, float]:
    """Median width and height of the marked objects that have a size (the teacher's), for points without one
    (added by the user in the review)."""
    sized = [(q[3], q[4]) for pts in tables for q in pts if len(q) >= 5 and q[3] > 0 and q[4] > 0]
    if not sized:
        return DEFAULT_BOX, DEFAULT_BOX
    return statistics.median(w for w, _ in sized), statistics.median(h for _, h in sized)


def boxes_of(points: list, labels: list[str], side: tuple[float, float]) -> list[list]:
    """Points [x, y, class(, w, h)] in 0..1 -> boxes [x0, y0, x1, y1, class index without the background]. A point
    without a size takes the median of its own photo's sized points, else `side`."""
    sized = [(q[3], q[4]) for q in points if len(q) >= 5 and q[3] > 0 and q[4] > 0]
    own = (statistics.median(w for w, _ in sized), statistics.median(h for _, h in sized)) if sized else side
    out = []
    for q in points:
        x, y, c = q[:3]
        w, h = (q[3], q[4]) if len(q) >= 5 and q[3] > 0 and q[4] > 0 else own
        out.append([max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2),
                    labels.index(c) - 1])
    return out


def train(train_items: list[dict], val_items: list[dict], labels: list[str], size: int, out: Path,
          log=print) -> Path:
    """Fine-tunes PicoDet-S in its own Python; items: {"path", "boxes"} (boxes_of). Returns the ONNX file."""
    found = setup()
    if found is None:
        raise DetectorError("counting with PicoDet needs SIM2CHIP_PICODET_PYTHON (PaddlePaddle) and "
                            "SIM2CHIP_PADDLEDET (a PaddleDetection checkout)")
    py, det = found
    req = {"paddledet": det, "out": str(out), "size": size, "classes": labels[1:],
           "steps": int(os.environ.get("SIM2CHIP_PICODET_STEPS") or STEPS),
           "train": train_items, "val": val_items}
    repo = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": repo}
    tail = []
    with subprocess.Popen([py, "-m", "sim2chip.detector_worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, env=env) as proc:
        proc.stdin.write(json.dumps(req))
        proc.stdin.close()
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-20:]
            if line.startswith(("picodet", "  ")):
                log(f"  {line.rstrip()}")
    if proc.returncode != 0 or not (out / "model.onnx").exists():
        raise DetectorError(f"PicoDet training failed (exit {proc.returncode}):\n" + "\n".join(tail[-10:]))
    return out / "model.onnx"


def to_tflite(onnx_path: Path, size: int) -> bytes:
    """The worker's ONNX as a TFLite for phones and PCs: batch 1, input RGB 0-255 (NHWC once converted), the
    normalization as its first two ops, float16 weights."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(onnx_path)
    g = m.graph
    for v in [g.input[0], *g.output]:
        v.type.tensor_type.shape.dim[0].dim_value = 1
    std = np.array(STD, np.float32).reshape(1, 3, 1, 1)
    g.initializer.extend([numpy_helper.from_array((1 / (255 * std)).astype(np.float32), "rgb_scale"),
                          numpy_helper.from_array((-np.array(MEAN, np.float32).reshape(1, 3, 1, 1) / std)
                                                  .astype(np.float32), "rgb_shift")])
    old, g.input[0].name = g.input[0].name, "rgb"
    for n in g.node:
        n.input[:] = ["rgb_normalized" if x == old else x for x in n.input]
    g.node.insert(0, helper.make_node("Add", ["rgb_scaled", "rgb_shift"], ["rgb_normalized"]))
    g.node.insert(0, helper.make_node("Mul", ["rgb", "rgb_scale"], ["rgb_scaled"]))
    onnx.checker.check_model(m)
    import onnx2tf.onnx2tf as o2t
    # onnx2tf downloads sample photos to check its conversion; random ones do (and work offline).
    o2t.download_test_image_data = lambda: np.random.default_rng(0).uniform(0, 255, (20, size, size, 3)) \
        .astype(np.float32)
    # onnx2tf hides the GPU from this process (CUDA_VISIBLE_DEVICES=-1), and the next seed's PaddlePaddle, started
    # from here, would inherit that and fail at once (2026-10-02): put it back after.
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "model.onnx"
            onnx.save(m, src)
            o2t.convert(input_onnx_file_path=str(src), output_folder_path=str(Path(tmp) / "tf"),
                        not_use_onnxsim=True, verbosity="error")
            return (Path(tmp) / "tf" / "model_float16.tflite").read_bytes()
    finally:
        if gpus is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def load(paths: list[Path], size: int) -> np.ndarray:
    """Photos as the model takes them: RGB, squashed to size x size with Pillow's bilinear resize (as ingest
    and the live test do)."""
    from PIL import Image
    out = np.zeros((len(paths), size, size, 3), np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
    return out


def predict(blob: bytes, x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Raw outputs per photo: boxes (anchors, 4) in 0..1 and scores (classes, anchors)."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, outs = it.get_input_details()[0], it.get_output_details()
    size = inp["shape"][1]
    res = []
    for img in x:
        it.set_tensor(inp["index"], img[None].astype(np.float32))
        it.invoke()
        a, b = (it.get_tensor(o["index"])[0] for o in outs)
        boxes, scores = (a, b) if a.shape[-1] == 4 and a.ndim == 2 and b.shape[-1] != 4 else (b, a)
        res.append((boxes / size, scores))
    return res


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x1, y1 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
    union = area(box) + area(others) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def decode(out: tuple[np.ndarray, np.ndarray], labels: list[str], threshold: float) -> list[tuple[float, float, str]]:
    """Raw outputs -> the centre of every object, in 0..1: per class, the boxes scoring at least threshold, best
    first, each kept unless it overlaps a kept one more than NMS_IOU; at most MAX_OBJECTS, best first."""
    boxes, scores = out
    kept = []
    for c in range(scores.shape[0]):
        idx = np.flatnonzero(scores[c] >= threshold)
        idx = idx[np.argsort(-scores[c][idx], kind="stable")]
        chosen = []
        for i in idx:
            if not chosen or iou(boxes[i], boxes[chosen]).max() <= NMS_IOU:
                chosen.append(i)
        kept += [(float(scores[c][i]), i, c) for i in chosen]
    kept.sort(key=lambda k: -k[0])
    return [(round(float(boxes[i][0] + boxes[i][2]) / 2, 4), round(float(boxes[i][1] + boxes[i][3]) / 2, 4),
             labels[c + 1]) for _, i, c in kept[:MAX_OBJECTS]]


def pick_threshold(f1: dict[float, float]) -> float:
    """The threshold with the best F1; among those within 0.01 of it, the closest to PREFERRED."""
    best = max(f1.values())
    return min((t for t, v in f1.items() if v >= best - 0.01), key=lambda t: abs(t - PREFERRED))
