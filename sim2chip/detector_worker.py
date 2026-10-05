"""Fine-tunes PicoDet-S on a job's boxes and exports it to ONNX, in the Python that has PaddlePaddle, from a
PaddleDetection checkout (release 2.8.1). The train stage runs it through detector.py.

    python -m sim2chip.detector_worker < request.json

Request: {"paddledet", "out", "size", "classes": [name, ...], "steps" (about how many training steps),
          "train": [{"path", "boxes": [[x0, y0, x1, y1, class index], ...]}], "val": [...]}, boxes in 0..1.
Writes out/model.onnx: input "image" (1, 3, size, size) normalized as PicoDet wants, outputs the boxes of every
anchor (1, anchors, 4) in input pixels and their class scores (1, classes, anchors), before NMS.
The best checkpoint by box AP on val is the one exported. Progress lines on stdout.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

# COCO-trained PicoDet-S 320 (PaddleDetection model zoo): the starting point, whatever the input size.
BASE = "configs/picodet/picodet_s_320_coco_lcnet.yml"
WEIGHTS = "https://paddledet.bj.bcebos.com/models/picodet_s_320_coco_lcnet.pdparams"
# A copy inside the checkout, when there is one (the cloud kit brings it: the model zoo's server is slow from
# Europe), named without the extension as PaddleDetection wants local weights.
LOCAL_WEIGHTS = "weights/picodet_s_320_coco_lcnet"
BATCH = 16
# Steps per epoch at least this many: PaddleDetection starts its data loader workers anew every epoch (about
# 1.6 s against milliseconds for the step). With 10 photos an epoch was one step, 900 of them: 25 min per seed with
# the GPU idle (2026-10-04). Small sets are listed several times per epoch instead, each copy augmented on its own;
# the number of steps stays the same. 7 still spent half the training restarting workers. On FSC-147 geese, 600
# steps, 3 seeds (2026-10-05): 50 per epoch F1 0.714, 100 0.719, 200 0.705 (3 checkpoints only), each a few
# seconds faster than the one before.
MIN_STEPS_PER_EPOCH = 100
MIN_EPOCHS = 3  # at least this many epochs (and as many checkpoints to pick the best from), whatever the steps
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Photos are handed over at most this many times the input size on their long side. Phone photos are 12 MP: the
# data loader decoded 16 of them per step to crop them to about 320 px, and the GPU waited (8 min per seed on 10
# photos, 2026-10-04). Twice the input keeps the detail RandomCrop and the larger multi-scale sizes need.
MAX_SIDE_X = 2


def shrink(items: list[dict], size: int, folder: Path) -> list[dict]:
    """Upright (EXIF applied, as the teacher saw them), downscaled copies of the photos, same boxes (0..1)."""
    from PIL import Image, ImageOps
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, it in enumerate(items):
        dst = folder / f"{i:05d}.jpg"
        with Image.open(it["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE_X * size, MAX_SIDE_X * size), Image.BILINEAR)
            im.save(dst, quality=95)
        out.append({**it, "path": str(dst)})
    return out


def coco(items: list[dict], classes: list[str], path: Path) -> None:
    from PIL import Image
    out = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]}
    for i, it in enumerate(items):
        with Image.open(it["path"]) as im:
            w, h = im.size
        out["images"].append({"id": i + 1, "file_name": it["path"], "width": w, "height": h})
        for x0, y0, x1, y1, c in it["boxes"]:
            bw, bh = (x1 - x0) * w, (y1 - y0) * h
            out["annotations"].append({"id": len(out["annotations"]) + 1, "image_id": i + 1, "category_id": int(c) + 1,
                                       "bbox": [x0 * w, y0 * h, bw, bh], "area": bw * bh, "iscrowd": 0})
    path.write_text(json.dumps(out))


def config(req: dict, out: Path, epochs: int, batch: int) -> Path:
    """The model zoo config with this job's data, classes, size and schedule. The learning rate follows the
    batch (the zoo's 0.32 is for 4 GPUs of 64), and the NMS keeps up to 1000 objects (crowds)."""
    size, base = req["size"], str(Path(req["paddledet"]) / BASE)
    local = Path(req["paddledet"]) / LOCAL_WEIGHTS
    weights = local if local.with_suffix(".pdparams").exists() else WEIGHTS
    sizes = [s for s in range(size - 64, size + 65, 32) if s >= 96]
    norm = "{is_scale: true, mean: %s, std: %s}" % (MEAN, STD)
    data = lambda part: f"""  name: COCODataSet
  image_dir: /
  anno_path: {out / f'{part}.json'}
  dataset_dir: /
"""  # noqa: E731
    text = f"""_BASE_: ['{base}']
pretrain_weights: {weights}
num_classes: {len(req['classes'])}
eval_height: {size}
eval_width: {size}
eval_size: [{size}, {size}]
epoch: {epochs}
snapshot_epoch: {max(1, epochs // 15)}
log_iter: 1000
TrainDataset:
{data('train')}  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
EvalDataset:
{data('val')}TestDataset:
  name: ImageFolder
  anno_path: {out / 'val.json'}
LearningRate:
  base_lr: {0.32 * batch / 256:.4f}
  schedulers:
  - !CosineDecay
    max_epochs: {epochs}
  - !LinearWarmup
    start_factor: 0.1
    steps: {min(100, epochs * 2)}
PicoHeadV2:
  nms:
    name: MultiClassNMS
    nms_top_k: {min(1000, sum(math.ceil(size / s) ** 2 for s in (8, 16, 32, 64)))}
    keep_top_k: 1000
    score_threshold: 0.025
    nms_threshold: 0.6
TrainReader:
  batch_size: {batch}
  drop_last: false
  batch_transforms:
  - BatchRandomResize: {{target_size: {sizes}, random_size: True, random_interp: True, keep_ratio: False}}
  - NormalizeImage: {norm}
  - Permute: {{}}
  - PadGT: {{}}
EvalReader:
  sample_transforms:
  - Decode: {{}}
  - Resize: {{interp: 2, target_size: [{size}, {size}], keep_ratio: False}}
  - NormalizeImage: {norm}
  - Permute: {{}}
TestReader:
  inputs_def:
    image_shape: [1, 3, {size}, {size}]
  sample_transforms:
  - Decode: {{}}
  - Resize: {{interp: 2, target_size: [{size}, {size}], keep_ratio: False}}
  - NormalizeImage: {norm}
  - Permute: {{}}
"""
    path = out / "picodet.yml"
    path.write_text(text)
    return path


def run(cmd: list[str], cwd: str, label: str) -> None:
    tail = []
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            tail = (tail + [line.rstrip()])[-30:]
            if "Best test bbox ap" in line or "Epoch: [" in line and "[0/" in line and "learning_rate" in line \
                    and int(line.split("Epoch: [")[1].split("]")[0]) % 10 == 0:
                print(f"  {label}: {line.split('INFO: ')[-1].strip()[:120]}", flush=True)
    if proc.returncode != 0:
        print("\n".join(tail), flush=True)
        raise SystemExit(f"{label} failed (exit {proc.returncode})")


def main():
    req = json.loads(sys.stdin.read())
    out, det, py = Path(req["out"]), req["paddledet"], sys.executable
    out.mkdir(parents=True, exist_ok=True)
    req["train"] = shrink(req["train"], req["size"], out / "images" / "train")
    req["val"] = shrink(req["val"], req["size"], out / "images" / "val")
    n = len(req["train"])
    repeat = 1 if math.ceil(n / BATCH) >= MIN_STEPS_PER_EPOCH else math.ceil(MIN_STEPS_PER_EPOCH * BATCH / n)
    coco(req["train"] * repeat, req["classes"], out / "train.json")
    coco(req["val"], req["classes"], out / "val.json")
    batch = max(1, min(BATCH, n * repeat))
    epochs = max(MIN_EPOCHS, math.ceil(req["steps"] / math.ceil(n * repeat / batch)))
    cfg = config(req, out, epochs, batch)
    print(f"picodet on {n} train" + (f" (each {repeat} times per epoch)" if repeat > 1 else "")
          + f" and {len(req['val'])} val photos, {epochs} epochs", flush=True)
    # Run from the job's folder: PaddleDetection writes files (bbox.json) where it runs, and the checkout may be
    # read-only.
    tools = Path(det) / "tools"
    run([py, str(tools / "train.py"), "-c", str(cfg), "--eval", "-o", f"save_dir={out / 'run'}"], str(out), "train")
    run([py, str(tools / "export_model.py"), "-c", str(cfg), "--output_dir", str(out / "infer"), "-o",
         f"weights={out / 'run' / 'best_model'}", "export.post_process=True", "export.nms=False"], str(out), "export")
    infer = out / "infer" / cfg.stem
    paddle2onnx = Path(py).parent / "paddle2onnx"
    run([str(paddle2onnx), "--model_dir", str(infer), "--model_filename", "model.pdmodel", "--params_filename",
         "model.pdiparams", "--opset_version", "13", "--save_file", str(out / "model.onnx")], str(out), "onnx")
    print(f"picodet exported to {out / 'model.onnx'}", flush=True)


if __name__ == "__main__":
    main()
