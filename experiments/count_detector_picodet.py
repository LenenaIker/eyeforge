"""PicoDet for experiments/count_detector.py: fine-tune with PaddleDetection, then write its val and test
detections for `count_detector.py score`. Runs in an environment with paddle, from a PaddleDetection checkout.

  python experiments/count_detector_picodet.py PADDLEDET CONFIG DATASET NAME     # NAME: e.g. picodet_s-320-0
Writes DATASET/preds/NAME.json; the run itself goes to DATASET/picodet/NAME.
"""
import json
import subprocess
import sys
from pathlib import Path

det, cfg, data, name = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4]
run = data / "picodet" / name
run.mkdir(parents=True, exist_ok=True)
py = sys.executable
if not (run / "best_model.pdparams").exists():
    subprocess.run([py, "tools/train.py", "-c", str(cfg), "--eval", "-o", f"save_dir={run}"], cwd=det, check=True)
out = {}
for side in ("val", "test"):
    subprocess.run([py, "tools/eval.py", "-c", str(cfg), "--output_eval", str(run / side), "-o",
                    f"weights={run / 'best_model'}", f"EvalDataset.anno_path={side}.json",
                    f"EvalDataset.image_dir=images/{side}"], cwd=det, check=True)
    coco = json.loads((data / f"{side}.json").read_text())
    images = {i["id"]: i for i in coco["images"]}
    out[side] = {i["file_name"]: [] for i in images.values()}
    for d in json.loads((run / side / "bbox.json").read_text()):
        im = images[d["image_id"]]
        x, y, w, h = d["bbox"]
        out[side][im["file_name"]].append([(x + w / 2) / im["width"], (y + h / 2) / im["height"],
                                           w / im["width"], h / im["height"], d["score"]])
(data / "preds").mkdir(exist_ok=True)
(data / "preds" / f"{name}.json").write_text(json.dumps(out))
