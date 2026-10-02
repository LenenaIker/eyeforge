"""Off-the-shelf detectors against our count head on FSC-147 geese: the same train and test photos as a count job
(geese_C: train corrected with the dataset's points), scored with the same point matching (experiments/count_arch.py).

  python experiments/count_detector.py export JOB OUT        # YOLO and COCO datasets: train (80%), val (20%), test
  python experiments/count_detector.py score JOB PRED.json   # threshold picked on val, then F1 and count error on test

Boxes for training are the dataset's points (the job's train was corrected to them) with the mean exemplar box of
the photo as their size, as the teacher marks them. PRED.json: {"val"|"test": {photo name: [[cx, cy, w, h, score], ...]}}, normalised to the image.
Matching uses the 224 px count grid's tolerance whatever the detector's input size, so the numbers compare.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from fsc147_count import CLASS, job_dir, truth_of  # noqa: E402
from sim2chip import counting  # noqa: E402

GRID = counting.grid_size(224)
THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]
VAL_SHARE = 0.2


def sides(job: Path) -> dict[str, dict]:
    """train / val / test -> {photo name: its points.json entry}; val is a fixed 20% of the job's train."""
    train = json.loads((job / "ingest" / "train" / "points.json").read_text())
    names = sorted(train)
    val = set(np.random.default_rng(0).permutation(names)[:round(len(names) * VAL_SHARE)])
    return {"train": {n: v for n, v in train.items() if n not in val},
            "val": {n: v for n, v in train.items() if n in val},
            "test": json.loads((job / "ingest" / "test" / "points.json").read_text())}


def export(job: Path, out: Path):
    from PIL import Image
    truth = truth_of()
    for side, table in sides(job).items():
        (out / "images" / side).mkdir(parents=True, exist_ok=True)
        (out / "labels" / side).mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": CLASS}]}
        for i, (n, v) in enumerate(sorted(table.items())):
            src = job / v["source"]
            dst = out / "images" / side / src.name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            with Image.open(src) as im:
                w, h = im.size
            coco["images"].append({"id": i + 1, "file_name": src.name, "width": w, "height": h})
            rows = []
            for x, y, _, bw, bh in truth[v["key"]]:
                x0, y0 = max(0.0, x - bw / 2), max(0.0, y - bh / 2)
                x1, y1 = min(1.0, x + bw / 2), min(1.0, y + bh / 2)
                rows.append(f"0 {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": i + 1, "category_id": 1,
                                            "bbox": [x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h],
                                            "area": (x1 - x0) * w * (y1 - y0) * h, "iscrowd": 0})
            (out / "labels" / side / (src.stem + ".txt")).write_text("\n".join(rows))
        (out / f"{side}.json").write_text(json.dumps(coco))
        print(side, len(table), "photos", sum(len(truth[v["key"]]) for v in table.values()), "objects")
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                   f"names:\n  0: {CLASS}\n")


def points(dets: list, threshold: float) -> list:
    return [(d[0], d[1], CLASS) for d in dets if d[4] >= threshold]


def by_crowd(preds: list, truths: list) -> dict:
    out = {}
    for name, lo, hi in (("<=15", 0, 15), ("16-30", 16, 30), (">30", 31, 10 ** 6)):
        sel = [i for i, t in enumerate(truths) if lo <= len(t) <= hi]
        if sel:
            s = counting.score([preds[i] for i in sel], [truths[i] for i in sel], GRID)
            out[name] = {"photos": len(sel), "recall": s["recall"], "count_mae": s["count_mae"]}
    return out


def score(job: Path, pred_file: Path) -> dict:
    pred, truth, split = json.loads(pred_file.read_text()), truth_of(), sides(job)

    def evaluate(side, th):
        table = split[side]
        names = sorted(table)
        return ([points(pred[side].get(Path(table[n]["source"]).name, []), th) for n in names],
                [truth[table[n]["key"]] for n in names])

    val = {t: counting.score(*evaluate("val", t), GRID)["f1"] for t in THRESHOLDS}
    th = max(THRESHOLDS, key=lambda t: (val[t], -abs(t - 0.5)))
    p, t = evaluate("test", th)
    res = {"threshold": th, **counting.score(p, t, GRID),
           "oracle": max(counting.score(*evaluate("test", x), GRID)["f1"] for x in THRESHOLDS),
           "crowd": by_crowd(p, t)}
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    cmd, name, path = sys.argv[1:4]
    {"export": export, "score": score}[cmd](job_dir(name), Path(path))
