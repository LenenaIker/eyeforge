"""Count benchmark on FSC-147 (one class): score the teacher's marks and the student's test predictions against
the dataset's own points, and write ground-truth corrections to simulate a user who reviewed some photos.
The dataset lives in data/fsc147 (or FSC_DIR).

  [FSC_CLASS=apples FSC_NAME=apple] python experiments/fsc147_count.py score JOB                 # teacher (all photos, test) and student vs truth
  python experiments/fsc147_count.py fix JOB N                 # truth as corrections for N train photos (all: -1)

Truth points get the mean size of the photo's exemplar boxes, as the teacher's marks do, so a prediction
anywhere in the object's core matches it in both cases.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim2chip import counting  # noqa: E402
from sim2chip.workspace import content_key  # noqa: E402

FSC = Path(os.environ.get("FSC_DIR", ROOT / "data" / "fsc147"))
# The FSC-147 class (its folder of photos under FSC) and the name our spec gives it.
FOLDER = os.environ.get("FSC_CLASS", "geese")
CLASS = os.environ.get("FSC_NAME", "goose")


def truth_of() -> dict[str, list]:
    from PIL import Image
    ann = json.loads((FSC / "annotation_FSC147_384.json").read_text())
    out = {}
    for f in sorted((FSC / FOLDER).iterdir()):
        a = ann[f.name]
        with Image.open(f) as im:
            w, h = im.size
        boxes = [(max(p[0] for p in b) - min(p[0] for p in b), max(p[1] for p in b) - min(p[1] for p in b))
                 for b in a["box_examples_coordinates"]]
        bw = sum(b[0] for b in boxes) / len(boxes) / w
        bh = sum(b[1] for b in boxes) / len(boxes) / h
        out[f.name] = [(x / w, y / h, CLASS, bw, bh) for x, y in a["points"]]
    return out


def job_dir(name: str) -> Path:
    return ROOT / "jobs" / name if (ROOT / "jobs" / name).exists() else ROOT / "data" / "jobs" / name


def sides(job: Path) -> dict[str, str]:
    """Photo name -> test / train / unused."""
    split = json.loads((job / "ingest" / "split.json").read_text())
    return {f.name: split.get(content_key(f), "unused") for f in sorted((FSC / FOLDER).iterdir())}


def score(job: Path) -> dict:
    truth, side = truth_of(), sides(job)
    grid = counting.grid_size(224)  # hardware "generic"
    verdicts = json.loads((job / "label" / "verdicts.json").read_text())
    out = {}

    def teacher(sel):
        keys = [k for k in sel if k in verdicts and verdicts[k]["objects"] is not None]
        preds = [[(o["x"], o["y"], o["class"]) for o in verdicts[k].get("teacher_objects", verdicts[k]["objects"])]
                 for k in keys]
        return counting.score(preds, [truth[k] for k in keys], grid)

    out["teacher_all"] = teacher(list(truth))
    out["teacher_test"] = teacher([k for k, s in side.items() if s == "test"])
    seeds = {}
    for f in sorted((job / "evaluate").glob("seed_*.json")):
        ev = json.loads(f.read_text())
        preds = [[tuple(p) for p in q["pred"]] for q in ev["predictions"]]
        counts = [q["count"] for q in ev["predictions"]] if all("count" in q for q in ev["predictions"]) else None
        seeds[f.stem] = counting.score(preds, [truth[q["key"]] for q in ev["predictions"]], grid, counts)
        if counts:  # by crowd size, as in count_density.py
            tr = [truth[q["key"]] for q in ev["predictions"]]
            err = [abs(sum(c.values()) - len(t)) for c, t in zip(counts, tr)]
            seeds[f.stem]["by_size"] = {f"{lo}-{hi}": round(sum(e for e, t in zip(err, tr) if lo <= len(t) <= hi)
                                                         / max(1, sum(lo <= len(t) <= hi for t in tr)), 1)
                                        for lo, hi in ((0, 15), (16, 30), (31, 10_000))}
    out["student"] = seeds
    out["student_f1_mean"] = round(sum(s["f1"] for s in seeds.values()) / max(len(seeds), 1), 4)
    out["student_mae_mean"] = round(sum(s["count_mae"] for s in seeds.values()) / max(len(seeds), 1), 3)
    out["sizes"] = {s: sum(v == s for v in side.values()) for s in ("test", "train", "unused")}
    return out


def fix(job: Path, n: int) -> None:
    truth, side = truth_of(), sides(job)
    train = sorted(k for k, s in side.items() if s == "train")
    pick = train if n < 0 else random.Random(0).sample(train, n)
    f = job / "input" / "corrections.json"
    fixes = json.loads(f.read_text()) if f.exists() else {"labels": {}, "assets": {}, "points": {}}
    fixes["points"] = {k: [[round(x, 4), round(y, 4), c] for x, y, c, *_ in truth[k]] for k in pick}
    f.write_text(json.dumps(fixes, indent=1))
    print(f"{len(pick)} train photos corrected with the truth in {f}")


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "score":
        print(json.dumps(score(job_dir(name)), indent=1))
    else:
        fix(job_dir(name), int(sys.argv[3]))
