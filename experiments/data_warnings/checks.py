"""Prototype of the data warnings (task 9, docs/propuesta_etiquetas_y_avisos.md): checks computed in code on a
labeling's marks and on the student's input size for the user's hardware, each with the plain English text the
chat would show in a yellow notice. Nothing here is wired into the product: thresholds are a proposal.

Sources it reads (all read only):
  - a job folder: spec.json (hardware, task) and label/verdicts.json (the marks the train stage uses: the
    teacher's, with the user's winning on reviewed photos), plus input/labeling.json for what was reviewed;
  - a library.db labeling (the single source of truth since 2026-10-10): mark rows (centre x, y and size w, h
    in 0..1; w and h null on a user's click);
  - FSC-147 subsets (points per object, three example boxes per photo: their mean size stands for every object).

Usage:
  python experiments/data_warnings/checks.py                 # every dataset found, job hardware + ESP32-CAM
  python experiments/data_warnings/checks.py --json out.json  # also write the results
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip import counting, hardware  # noqa: E402  (pure Python, no TensorFlow)
from sim2chip.detector import NMS_IOU  # noqa: E402

CRUCIAL = Path("/media/lenena-iker/Crucial")
P2S = CRUCIAL / "runs/eyeforge/p2s_test"

# Proposed thresholds (the user decides; reasons in the doc).
SMALL_PX = 8            # an object under this many px (square root of its box area) at the model's input
SMALL_SHARE = 0.25      # warn when at least this share of a class's objects is that small
MERGE_SHARE = 0.10      # warn when at least this share of a class's objects would merge with a neighbour
FEW_STRONG = 10         # photos per class under which training is close to guessing (Vertex AI minimum)
FEW_MILD = 30           # photos per class Apple Create ML and Teachable Machine guides suggest to start
IMBALANCE = 3.0         # largest class over smallest, in photos (classify) or objects (count)
TEST_MIN = 20           # ingest.MIN_TEST: fewer test photos per class move the score 5+ points
TEST_SHARE = 0.5        # ingest.split: half of the photos go to the test while there are fewer than its caps
EDGE_SHARE = 0.30       # optional: share of objects cut by the photo's border
EDGE_EPS = 0.005
BACKGROUND = "empty"     # spec.background_class: skipped by few_photos when no photo uses it

HW_NAMES = {"esp32cam": "ESP32-CAM", "esp32s3": "ESP32-S3", "jetson_nano": "Jetson Nano",
            "generic": "a phone or PC"}


@dataclass
class Obj:
    cls: str
    x: float
    y: float
    w: float | None = None  # share of the photo width; None for a click without size
    h: float | None = None

    def box(self):
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class Data:
    name: str
    task: str                       # "count" or "classify"
    classes: list[str]
    hardware: str | None            # the job's profile; None for a library labeling without a job
    photos: dict[str, list[Obj]] = field(default_factory=dict)   # count: objects per photo
    sorted: dict[str, str] = field(default_factory=dict)         # classify: class per photo
    reviewed: int = 0


def plain(key: str) -> str:
    """A class key as a user reads it: sun_cream -> sun cream."""
    return key.replace("_", " ")


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------- loaders

def from_job(job: Path) -> Data:
    spec = json.loads((job / "spec.json").read_text())
    verdicts = json.loads((job / "label/verdicts.json").read_text())
    task = "count" if spec["task"] == "count" else "classify"
    d = Data(f"job {job.name}", task, [c["id"] for c in spec["classes"]], spec["hardware"])
    lab = job / "input/labeling.json"
    if lab.exists():
        d.reviewed = sum(p["reviewed"] for p in json.loads(lab.read_text())["photos"].values())
    for name, v in verdicts.items():
        if task == "classify":
            c = v.get("assigned") or v.get("class")
            if c in d.classes:
                d.sorted[name] = c
            continue
        if v.get("objects") is None:
            continue
        objs = []
        for o in v["objects"]:
            b = o.get("box")
            if b:
                objs.append(Obj(o["class"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]))
            else:
                objs.append(Obj(o["class"], o["x"], o["y"]))
        d.photos[name] = objs
    return d


def from_library(db: Path, hw: str | None = None) -> list[Data]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for lid, lname, task, ds in con.execute("select id, name, task, dataset_id from labeling"):
        classes = [k for (k,) in con.execute("select key from class where labeling_id=? order by position", (lid,))]
        d = Data(f"library {db.parents[1].name}: {lname}", task, classes, hw)
        d.reviewed = con.execute("select count(*) from review where labeling_id=?", (lid,)).fetchone()[0]
        photos = [p for (p,) in con.execute("select id from photo where dataset_id=?", (ds,))]
        rows = con.execute("select m.photo_id, c.key, m.x, m.y, m.w, m.h from mark m join class c on c.id=m.class_id "
                           "where c.labeling_id=?", (lid,)).fetchall()
        per = defaultdict(list)
        for pid, key, x, y, w, h in rows:
            per[pid].append((key, x, y, w, h))
        for pid in photos:
            if task == "classify":
                if per[pid]:
                    d.sorted[str(pid)] = per[pid][0][0]
            else:
                d.photos[str(pid)] = [Obj(k, x, y, w, h) for k, x, y, w, h in per[pid] if x is not None]
        out.append(d)
    return out


def from_fsc147(subset: str) -> Data:
    from PIL import Image
    base = CRUCIAL / "datasets/eyeforge/fsc147"
    ann = json.loads((base / "annotation_FSC147_384.json").read_text())
    one = {"apples": "apple", "geese": "goose", "bread_rolls": "bread_roll", "strawberries": "strawberry"}[subset]
    d = Data(f"FSC-147 {subset}", "count", [one], None)
    for name in (base / f"{subset}.txt").read_text().split():
        a = ann[name]
        with Image.open(base / "images" / name) as im:
            W, H = im.size
        ex = a["box_examples_coordinates"]
        ws = [abs(b[2][0] - b[0][0]) / W for b in ex]
        hs = [abs(b[2][1] - b[0][1]) / H for b in ex]
        w, h = statistics.mean(ws), statistics.mean(hs)
        d.photos[name] = [Obj(one, px / W, py / H, w, h) for px, py in a["points"]]
    d.reviewed = len(d.photos)  # the dataset's own labels are the truth
    return d


# ---------------------------------------------------------------- checks

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merges(hw: hardware.Hardware, a: Obj, b: Obj) -> bool:
    """Would the model see a and b (same class, same photo) as one object? The grid counter (microcontrollers)
    joins neighbouring cells of one class into a blob: centres in the same or a side-by-side cell are one
    object (counting.decode). PicoDet drops a box that overlaps a better one by more than NMS_IOU."""
    if hw.counter == "grid":
        g = counting.grid_size(hw.input_size)
        ca = (min(int(a.x * g), g - 1), min(int(a.y * g), g - 1))
        cb = (min(int(b.x * g), g - 1), min(int(b.y * g), g - 1))
        return abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) <= 1
    if a.w is None or b.w is None:
        return False
    return iou(a.box(), b.box()) > NMS_IOU


def same_cell(hw: hardware.Hardware, a: Obj, b: Obj) -> bool:
    """Stricter variant for the grid: both centres in one cell, so the model can only ever give one point."""
    g = counting.grid_size(hw.input_size)
    return (min(int(a.x * g), g - 1), min(int(a.y * g), g - 1)) == (min(int(b.x * g), g - 1), min(int(b.y * g), g - 1))


def touches(a: Obj, b: Obj) -> bool:
    return a.w is not None and b.w is not None and iou(a.box(), b.box()) > 0


def check(d: Data, hw_name: str) -> list[dict]:
    hw = hardware.get(hw_name)
    size = hw.count_input() if d.task == "count" else hw.input_size
    device = HW_NAMES.get(hw_name, hw_name)
    out = []

    def warn(kind, cls, text, **numbers):
        out.append({"check": kind, "class": cls, "text": text, **numbers})

    # photos per class, the same for both tasks: classify counts sorted photos, count the photos showing it
    if d.task == "classify":
        per_class = Counter(d.sorted.values())
    else:
        per_class = Counter(c for objs in d.photos.values() for c in {o.cls for o in objs})
    for c in d.classes:
        n = per_class.get(c, 0)
        if n == 0 and c == BACKGROUND:
            continue  # the "none of the others" class of a classify labeling, unused in this dataset
        if n < FEW_STRONG:  # the mild level also says the test is small: half the photos go to it
            some = "No photo is" if n == 0 else f"Only {plural(n, 'photo')} {'is' if n == 1 else 'are'}"
            warn("few_photos", c, f"{some} "
                 f"{'sorted as' if d.task == 'classify' else 'showing'} \"{plain(c)}\". The model needs at least "
                 f"{FEW_STRONG} to learn anything about it, and {FEW_MILD} or more to do it well. Add photos "
                 f"of {plain(c)} in different places and light.", photos=n, level="strong")
        elif n < FEW_MILD:
            warn("few_photos", c, f"{n} photos {'are sorted as' if d.task == 'classify' else 'show'} "
                 f"\"{plain(c)}\". That is enough to start, but {FEW_MILD} or more, in different places and "
                 "light, make the model much steadier.", photos=n, level="mild")
        elif round(n * TEST_SHARE) < TEST_MIN:  # enough photos to train, too few to score: only then, not on top of few_photos
            test = round(n * TEST_SHARE)
            warn("small_test", c, f"About {test} photos of \"{plain(c)}\" are kept apart to test the model, so "
                 "its score for this class can be off by 5 points or more. More photos make the score "
                 "trustworthy.", test=test)

    # imbalance: photos per class (classify) or objects per class (count, several classes)
    if len(d.classes) > 1:
        sizes = per_class if d.task == "classify" else Counter(o.cls for objs in d.photos.values() for o in objs)
        big = max(d.classes, key=lambda c: sizes.get(c, 0))
        small = min(d.classes, key=lambda c: sizes.get(c, 0))
        a, b = sizes.get(big, 0), sizes.get(small, 0)
        if b and a / b >= IMBALANCE:
            unit = "photos" if d.task == "classify" else "marked objects"
            warn("imbalance", small, f"\"{plain(big)}\" has {a} {unit} and \"{plain(small)}\" only {b} "
                 f"({a / b:.0f} times fewer). The model will lean towards {plain(big)} when unsure. Add photos "
                 f"of {plain(small)}, or leave some of {plain(big)} out.", ratio=round(a / b, 1), big=big)

    if d.task != "count":
        return out

    by_class: dict[str, list[Obj]] = defaultdict(list)
    for objs in d.photos.values():
        for o in objs:
            by_class[o.cls].append(o)
    for c, objs in by_class.items():
        sized = [o for o in objs if o.w is not None]
        # the photo is squashed to size x size, so each side scales on its own
        px = [math.sqrt(o.w * size * o.h * size) for o in sized]
        if px:
            share = sum(p < SMALL_PX for p in px) / len(px)
            if share >= SMALL_SHARE:
                med = statistics.median(px)
                warn("small_objects", c, f"{share:.0%} of your {plain(c)} marks are too small for this device. On "
                     f"{device} the model sees each photo at {size} x {size} px, and those end up smaller than "
                     f"{SMALL_PX} px{f' (half of all of them under {med:.0f} px)' if med < SMALL_PX else ''}: "
                     f"it will miss {'most of them' if share >= 0.5 else 'them'}. Take the photos closer, so each "
                     f"{plain(c)} fills more of the picture.",
                     share=round(share, 2), median_px=round(statistics.median(px), 1), input=size, objects=len(px))
        # neighbours: the model would count two as one
        merged = touching = strict = 0
        for objs_p in d.photos.values():
            same = [o for o in objs_p if o.cls == c]
            for i, a in enumerate(same):
                others = same[:i] + same[i + 1:]
                merged += any(merges(hw, a, b) for b in others)
                touching += any(touches(a, b) for b in others)
                strict += hw.counter == "grid" and any(same_cell(hw, a, b) for b in others)
        if objs and merged / len(objs) >= MERGE_SHARE:
            close = ("overlap another one so much" if hw.counter != "grid" else "are so close to another one")
            warn("merged", c, f"{merged / len(objs):.0%} of your {plain(c)} marks {close} that the model on "
                 f"{device} will see two as one, and count fewer than there are. Photos where they stand apart, "
                 "or taken closer, count better.",
                 share=round(merged / len(objs), 2), touching=round(touching / len(objs), 2),
                 same_cell=round(strict / len(objs), 2), objects=len(objs))
        # optional: cut by the photo's border
        cut = sum(1 for o in sized if min(o.box()[:2]) <= EDGE_EPS or max(o.box()[2:]) >= 1 - EDGE_EPS)
        if sized and cut / len(sized) >= EDGE_SHARE:
            warn("cut_by_edge", c, f"{cut / len(sized):.0%} of your {plain(c)} marks are cut off by the edge of "
                 f"the photo. Half a {plain(c)} looks different from a whole one: keep them inside the picture "
                 "when you can.", share=round(cut / len(sized), 2), optional=True)

    # existing product check (counting.crowded), for comparison
    dense = counting.crowded([len(v) for v in d.photos.values()])
    if dense:
        warn("crowded", None, dense["note"], share=dense["share"], existing=True)
    # optional: the score is measured against the teacher, not against the user
    if d.reviewed < TEST_MIN:
        warn("unreviewed_test", None, f"You checked {plural(d.reviewed, 'photo')}. The model's score will be "
             f"measured partly against the teacher's own marks, so it may look better than it is. Check at least "
             f"{TEST_MIN} photos to get a score you can trust.", reviewed=d.reviewed, optional=True)
    return out


def stats(d: Data, hw_name: str) -> dict:
    hw = hardware.get(hw_name)
    size = hw.count_input() if d.task == "count" else hw.input_size
    s = {"task": d.task, "photos": len(d.photos) if d.task == "count" else len(d.sorted), "reviewed": d.reviewed,
         "input": size}
    if d.task == "classify":
        s["per_class"] = dict(Counter(d.sorted.values()))
    else:
        objs = [o for v in d.photos.values() for o in v]
        px = [math.sqrt(o.w * o.h) * size for o in objs if o.w is not None]
        s.update(objects=len(objs), empty=sum(not v for v in d.photos.values()),
                 median_px=round(statistics.median(px), 1) if px else None,
                 under_8px=round(sum(p < SMALL_PX for p in px) / len(px), 2) if px else None)
    return s


def datasets() -> list[Data]:
    found = []
    for job in sorted((P2S / "jobs").iterdir()):
        if job.name == "bottles_scene":  # same labeling and marks as bottles_few
            continue
        found.append(from_job(job))
    for db in (CRUCIAL / "runs/eyeforge/web_local/library/library.db",
               CRUCIAL / "datasets/eyeforge/e2e_labeling/library/library.db"):
        if db.exists():
            found += from_library(db)
    for subset in ("apples", "geese", "bread_rolls", "strawberries"):
        found.append(from_fsc147(subset))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--hardware", nargs="*", help="profiles to check besides the job's (default esp32cam)")
    args = ap.parse_args()
    extra = args.hardware if args.hardware is not None else ["esp32cam"]
    results = []
    for d in datasets():
        for hw in dict.fromkeys(([d.hardware] if d.hardware else ["generic"]) + extra):
            ws = check(d, hw)
            r = {"dataset": d.name, "hardware": hw, "own_hardware": hw == (d.hardware or "generic"),
                 "stats": stats(d, hw), "warnings": ws}
            results.append(r)
            print(f"\n== {d.name}  [{hw}{'' if r['own_hardware'] else ', what if'}]  {r['stats']}")
            for w in ws:
                tag = "optional" if w.get("optional") else "existing" if w.get("existing") else w.get("level", "warn")
                print(f"  - {w['check']} ({tag}): {w['text']}")
            if not ws:
                print("  (no warning)")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
