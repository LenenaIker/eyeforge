"""Teachers and students of follow.py against far points marked by hand (truth/mark.html -> truth.json).

    python experiments/line_follow/roads_truth.py PHOTOS TRUTH.json RUN [RUN ...]

RUN: a follow.py output folder (labels.json, <profile>_<size>_<recipe>_pred.npy). For each run: the teacher's far
point against the hand-marked one (error in photo widths, and the view targets' side, as in follow.scores), then
each student's predictions against the targets made from the hand marks. Photos marked "no path" or that a run's
teacher skipped are left out of that run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from follow import OFFSETS, side  # noqa: E402


def targets(x: float, w: int, h: int) -> list[float]:
    """The view targets follow.views makes from a far point x (0-1 of the photo): per offset, plain and mirrored."""
    s = min(w, h)
    out = []
    for cx in np.linspace(s / 2, w - s / 2, OFFSETS):
        t = float(np.clip((x * w - cx) / (s / 2), -1, 1))
        out += [t, -t]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("truth", type=Path)
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    truth = json.loads(a.truth.read_text())
    sizes = {p.name: ImageOps.exif_transpose(Image.open(p)).size for p in sorted(a.photos.glob("*.jpg"))}
    print(f"{len(truth)} marked, {sum(v['x'] is None for v in truth.values())} with no path")
    for run in a.runs:
        labels = json.loads((run / "labels.json").read_text())
        good = [lab for lab in labels if lab["ok"]]  # the order follow.py trained and predicted in
        keep = [i for i, lab in enumerate(good) if truth.get(lab["photo"], {}).get("x") is not None]
        tx = np.array([truth[good[i]["photo"]]["x"] for i in keep])
        px = np.array([good[i]["x"] for i in keep])
        yt = np.array([targets(truth[good[i]["photo"]]["x"], *sizes[good[i]["photo"]]) for i in keep])
        yp = np.array([targets(good[i]["x"], *sizes[good[i]["photo"]]) for i in keep])
        st, sp = side(yt.ravel()), side(yp.ravel())
        print(f"\n{run.name}: teacher on {len(keep)} photos: far point error {np.abs(px - tx).mean():.3f} photo widths "
              f"(median {np.median(np.abs(px - tx)):.3f}), view side right {(st == sp).mean():.3f}, "
              f"opposite side {(st * sp == -1).mean():.3f}")
        for f in sorted(run.glob("*_pred.npy")):
            pred = np.load(f)[keep]
            sp = side(pred.ravel())
            print(f"  student {f.stem[:-5]:24} error {np.abs(pred - yt).mean():.3f}, side right {(st == sp).mean():.3f}, "
                  f"opposite side {(st * sp == -1).mean():.3f}")


if __name__ == "__main__":
    main()
