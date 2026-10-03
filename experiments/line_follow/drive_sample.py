"""Repeated samples of Sully Chen's driving dataset 1 (dashcam, steering wheel angle, MIT) for the teacher tests.

    python experiments/line_follow/drive_sample.py DATA OUT [--samples 3] [--per-class 50]

DATA/data.txt has "<n>.jpg <angle>", degrees of steering wheel, positive to the right (checked by eye 2026-10-03).
Classes as in the trails test: left (-1) under -TURN, straight (0) within +-STRAIGHT, right (+1) over TURN; the
frames in between are left out so the classes do not blur into each other. Each sample is its own folder of links
with labels.json; no two chosen frames, in any sample, are closer than GAP frames in the video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TURN, STRAIGHT, GAP = 15.0, 3.0, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [l.split() for l in (a.data / "data.txt").read_text().splitlines() if l.strip()]
    names = [r[0] for r in rows]
    ang = np.array([float(r[1].split(",")[0]) for r in rows])
    cls = np.where(ang < -TURN, -1, np.where(ang > TURN, 1, np.where(np.abs(ang) <= STRAIGHT, 0, 9)))
    rng = np.random.default_rng(a.seed)
    taken = np.zeros(len(rows), bool)
    for s in range(a.samples):
        d = a.out / f"s{s}"
        d.mkdir(parents=True, exist_ok=True)
        labels = {}
        for c in (-1, 0, 1):
            got = 0
            for i in rng.permutation(np.where(cls == c)[0]):
                if got == a.per_class:
                    break
                if taken[max(0, i - GAP):i + GAP + 1].any():
                    continue
                taken[i] = True
                got += 1
                link = d / names[i]
                if not link.exists():
                    link.symlink_to((a.data / names[i]).resolve())
                labels[names[i]] = int(c)
        (d / "labels.json").write_text(json.dumps(labels, indent=0))
        print(d.name, len(labels), "frames")


if __name__ == "__main__":
    main()
