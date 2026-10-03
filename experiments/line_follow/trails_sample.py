"""Sample frames of the IDSIA forest trails dataset (Giusti et al. 2016, CC BY 4.0, zenodo.org/records/7755648)
into one flat folder of links, with their labels.

    python experiments/line_follow/trails_sample.py DATA OUT --folders 011,012,013 --per-camera 100

DATA/<folder>/videos/{lc,sc,rc}/*.frames/frameN.jpg. The left camera looks 30 degrees left of the walking
direction, so its frames are "turn right" (TR, +1); the centre camera's are "go straight" (GS, 0); the right
camera's "turn left" (TL, -1). Frames are taken evenly along each camera's sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABEL = {"lc": 1, "sc": 0, "rc": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--folders", required=True)
    ap.add_argument("--per-camera", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for folder in a.folders.split(","):
        for cam, lab in LABEL.items():
            frames = sorted((a.data / folder / "videos" / cam).glob("*.frames/frame*.jpg"),
                            key=lambda f: (f.parent.name, int(f.stem[5:])))
            if not frames:
                continue
            for f in [frames[i] for i in np.linspace(0, len(frames) - 1, a.per_camera).astype(int)]:
                name = f"{folder}_{cam}_{f.parent.stem}_{f.stem}.jpg"
                link = a.out / name
                if not link.exists():
                    link.symlink_to(f.resolve())
                labels[name] = lab
    (a.out / "labels.json").write_text(json.dumps(labels, indent=0))
    print(len(labels), "frames", {v: list(labels.values()).count(v) for v in (-1, 0, 1)})


if __name__ == "__main__":
    main()
