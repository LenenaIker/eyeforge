"""The reel's thumbnails from the shot renders: cNN.jpg is what the student camera saw, sNN.jpg the same picture
with Isaac's instance segmentation over it (index_NNNN.png, written by isaac_idx.py: walkscene's isaac with the
per-bin ids saved)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SIZE = (400, 300)
COLOURS = [(224, 255, 79), (79, 214, 255), (255, 92, 200), (255, 170, 60), (150, 120, 255), (90, 255, 170)]
OUT.mkdir(parents=True, exist_ok=True)
for k, v in enumerate(json.loads((RUN / "render.json").read_text())["views"]):
    rgb = np.asarray(Image.open(RUN / f"render_{v['name']}.png").convert("RGB")).astype(np.float32)
    index = np.asarray(Image.open(RUN / f"index_{v['name']}.png")).astype(np.int32)
    seg = rgb * 0.55
    for i in range(1, index.max() + 1):
        m = index == i
        if not m.any():
            continue
        c = np.array(COLOURS[(i - 1) % len(COLOURS)], np.float32)
        seg[m] = rgb[m] * 0.45 + c * 0.55
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        seg[edge] = c
    for name, im in (("c", rgb), ("s", seg)):
        Image.fromarray(im.clip(0, 255).astype(np.uint8)).resize(SIZE, Image.LANCZOS).save(OUT / f"{name}{k:02d}.jpg", quality=88)
    print(k, "bins", int(index.max()))
