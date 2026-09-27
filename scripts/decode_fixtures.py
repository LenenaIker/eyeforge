"""Writes the fixtures the web app's live test is checked against, so the phone counts exactly like the
evaluation:
- web/src/live/decode.fixtures.json: count grids and the points sim2chip.counting.decode gives for them
  (TypeScript port: web/src/live/decode.ts). Run it again after any change to counting.decode.
- web/src/live/resize.fixtures.json: images and Pillow's bilinear resize of them, as ingest does it
  (TypeScript port: web/src/live/resize.ts).

    python scripts/decode_fixtures.py
"""
import json
import random
from pathlib import Path

from PIL import Image

from sim2chip.counting import decode

LIVE = Path(__file__).resolve().parents[1] / "web" / "src" / "live"
OUT = LIVE / "decode.fixtures.json"


def grid(rng: random.Random, size: int, labels: list[str], blobs: int) -> list:
    """Mostly background, with a few blobs of one class each (touching blobs and weak cells included)."""
    n = len(labels)
    g = [[[0.0] * n for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            noise = [rng.random() * 0.2 for _ in range(n - 1)]
            g[i][j] = [1 - sum(noise)] + noise
    for _ in range(blobs):
        c, ci, cj, r = rng.randrange(1, n), rng.randrange(size), rng.randrange(size), rng.choice([0.6, 1.2, 2.0])
        for i in range(size):
            for j in range(size):
                d = ((i - ci) ** 2 + (j - cj) ** 2) ** 0.5
                if d <= r + 1:
                    p = max(0.0, min(0.98, 0.95 - 0.25 * d + rng.uniform(-0.05, 0.05)))
                    rest = (1 - p) / (n - 1)
                    g[i][j] = [rest] * n
                    g[i][j][c] = p
    return [[[round(v, 5) for v in cell] for cell in row] for row in g]


def main() -> None:
    rng = random.Random(7)
    cases = []
    for k in range(12):
        labels = ["empty", "pot"] if k % 3 else ["empty", "car", "bike"]
        size = 28 if k % 4 else 12
        g = grid(rng, size, labels, blobs=rng.randrange(0, 7))
        th = rng.choice([0.3, 0.5, 0.7])
        cases.append({"labels": labels, "threshold": th, "grid": g,
                      "points": [list(p) for p in decode(g, labels, th)]})
    OUT.write_text(json.dumps(cases, separators=(",", ":")))
    print(f"{len(cases)} cases, {sum(len(c['points']) for c in cases)} points -> {OUT}")

    resized = []
    for w, h, out in ((97, 61, 16), (300, 180, 28), (20, 45, 28)):  # reduce a lot, a little, and enlarge
        px = bytes(rng.randrange(256) for _ in range(w * h * 3))
        im = Image.frombytes("RGB", (w, h), px)
        resized.append({"w": w, "h": h, "out": out, "rgb": list(px),
                        "want": list(im.resize((out, out), Image.BILINEAR).tobytes())})
    (LIVE / "resize.fixtures.json").write_text(json.dumps(resized, separators=(",", ":")))
    print(f"{len(resized)} resize cases -> {LIVE / 'resize.fixtures.json'}")


if __name__ == "__main__":
    main()
