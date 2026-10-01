"""Counting task helpers without TensorFlow: the model's output grid to points, and points against the truth.

The count model (see student.py) outputs a grid of class probabilities, one cell per STRIDE x
STRIDE pixels of the input, background first. A blob of neighbouring cells whose best class is not the
background is one object: its probability-weighted centre is the object's point. Points are (x, y, class)
with x and y in 0..1 of the image width and height, so they survive any resize of the image.
The same decoding runs in evaluate, in review and in any app that uses the exported model.
The model also outputs a density grid (objects per cell and class): its sum is the count, steadier than the
number of points in crowds and with big objects (see student.DENSITY_SIGMA). Points say where, the
density how many.
True points may carry the object's box size, (x, y, class, w, h) in 0..1: a prediction anywhere in the box's
core (CORE of its width and height) counts as found, which is fair to big objects close to the camera.
Training could mark that core instead of one cell (TRAIN_CORE); on the pots test it split big pots into
several blobs (count error 4.3 -> 4.9 per photo), so it is off.
"""
from __future__ import annotations

from collections import Counter

STRIDE = 8          # input pixels per output cell
THRESHOLD = 0.5     # a cell is an object when its best non-background class reaches this probability
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # candidates; training picks the best on validation
MATCH_CELLS = 2.0   # a prediction matches a true point within this many cells (or within its core)
CORE = 0.4          # share of an object's box width and height where a prediction matches it
TRAIN_CORE = 0.0    # share marked as the object in training; 0: only the centre cell


def grid_size(input_size: int) -> int:
    return input_size // STRIDE


SMALL_SHARE = 0.5  # warn when at least this share of the marked objects is smaller than one output cell
CROWDED = 30         # objects in one photo from which the model misses most of them
CROWDED_SHARE = 0.05  # warn from this share of crowded photos: on geese 7 photos in 80 made most of the count error


def small_objects(boxes: list[list[float]], input_size: int) -> dict | None:
    """boxes: [w, h] of every marked object, as a share of the photo. The model sees the photo squeezed to
    input_size and looks for objects on a grid of STRIDE px cells: an object smaller than a cell is a few pixels
    the model can hardly tell from anything. Returns what to tell the user when most objects are like that
    (seen 2026-10-01: pots in street photos, median 6 px at 224, F1 9%). A bigger input did not help: the same
    photos at 320, 448 and 640 px scored lower (the cells then see too little around big, close objects)."""
    import statistics
    if not boxes:
        return None
    px = [(w * h) ** 0.5 * input_size for w, h in boxes]
    share = sum(p < STRIDE for p in px) / len(px)
    if share < SMALL_SHARE:
        return None
    return {"share": round(share, 2), "median_px": round(statistics.median(px), 1), "input": input_size,
            "cell_px": STRIDE,
            "note": (f"{round(share * 100)}% of the marked objects are smaller than {STRIDE} px once the photo is "
                     f"shrunk to the model's {input_size} px (median {statistics.median(px):.0f} px): the model can "
                     "hardly see them. Photos taken closer, where each object fills more of the picture, count "
                     "much better.")}


def decode(probs, labels: list[str], threshold: float = THRESHOLD) -> list[tuple[float, float, str]]:
    """probs: (rows, cols, classes) with the background first. Returns one point per blob of cells."""
    rows, cols, n = len(probs), len(probs[0]), len(labels)
    best = [[max(range(1, n), key=lambda c: probs[i][j][c]) for j in range(cols)] for i in range(rows)]
    on = [[probs[i][j][best[i][j]] >= threshold for j in range(cols)] for i in range(rows)]
    seen = [[False] * cols for _ in range(rows)]
    points = []
    for i in range(rows):
        for j in range(cols):
            if not on[i][j] or seen[i][j]:
                continue
            cls, todo, blob = best[i][j], [(i, j)], []
            seen[i][j] = True
            while todo:
                a, b = todo.pop()
                blob.append((a, b, float(probs[a][b][cls])))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y, x = a + da, b + db
                    if 0 <= y < rows and 0 <= x < cols and on[y][x] and not seen[y][x] and best[y][x] == cls:
                        seen[y][x] = True
                        todo.append((y, x))
            w = sum(p for _, _, p in blob)
            points.append((round(sum((b + 0.5) * p for _, b, p in blob) / w / cols, 4),
                           round(sum((a + 0.5) * p for a, _, p in blob) / w / rows, 4), labels[cls]))
    return points


def match(pred: list, truth: list, radius: float) -> tuple[int, int, int]:
    """Greedy nearest matching of points of the same class. Returns (true pos, false pos, false neg).
    A true point with a box size also matches anywhere within its core."""
    reach = [max(radius, CORE / 2 * max(t[3], t[4])) if len(t) >= 5 else radius for t in truth]
    pairs = sorted(((abs(p[0] - t[0]) ** 2 + abs(p[1] - t[1]) ** 2) ** 0.5, i, k)
                   for i, p in enumerate(pred) for k, t in enumerate(truth) if p[2] == t[2])
    used_p, used_t = set(), set()
    for d, i, k in pairs:
        if d <= reach[k] and i not in used_p and k not in used_t:
            used_p.add(i)
            used_t.add(k)
    return len(used_p), len(pred) - len(used_p), len(truth) - len(used_t)


def threshold_curve(grids, truths: list[list], labels: list[str], grid: int,
                    counts: list[dict] | None = None) -> dict[float, dict]:
    """Scores on a validation set at every candidate threshold (counts: see score; they do not depend on it)."""
    return {t: score([decode(g, labels, t) for g in grids], truths, grid, counts) for t in THRESHOLDS}


def best_threshold(grids, truths: list[list], labels: list[str], grid: int) -> tuple[float, dict]:
    """The threshold with the best points F1 on a validation set, and its scores."""
    curve = threshold_curve(grids, truths, labels, grid)
    t = max(curve, key=lambda t: (curve[t]["f1"], -curve[t]["count_mae"]))
    return t, curve[t]


STEADY_MARGIN = 0.1  # relative F1 given up to stay near THRESHOLD


def steady_threshold(f1: dict[float, float]) -> float:
    """With a few validation photos the best threshold is mostly noise and lands on the extremes: pots, 19 photos,
    cross-validation chose 0.7 to 0.9 and found 5 to 27% of the test pots, while 0.5 scored twice as well
    (2026-10-01). Among the thresholds within STEADY_MARGIN of the best F1, the one nearest THRESHOLD."""
    best = max(f1.values())
    near = [t for t, v in f1.items() if v >= best * (1 - STEADY_MARGIN)]
    return min(near, key=lambda t: (abs(t - THRESHOLD), -f1[t]))


def crowded(counts: list[int]) -> dict | None:
    """counts: marked objects in every photo. Crowds are where counting is weakest. Measured on FSC-147 (2026-10-01,
    dataset labels, 224 px, count from the density output): photos with more than 30 objects were off by 15 to 70%
    (geese about 40%, apples 50%, strawberries 15 to 30%, bread rolls 70%); with the points alone, geese crowds
    were off by two thirds. Returns what to tell the user when many photos are that crowded."""
    if not counts:
        return None
    share = sum(c > CROWDED for c in counts) / len(counts)
    if share < CROWDED_SHARE:
        return None
    return {"share": round(share, 2), "max": max(counts), "limit": CROWDED,
            "note": (f"{round(share * 100)}% of the marked photos have more than {CROWDED} objects (up to "
                     f"{max(counts)}). In crowds like these, expect the count to be off by a third to a half. "
                     "A narrower view, where each photo holds fewer objects, counts much better.")}


def density_counts(density, labels: list[str]) -> dict[str, int]:
    """density: (rows, cols, classes - 1), background left out -> the count of every class, rounded as shown."""
    import numpy as np
    total = np.asarray(density, dtype=np.float64).sum((0, 1))
    return {lab: int(round(float(v))) for lab, v in zip(labels[1:], total)}


def score(preds: list[list], truths: list[list], grid: int, counts: list[dict] | None = None) -> dict:
    """Detection and counting quality over a set of images.

    f1 (of the points, matched within MATCH_CELLS cells) is the headline number; count_mae is the mean
    absolute error of the number of objects per image and class, exact the share of images counted exactly.
    counts: the count of every image by class (density_counts); without them the points are counted."""
    tp = fp = fn = 0
    errors, exact = [], 0
    for i, (p, t) in enumerate(zip(preds, truths)):
        a, b, c = match(p, t, MATCH_CELLS / grid)
        tp, fp, fn = tp + a, fp + b, fn + c
        cp = Counter(counts[i]) if counts is not None else Counter(x[2] for x in p)
        ct = Counter(x[2] for x in t)
        err = sum(abs(cp[k] - ct[k]) for k in set(cp) | set(ct))
        errors.append(err)
        exact += err == 0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = max(len(errors), 1)
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "count_mae": round(sum(errors) / n, 3), "exact": round(exact / n, 4)}
