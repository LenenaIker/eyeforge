"""Counting task helpers without TensorFlow: the model's output grid to points, and points against the truth.

The count model (see stages/tflite_utils.py) outputs a grid of class probabilities, one cell per STRIDE x
STRIDE pixels of the input, background first. A blob of neighbouring cells whose best class is not the
background is one object: its probability-weighted centre is the object's point. Points are (x, y, class)
with x and y in 0..1 of the image width and height, so they survive any resize of the image.
The same decoding runs in evaluate, in review and in any app that uses the exported model.
"""
from __future__ import annotations

from collections import Counter

STRIDE = 8          # input pixels per output cell
THRESHOLD = 0.5     # a cell is an object when its best non-background class reaches this probability
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # candidates; training picks the best on validation
MATCH_CELLS = 2.0   # a prediction matches a true point within this many cells


def grid_size(input_size: int) -> int:
    return input_size // STRIDE


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
    """Greedy nearest matching of points of the same class. Returns (true pos, false pos, false neg)."""
    pairs = sorted(((abs(p[0] - t[0]) ** 2 + abs(p[1] - t[1]) ** 2) ** 0.5, i, k)
                   for i, p in enumerate(pred) for k, t in enumerate(truth) if p[2] == t[2])
    used_p, used_t = set(), set()
    for d, i, k in pairs:
        if d <= radius and i not in used_p and k not in used_t:
            used_p.add(i)
            used_t.add(k)
    return len(used_p), len(pred) - len(used_p), len(truth) - len(used_t)


def best_threshold(grids, truths: list[list], labels: list[str], grid: int) -> tuple[float, dict]:
    """The threshold with the best points F1 on a validation set, and its scores."""
    scored = [(score([decode(g, labels, t) for g in grids], truths, grid), t) for t in THRESHOLDS]
    best, t = max(scored, key=lambda st: (st[0]["f1"], -st[0]["count_mae"]))
    return t, best


def score(preds: list[list], truths: list[list], grid: int) -> dict:
    """Detection and counting quality over a set of images.

    f1 (of the points, matched within MATCH_CELLS cells) is the headline number; count_mae is the mean
    absolute error of the number of objects per image and class, exact the share of images counted exactly."""
    tp = fp = fn = 0
    errors, exact = [], 0
    for p, t in zip(preds, truths):
        a, b, c = match(p, t, MATCH_CELLS / grid)
        tp, fp, fn = tp + a, fp + b, fn + c
        cp, ct = Counter(x[2] for x in p), Counter(x[2] for x in t)
        err = sum(abs(cp[k] - ct[k]) for k in set(cp) | set(ct))
        errors.append(err)
        exact += err == 0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = max(len(errors), 1)
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "count_mae": round(sum(errors) / n, 3), "exact": round(exact / n, 4)}
