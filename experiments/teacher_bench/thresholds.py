"""The user's preference picks the threshold: find more (F2), balanced (F1) or fewer false marks (F0.5).

    python experiments/teacher_bench/thresholds.py

The threshold is chosen on the calibration photos only (split.json, from run_sam3_exemplars.py) and scored on the
rest; models without scores are scored on the same test photos for comparison.
"""
from __future__ import annotations

import json

import common
from report import match

PREFS = {"find more (F2)": 2.0, "balanced (F1)": 1.0, "fewer false (F0.5)": 0.5}


def counts(preds, photos, th=0.0):
    tp = fp = fn = 0
    truth = common.truth()
    for p in photos:
        boxes = [b for b in preds.get(p, []) if b.get("score", 1.0) >= th]
        for b in boxes:
            (x0, x1), (y0, y1) = sorted(b["box"][::2]), sorted(b["box"][1::2])
            b["box"] = [x0, y0, x1, y1]
        found, used = match(boxes, truth[p])
        a = sum(f is not None for f in found)
        tp, fp, fn = tp + a, fp + used.count(False), fn + len(truth[p]) - a
    return tp, fp, fn


def fbeta(tp, fp, fn, beta):
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return (1 + beta ** 2) * p * r / max(1e-9, beta ** 2 * p + r), p, r


def main():
    split = json.loads((common.OUT / "split.json").read_text())
    calib, test = split["calibration"], split["test"]
    print(f"calibration {len(calib)} photos, test {len(test)} photos "
          f"({sum(len(common.truth()[p]) for p in test)} objects)\n")
    print(f"{'model':<16}{'preference':<20}{'threshold':>10}{'F1':>7}{'F2':>7}{'F0.5':>7}{'prec':>7}{'recall':>8}")
    rows = []
    for name in ("sam3", "sam3_ex"):
        preds = json.loads((common.OUT / f"{name}.json").read_text())["preds"]
        for pref, beta in PREFS.items():
            th = max((t / 100 for t in range(20, 96, 5)), key=lambda t: fbeta(*counts(preds, calib, t), beta)[0])
            c = counts(preds, test, th)
            f1, p, r = fbeta(*c, 1)
            rows.append((name, pref, th, f1, fbeta(*c, 2)[0], fbeta(*c, 0.5)[0], p, r))
    for name in ("glm", "glm_short_px", "locate", "locate_v2", "qwen_v2"):
        f = common.OUT / f"{name}.json"
        if f.exists():
            c = counts(json.loads(f.read_text())["preds"], test)
            f1, p, r = fbeta(*c, 1)
            rows.append((name, "-", None, f1, fbeta(*c, 2)[0], fbeta(*c, 0.5)[0], p, r))
    for name, pref, th, f1, f2, f05, p, r in rows:
        print(f"{name:<16}{pref:<20}{'' if th is None else th:>10}{f1:>7.3f}{f2:>7.3f}{f05:>7.3f}{p:>7.3f}{r:>8.3f}")
    (common.OUT / "thresholds.json").write_text(json.dumps(
        [dict(zip(("model", "preference", "threshold", "f1", "f2", "f05", "precision", "recall"), r)) for r in rows],
        indent=1))


if __name__ == "__main__":
    main()
