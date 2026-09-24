"""Accuracy of the teacher (label stage) against ground truth.

    python scripts/label_agreement.py data/jobs/teacher_bench data/review/teacher_bench/ground_truth.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(job: Path, gt_file: Path) -> None:
    verdicts = json.loads((job / "label" / "verdicts.json").read_text())
    gt = json.loads(gt_file.read_text())
    labels = sorted(set(gt.values()))
    n = len(gt)
    assigned = {k: v["assigned"] for k, v in verdicts.items()}
    unsure = sum(1 for k in gt if assigned.get(k, "unsure") == "unsure")
    decided = [k for k in gt if assigned.get(k, "unsure") != "unsure"]
    right = sum(assigned[k] == gt[k] for k in decided)
    print(f"photos {n}, set aside as unsure {unsure} ({unsure / n:.0%})")
    print(f"accuracy on the rest {right}/{len(decided)} = {right / max(len(decided), 1):.1%}")
    print(f"usable correct labels {right}/{n} = {right / n:.1%}")
    print("\nconfusion (rows: truth, cols: teacher):")
    cols = labels + ["unsure"]
    print(" " * 16 + " ".join(f"{c[:8]:>8s}" for c in cols))
    for t in labels:
        c = Counter(assigned.get(k, "unsure") for k in gt if gt[k] == t)
        print(f"{t:16s}" + " ".join(f"{c.get(x, 0):8d}" for x in cols))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
