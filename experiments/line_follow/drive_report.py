"""All path-direction teachers on the repeated road samples (drive_sample.py): accuracy per sample, mean and spread.

    python experiments/line_follow/drive_report.py runs/line_follow/drive [--samples 3]

VLMs: trails_vlm.py answers (vlm_s<i>/). SAM 3 + geometry: masks_s<i>/ with the rules of trails_eval.py. Molmo2:
molmo2_s<i>.json points. For the two that give an x, the threshold k of each sample is fitted on the other samples
(never on the sample it is scored on).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from trails_eval import RULES, best_k, classify, rule_x, trail_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    S = [f"s{i}" for i in range(a.samples)]
    labels = {s: json.loads((a.root / s / "labels.json").read_text()) for s in S}
    names = {s: sorted(labels[s]) for s in S}
    y = {s: np.array([labels[s][n] for n in names[s]]) for s in S}
    acc: dict[str, list[float]] = {}
    extra: dict[str, list[float]] = {}

    for f in sorted((a.root / f"vlm_{S[0]}").glob("*.json")):
        if f.name == "summary.json":
            continue
        model = f.stem.replace("__", "/")
        for s in S:
            ans = json.loads((a.root / f"vlm_{s}" / f.name).read_text()) if (a.root / f"vlm_{s}" / f.name).exists() \
                else {}
            pred = np.array([9 if ans.get(n) is None else ans[n] for n in names[s]])
            acc.setdefault(model, []).append(float((pred == y[s]).mean()))
            extra.setdefault(model, []).append(float((pred == 9).mean()))

    xs: dict[str, dict[str, np.ndarray]] = {}
    for s in S:
        masks = a.root / f"masks_{s}"
        if masks.exists():
            ms = [trail_mask(masks, n) for n in names[s]]
            for r in RULES:
                xs.setdefault(f"SAM 3 + {r}", {})[s] = np.array(
                    [np.nan if m is None or (v := rule_x(m, r)) is None else v for m in ms])
        mj = a.root / f"molmo2_{s}.json"
        if mj.exists():
            pts = json.loads(mj.read_text())["x"]
            xs.setdefault("Molmo2-8B (points)", {})[s] = np.array(
                [np.nan if pts.get(n) is None else pts[n] for n in names[s]])
    for name, per in xs.items():
        if len(per) < len(S):
            continue
        for s in S:
            others = [o for o in S if o != s]
            k = best_k(np.concatenate([per[o] for o in others]), np.concatenate([y[o] for o in others]))
            acc.setdefault(name, []).append(float((classify(per[s], k) == y[s]).mean()))
            extra.setdefault(name, []).append(float(np.isnan(per[s]).mean()))

    print(f"{'teacher':32} {'mean acc':>8} {'sd':>6}  per sample           no answer")
    for name, v in sorted(acc.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{name:32} {np.mean(v):8.3f} {np.std(v):6.3f}  {' '.join(f'{x:.3f}' for x in v)}   "
              f"{np.mean(extra[name]):.3f}")
    (a.root / "report.json").write_text(json.dumps({k: {"acc": v, "no_answer": extra[k]} for k, v in acc.items()},
                                                   indent=1))


if __name__ == "__main__":
    main()
