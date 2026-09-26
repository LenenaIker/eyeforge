"""Accuracy of each int8 model on the user's real test photos.

Output: evaluate/seed_<n>.json (accuracy, per class, confusion, prediction per photo),
        evaluate/summary.json (mean and std)
"""
from __future__ import annotations

import json
import statistics

from .base import Context, Stage


class Evaluate(Stage):
    name = "evaluate"
    version = "2"
    runtime = "tf"
    after = ("train", "ingest")

    def enabled(self, spec) -> bool:
        ud = spec.user_data
        return bool(ud.labeled or ud.raw or ud.test) and ud.test_per_class > 0

    def run(self, ctx: Context) -> dict:
        import numpy as np

        from . import tflite_utils as tu

        labels = ctx.spec.labels
        test_dir = ctx.job.stage_dir("ingest") / "test"
        xt, yt = tu.load_dir(test_dir, labels, ctx.hardware.input_size)
        names = [f.name for c in labels for f in sorted((test_dir / c).glob("*.png"))]  # load_dir's order
        if not len(xt):
            raise ValueError("no real test photos")
        out = ctx.out(self.name)
        accs = {}
        for d in sorted(ctx.job.stage_dir("train").glob("seed_*")):
            preds = tu.predict_int8((d / "model_int8.tflite").read_bytes(), xt)
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(yt, preds):
                cm[t, p] += 1
            acc = float((preds == yt).mean())
            per_class = {c: round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i, c in enumerate(labels)}
            seed = int(d.name.split("_")[1])
            accs[seed] = acc
            predictions = [{"file": n, "truth": labels[t], "pred": labels[p]} for n, t, p in zip(names, yt, preds)]
            (out / f"{d.name}.json").write_text(json.dumps({"accuracy": round(acc, 4), "per_class": per_class,
                                                            "confusion": cm.tolist(), "labels": labels,
                                                            "predictions": predictions}, indent=2))
            ctx.log(f"seed {seed}: real int8 accuracy {acc:.3f}  {per_class}")
        vals = list(accs.values())
        summary = {"mean": round(statistics.mean(vals), 4), "std": round(statistics.pstdev(vals), 4),
                   "seeds": sorted(accs), "per_seed": {str(k): round(v, 4) for k, v in accs.items()},
                   "test_size": int(len(xt))}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ctx.log(f"real int8 accuracy {summary['mean']:.3f} +/- {summary['std']:.3f}")
        return summary
