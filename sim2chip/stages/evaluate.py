"""Accuracy of each int8 model on the user's real test photos.

Output: evaluate/seed_<n>.json (accuracy, per class, confusion, prediction per photo),
        evaluate/summary.json (mean and std)
Count task: points F1 as the headline (`metric`), plus count error and exact counts, and the predicted and
true points of every photo.
"""
from __future__ import annotations

import json
import statistics

from .base import Context, Stage


class Evaluate(Stage):
    name = "evaluate"
    version = "2"
    runtime = "tf"
    uses = ("task",)
    after = ("train", "ingest")

    def enabled(self, spec) -> bool:
        ud = spec.user_data
        return bool(ud.labeled or ud.raw or ud.test) and ud.test_per_class > 0

    def run(self, ctx: Context) -> dict:
        import numpy as np

        from . import tflite_utils as tu

        if ctx.spec.task == "count":
            return self._count(ctx)
        labels = ctx.spec.labels
        test_dir = ctx.job.stage_dir("ingest") / "test"
        xt, yt = tu.load_dir(test_dir, labels, ctx.hardware.input_size)
        names = [f.name for f, _ in tu.list_dir(test_dir, labels)]
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
        summary = {"metric": "accuracy", "mean": round(statistics.mean(vals), 4), "std": round(statistics.pstdev(vals), 4),
                   "seeds": sorted(accs), "per_seed": {str(k): round(v, 4) for k, v in accs.items()},
                   "test_size": int(len(xt))}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ctx.log(f"real int8 accuracy {summary['mean']:.3f} +/- {summary['std']:.3f}")
        return summary

    def _count(self, ctx: Context) -> dict:
        import numpy as np

        from .. import counting
        from . import tflite_utils as tu

        labels, size = ctx.spec.labels, ctx.hardware.input_size
        test_dir = ctx.job.stage_dir("ingest") / "test"
        xt, _, names, truth = tu.load_points(test_dir, labels, size)
        if not len(xt):
            raise ValueError("no real test photos")
        info = json.loads((test_dir / "points.json").read_text())
        out = ctx.out(self.name)
        grid = counting.grid_size(size)
        scores = {}
        chosen = json.loads((ctx.job.stage_dir("train") / "summary.json").read_text())
        for d in sorted(ctx.job.stage_dir("train").glob("seed_*")):
            th = chosen[d.name.split("_")[1]].get("threshold", counting.THRESHOLD)
            preds = [counting.decode(g, labels, th) for g in tu.predict_int8_grid((d / "model_int8.tflite").read_bytes(), xt)]
            sc = counting.score(preds, truth, grid)
            seed = int(d.name.split("_")[1])
            scores[seed] = sc
            predictions = [{"file": n, "key": info[n]["key"], "source": info[n]["source"],
                            "truth": [list(t) for t in tr], "pred": [list(p) for p in pr]}
                           for n, tr, pr in zip(names, truth, preds)]
            (out / f"{d.name}.json").write_text(json.dumps({**sc, "labels": labels, "predictions": predictions},
                                                           indent=1))
            ctx.log(f"seed {seed}: real int8 {sc}")
        f1 = [s["f1"] for s in scores.values()]
        summary = {"metric": "f1", "mean": round(statistics.mean(f1), 4), "std": round(statistics.pstdev(f1), 4),
                   "count_mae": round(float(np.mean([s["count_mae"] for s in scores.values()])), 3),
                   "exact": round(float(np.mean([s["exact"] for s in scores.values()])), 4),
                   "seeds": sorted(scores), "per_seed": {str(k): v["f1"] for k, v in scores.items()},
                   "test_size": int(len(xt)), "objects": sum(len(t) for t in truth)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ctx.log(f"real int8 points F1 {summary['mean']:.3f} +/- {summary['std']:.3f}, "
                f"count error {summary['count_mae']} per photo")
        return summary
