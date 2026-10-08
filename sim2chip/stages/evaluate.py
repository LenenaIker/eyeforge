"""Accuracy of each int8 model on the user's real test photos.

Output: evaluate/seed_<n>.json (accuracy, per class, confusion, prediction per photo),
        evaluate/summary.json (mean and std)
Count task: points F1 as the headline (`metric`), plus count error and exact counts, and the predicted and
true points of every photo.
When the job had a model before this run (input/previous/, see workspace.snapshot), that model is scored on
the same test photos: evaluate/previous.json {version, score, count_mae, then} or {version, skipped: why}. then is
its score when it was trained: it differs from score when the test photos or their labels changed since.
"""
from __future__ import annotations

import json
import statistics

from ..workspace import PREVIOUS
from .base import Context, Stage
from ..tasks import recipe_of, task_of


def previous_model(ctx: Context, size: int | None = None):
    """(version, model bytes, its report) of the model this run replaces, or (version, None, why not). size: the
    input side of this run's model, when it is not the profile's (a count detector)."""
    d = ctx.job.input / PREVIOUS
    if not (d / "report.json").exists():
        return None, None, None
    n = json.loads((d / "meta.json").read_text())["n"] if (d / "meta.json").exists() else None
    rep = json.loads((d / "report.json").read_text())
    model = d / rep.get("model_path", "model_int8.tflite")  # a count detector ships model.tflite only
    if not model.exists():
        return None, None, None
    if rep.get("labels") != ctx.spec.labels or rep.get("task") != ctx.spec.task or \
            (rep.get("input") or [None])[0] != (size or ctx.hardware.input_size):
        return n, None, "the setup changed (classes, task or input size), the scores do not compare"
    return n, model.read_bytes(), rep


def score_then(rep: dict) -> dict:
    """{then: the shipped model's score on its own test photos}, or nothing for reports without it."""
    then = ((rep.get("real_test") or {}).get("per_seed") or {}).get(str(rep.get("seed")))
    return {} if then is None else {"then": then}


class Evaluate(Stage):
    name = "evaluate"
    version = "5"  # 3: the previous model's score when it was trained; 4: count from the density output; 5: count detector
    runtime = "tf"
    uses = ("task",)
    after = ("train", "ingest")
    reads_input = True  # the previous model, to compare

    def enabled(self, spec) -> bool:
        ud = spec.user_data
        return bool(ud.labeled or ud.raw or ud.test) and ud.test_per_class > 0

    def run(self, ctx: Context) -> dict:
        import numpy as np

        from .. import student as tu

        if task_of(ctx.spec).points:
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
        n, model, rep = previous_model(ctx)
        if n is not None:
            prev = {"version": n, "skipped": rep} if model is None else \
                {"version": n, "score": round(float((tu.predict_int8(model, xt) == yt).mean()), 4), **score_then(rep)}
            (out / "previous.json").write_text(json.dumps(prev, indent=1))
            ctx.log(f"previous model (v{n}) on the same test photos: {prev}")
        vals = list(accs.values())
        summary = {"metric": task_of(ctx.spec).metric, "mean": round(statistics.mean(vals), 4), "std": round(statistics.pstdev(vals), 4),
                   "seeds": sorted(accs), "per_seed": {str(k): round(v, 4) for k, v in accs.items()},
                   "test_size": int(len(xt))}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ctx.log(f"real int8 accuracy {summary['mean']:.3f} +/- {summary['std']:.3f}")
        return summary

    def _count(self, ctx: Context) -> dict:
        import numpy as np

        from .. import counting, detector
        from .. import student as tu

        labels = ctx.spec.labels
        test_dir = ctx.job.stage_dir("ingest") / "test"
        info = json.loads((test_dir / "points.json").read_text())
        names = sorted(info)
        if not names:
            raise ValueError("no real test photos")
        truth = [[tuple(q) for q in info[n]["points"]] for n in names]  # with box sizes: a hit anywhere in the core counts
        out = ctx.out(self.name)
        chosen = json.loads((ctx.job.stage_dir("train") / "summary.json").read_text())
        recipes = {s: recipe_of(r) for s, r in chosen.items()}
        boxes = any(r.boxes for r in recipes.values())
        # A detector sees the full size photos at its own size; the grid head the ingest copies at the profile's.
        size = next(iter(chosen.values())).get("input", ctx.hardware.input_size)
        grid = counting.grid_size(size)
        if boxes:
            xt = detector.load([ctx.job.root / info[n]["source"] for n in names], size)
        else:
            xt = tu.load_points(test_dir, labels, size)[0]

        def run(blob: bytes, th: float, as_boxes: bool):
            """Points and, from a density output, counts of every test photo."""
            if as_boxes:
                return [detector.decode(o, labels, th) for o in detector.predict(blob, xt)], None
            grids, dens = tu.predict_int8_count(blob, xt)
            return [counting.decode(g, labels, th) for g in grids], \
                None if dens is None else [counting.density_counts(x, labels) for x in dens]

        scores = {}
        for d in sorted(ctx.job.stage_dir("train").glob("seed_*")):
            key = d.name.split("_")[1]
            th = chosen[key].get("threshold", counting.THRESHOLD)
            blob = (d / recipes[key].model_file).read_bytes()
            preds, counts = run(blob, th, boxes)
            sc = counting.score(preds, truth, grid, counts)
            seed = int(d.name.split("_")[1])
            scores[seed] = sc
            predictions = [{"file": n, "key": info[n]["key"], "source": info[n]["source"],
                            "truth": [list(t) for t in tr], "pred": [list(p) for p in pr],
                            **({"count": counts[i]} if counts else {})}
                           for i, (n, tr, pr) in enumerate(zip(names, truth, preds))]
            (out / f"{d.name}.json").write_text(json.dumps({**sc, "labels": labels, "predictions": predictions},
                                                           indent=1))
            ctx.log(f"seed {seed}: real test {sc}")
        n, model, rep = previous_model(ctx, size)
        if n is not None:
            if model is None:
                prev = {"version": n, "skipped": rep}
            else:
                output = rep.get("output") or {}
                preds, counts = run(model, output.get("threshold", counting.THRESHOLD), output.get("kind") == "boxes")
                sc = counting.score(preds, truth, grid, counts)
                prev = {"version": n, "score": sc["f1"], "count_mae": sc["count_mae"], **score_then(rep)}
            (out / "previous.json").write_text(json.dumps(prev, indent=1))
            ctx.log(f"previous model (v{n}) on the same test photos: {prev}")
        f1 = [s["f1"] for s in scores.values()]
        summary = {"metric": task_of(ctx.spec).metric, "mean": round(statistics.mean(f1), 4), "std": round(statistics.pstdev(f1), 4),
                   "count_mae": round(float(np.mean([s["count_mae"] for s in scores.values()])), 3),
                   "exact": round(float(np.mean([s["exact"] for s in scores.values()])), 4),
                   "seeds": sorted(scores), "per_seed": {str(k): v["f1"] for k, v in scores.items()},
                   "test_size": len(names), "objects": sum(len(t) for t in truth)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ctx.log(f"real test points F1 {summary['mean']:.3f} +/- {summary['std']:.3f}, "
                f"count error {summary['count_mae']} per photo")
        return summary
