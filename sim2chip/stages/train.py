"""Train the student model on synthetic crops plus the user's real photos, one model per seed.

Model: the hardware profile's backbone (MobileNetV2 for microcontrollers), pretrained on ImageNet,
quantized to int8: one model with int8 input and output for TFLite Micro, and the same with float32
input and output for phones and PCs. Validation is a held-out 10% of the training mix; the real test
set is never used here.
Count task: whole images with their points (synthetic from the points stage, real from ingest), and a model
that outputs a grid of class probabilities (see counting.py); validation also reports points F1 and count error.
Output: train/seed_<n>/{model_int8.tflite, model_float_io.tflite, history.json}, train/summary.json
"""
from __future__ import annotations

import json
import shutil

from .base import Context, Stage


class Train(Stage):
    name = "train"
    version = "2"
    runtime = "tf"
    gpu = "any"
    max_hours = 2
    uses = ("training", "hardware", "classes", "background_class", "task")
    after = ("crops", "points", "ingest")

    def run(self, ctx: Context) -> dict:
        from . import tflite_utils as tu

        if ctx.spec.task == "count":
            return self._count(ctx)
        labels = ctx.spec.labels
        size = ctx.hardware.input_size
        synth_dir = ctx.job.stage_dir("crops")
        real_dir = ctx.job.stage_dir("ingest") / "train"
        xs, ys = [], []
        if synth_dir.exists():
            x, y = tu.load_dir(synth_dir, labels, size)
            xs.append(x), ys.append(y)
        if real_dir.exists():
            xr, yr = tu.load_dir(real_dir, labels, size)
            # Repeat the few real photos so they weigh about 1/real_weight of the synthetic set.
            n_synth = len(xs[0]) if xs else 0
            rep = max(1, n_synth // (max(len(xr), 1) * ctx.spec.training.real_weight))
            xs += [xr] * rep
            ys += [yr] * rep
        if not xs:
            raise ValueError("nothing to train on: no synthetic crops and no user photos")
        import numpy as np
        x, y = np.concatenate(xs), np.concatenate(ys)
        ctx.log(f"train set {len(x)} {np.bincount(y, minlength=len(labels)).tolist()}")

        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        summary = {}
        for seed in ctx.spec.training.seeds:
            d = out / f"seed_{seed}"
            d.mkdir(parents=True)
            blob, blob_float_io, hist = tu.train(x, y, labels, ctx.hardware, ctx.spec.training.epochs, seed)
            (d / "model_int8.tflite").write_bytes(blob)
            (d / "model_float_io.tflite").write_bytes(blob_float_io)
            (d / "history.json").write_text(json.dumps(hist))
            summary[seed] = {"val_acc": round(max(hist["val_accuracy"]), 4), "kb": round(len(blob) / 1024, 1)}
            ctx.log(f"seed {seed}: val acc {summary[seed]['val_acc']}, {summary[seed]['kb']} KB")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"seeds": summary}

    def _count(self, ctx: Context) -> dict:
        import numpy as np

        from . import tflite_utils as tu

        labels, size = ctx.spec.labels, ctx.hardware.input_size
        xs, ms, ps = [], [], []
        synth = ctx.job.stage_dir("points")
        if (synth / "points.json").exists():
            x, m, _, p = tu.load_points(synth, labels, size)
            xs.append(x), ms.append(m), ps.extend(p)
        real = ctx.job.stage_dir("ingest") / "train"
        if (real / "points.json").exists():
            xr, mr, _, pr = tu.load_points(real, labels, size)
            rep = max(1, len(ps) // (max(len(xr), 1) * ctx.spec.training.real_weight))
            xs += [xr] * rep
            ms += [mr] * rep
            ps += pr * rep
        if not ps:
            raise ValueError("nothing to train on: no synthetic frames and no marked photos")
        x, m = np.concatenate(xs), np.concatenate(ms)
        ctx.log(f"train set {len(x)} images, {sum(len(p) for p in ps)} objects")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        summary = {}
        for seed in ctx.spec.training.seeds:
            d = out / f"seed_{seed}"
            d.mkdir(parents=True)
            blob, blob_float_io, hist = tu.train_count(x, m, ps, labels, ctx.hardware, ctx.spec.training.epochs, seed)
            (d / "model_int8.tflite").write_bytes(blob)
            (d / "model_float_io.tflite").write_bytes(blob_float_io)
            (d / "history.json").write_text(json.dumps(hist))
            v = hist["val_count"]
            summary[seed] = {"val_f1": v["f1"], "val_count_mae": v["count_mae"], "threshold": hist["threshold"],
                             "kb": round(len(blob) / 1024, 1)}
            ctx.log(f"seed {seed}: val F1 {v['f1']} at threshold {hist['threshold']}, count error {v['count_mae']} per image, {summary[seed]['kb']} KB")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"seeds": summary}
