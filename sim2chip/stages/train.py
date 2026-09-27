"""Train the student model on synthetic crops plus the user's real photos, one model per seed.

Model: the hardware profile's backbone (MobileNetV2 for microcontrollers), pretrained on ImageNet,
quantized to int8: one model with int8 input and output for TFLite Micro, and the same with float32
input and output for phones and PCs. Validation is a held-out 10% of the training mix; the real test
set is never used here.
Count task: whole images with their points (synthetic from the points stage, real from ingest), and a model
that outputs a grid of class probabilities (see counting.py); validation also reports points F1 and count error.
Seeds train at the same time, each in its own process on the same GPU (SIM2CHIP_PARALLEL_SEEDS, default 3):
one small model leaves most of an L40S idle, and three in a row were a third of a cloud run.
Output: train/seed_<n>/{model_int8.tflite, model_float_io.tflite, history.json}, train/summary.json
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .base import Context, Stage

PARALLEL_SEEDS = int(os.environ.get("SIM2CHIP_PARALLEL_SEEDS", "3"))
DATA_FILE = "_train_set.pkl"  # the training arrays, handed to the seed processes, removed at the end


class Train(Stage):
    name = "train"
    version = "2"
    runtime = "tf"
    gpu = "any"
    max_hours = 2
    uses = ("training", "hardware", "classes", "background_class", "task")
    after = ("crops", "points", "ingest")

    def run(self, ctx: Context) -> dict:
        count = ctx.spec.task == "count"
        data = self._count_data(ctx) if count else self._class_data(ctx)
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        seeds = ctx.spec.training.seeds
        args = {"count": count, "labels": ctx.spec.labels, "hardware": ctx.spec.hardware,
                "epochs": ctx.spec.training.epochs, "out": str(out)}
        workers = min(len(seeds), PARALLEL_SEEDS)
        if workers <= 1:
            results = [_fit_seed(args, seed, data) for seed in seeds]
        else:
            ctx.log(f"training {len(seeds)} seeds, {workers} at a time on the same GPU")
            path = out / DATA_FILE
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            del data
            try:
                # spawn, not fork: TensorFlow is already imported here and does not survive a fork.
                with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_share_gpu) as pool:
                    # Pretrained weights first, once: in a fresh container the seeds would all download the
                    # same file at once and one read it half written (Nebius, 2026-09-27).
                    pool.submit(_pretrained_weights, ctx.spec.hardware).result()
                    results = list(pool.map(_fit_seed_from_file, [args] * len(seeds), seeds,
                                            [str(path)] * len(seeds)))
            finally:
                path.unlink(missing_ok=True)
        summary = dict(zip(seeds, results))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"seeds": summary}

    @staticmethod
    def _class_data(ctx: Context):
        import numpy as np

        from . import tflite_utils as tu

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
        x, y = np.concatenate(xs), np.concatenate(ys)
        ctx.log(f"train set {len(x)} {np.bincount(y, minlength=len(labels)).tolist()}")
        return x, y

    @staticmethod
    def _count_data(ctx: Context):
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
        return x, m, ps


def _share_gpu() -> None:
    """Start of every seed process: take GPU memory as needed, else the first process takes all of it."""
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def _pretrained_weights(hardware: str) -> None:
    import tensorflow as tf

    from ..hardware import get as get_hardware
    hw = get_hardware(hardware)
    tf.keras.applications.MobileNetV2(input_shape=(hw.input_size, hw.input_size, 3), alpha=hw.alpha,
                                      include_top=False, weights="imagenet")


def _fit_seed_from_file(args: dict, seed: int, path: str) -> dict:
    """Entry point of a seed process: the arrays the parent wrote, then one seed."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return _fit_seed(args, seed, data)


def _fit_seed(args: dict, seed: int, data) -> dict:
    from ..hardware import get as get_hardware
    from . import tflite_utils as tu

    labels, hw, epochs = args["labels"], get_hardware(args["hardware"]), args["epochs"]
    d = Path(args["out"]) / f"seed_{seed}"
    d.mkdir(parents=True)
    if args["count"]:
        blob, blob_float_io, hist = tu.train_count(*data, labels, hw, epochs, seed)
        v = hist["val_count"]
        info = {"val_f1": v["f1"], "val_count_mae": v["count_mae"], "threshold": hist["threshold"],
                "kb": round(len(blob) / 1024, 1)}
        msg = (f"seed {seed}: val F1 {v['f1']} at threshold {hist['threshold']}, count error {v['count_mae']} "
               f"per image, {info['kb']} KB")
    else:
        blob, blob_float_io, hist = tu.train(*data, labels, hw, epochs, seed)
        info = {"val_acc": round(max(hist["val_accuracy"]), 4), "kb": round(len(blob) / 1024, 1)}
        msg = f"seed {seed}: val acc {info['val_acc']}, {info['kb']} KB"
    (d / "model_int8.tflite").write_bytes(blob)
    (d / "model_float_io.tflite").write_bytes(blob_float_io)
    (d / "history.json").write_text(json.dumps(hist))
    Context.log(msg)
    return info
