"""Train the student model on synthetic crops plus the user's real photos, one model per seed.

Model: the hardware profile's backbone (MobileNetV2 for microcontrollers), pretrained on ImageNet,
quantized to int8: one model with int8 input and output for TFLite Micro, and the same with float32
input and output for phones and PCs. The recipe follows the data, not the spec: with only real photos and fewer
than HEAD_ONLY_BELOW per class, the backbone stays frozen and only the last layer learns, scored by cross-
validation (student.train_head; for count, student.train_count_head, with or without synthetic frames and
its threshold also chosen on the real photos); otherwise the whole network is fine-tuned and validated on a held-out 10%
of the training mix. The real test set is never used here.
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
# Real photos per class under which only the last layer learns (with no synthetic images). Fine-tuning the
# whole network needs more than a few dozen: on 35 lawn photos it scored 56% against 80% for the last layer.
HEAD_ONLY_BELOW = 100
DATA_FILE = "_train_set.pkl"  # the training arrays, handed to the seed processes, removed at the end


class Train(Stage):
    name = "train"
    version = "7"  # 3: blur from the hardware profile; 4: turns and tilts, last layer only with few photos; 5: same for count; 6: one steady count threshold; 7: count density output
    runtime = "tf"
    gpu = "any"
    max_hours = 2
    uses = ("training", "hardware", "classes", "background_class", "task")
    after = ("crops", "points", "ingest")

    def run(self, ctx: Context) -> dict:
        count = ctx.spec.task == "count"
        data, head = self._count_data(ctx) if count else self._class_data(ctx)
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        seeds = ctx.spec.training.seeds
        args = {"count": count, "labels": ctx.spec.labels, "hardware": ctx.spec.hardware,
                "real_weight": ctx.spec.training.real_weight,
                "epochs": ctx.spec.training.epochs, "orientation": ctx.spec.training.orientation, "head": head, "out": str(out)}
        if head and count:
            ctx.log(f"few real photos (under {HEAD_ONLY_BELOW} per class): the pretrained network stays frozen and "
                    "only its last layers learn; threshold and score by cross-validation on the real photos")
        elif head:
            ctx.log(f"few photos (under {HEAD_ONLY_BELOW} per class, no synthetic): the pretrained network stays "
                    "frozen and only its last layer learns, scored by cross-validation")
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
        if all("cv_curve" in r for r in results):
            common_threshold(ctx, summary)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return {"seeds": summary}

    @staticmethod
    def _class_data(ctx: Context):
        import numpy as np

        from .. import student as tu

        labels = ctx.spec.labels
        size = ctx.hardware.input_size
        synth_dir = ctx.job.stage_dir("crops")
        real_dir = ctx.job.stage_dir("ingest") / "train"
        xs, ys, n_synth = [], [], 0
        if synth_dir.exists():
            x, y = tu.load_dir(synth_dir, labels, size)
            xs.append(x), ys.append(y)
            n_synth = len(x)
        if real_dir.exists():
            xr, yr = tu.load_dir(real_dir, labels, size)
            # Repeat the few real photos so they weigh about 1/real_weight of the synthetic set.
            rep = max(1, n_synth // (max(len(xr), 1) * ctx.spec.training.real_weight))
            xs += [xr] * rep
            ys += [yr] * rep
        if not xs:
            raise ValueError("nothing to train on: no synthetic crops and no user photos")
        x, y = np.concatenate(xs), np.concatenate(ys)
        ctx.log(f"train set {len(x)} {np.bincount(y, minlength=len(labels)).tolist()}")
        head = not n_synth and len(x) < HEAD_ONLY_BELOW * len(labels)
        return (x, y), head

    @staticmethod
    def _count_data(ctx: Context):
        """Synthetic frames and real photos with their points. With few real photos they stay apart for the head
        recipe (its cross-validation holds out real photos only); otherwise one mix with the real repeated."""
        import numpy as np

        from .. import student as tu

        labels, size = ctx.spec.labels, ctx.hardware.input_size
        empty = (np.zeros((0, size, size, 3), np.uint8), np.zeros((0, size, size), np.uint8), [])
        xs, ms, ps = empty
        synth = ctx.job.stage_dir("points")
        if (synth / "points.json").exists():
            xs, ms, _, ps = tu.load_points(synth, labels, size)
        xr, mr, pr = empty
        real = ctx.job.stage_dir("ingest") / "train"
        if (real / "points.json").exists():
            xr, mr, _, pr = tu.load_points(real, labels, size)
        if not len(xs) and not len(xr):
            raise ValueError("nothing to train on: no synthetic frames and no marked photos")
        objects = sum(len(p) for p in ps + pr)
        ctx.log(f"train set {len(xs)} synthetic and {len(xr)} real images, {objects} objects")
        # Cross-validation needs at least two real photos.
        if 2 <= len(xr) < HEAD_ONLY_BELOW * max(1, len(labels) - 1):
            return (xs, ms, xr, mr, pr), True
        rep = max(1, len(ps) // (max(len(xr), 1) * ctx.spec.training.real_weight))
        return (np.concatenate([xs] + [xr] * rep), np.concatenate([ms] + [mr] * rep), ps + pr * rep), False


def common_threshold(ctx: Context, summary: dict) -> None:
    """Count with few photos: one threshold for every seed, from their cross-validation curves averaged. Each
    seed alone saw the same few photos and picked 0.7, 0.8 or 0.9 by chance (2026-10-01); the average is steadier,
    and the seed is then picked by its score at that same threshold."""
    from .. import counting

    ts = list(next(iter(summary.values()))["cv_curve"])
    mean = {float(t): sum(r["cv_curve"][t]["f1"] for r in summary.values()) / len(summary) for t in ts}
    th = counting.steady_threshold(mean)
    for r in summary.values():
        at = r["cv_curve"][str(th)]
        r.update(threshold=th, val_f1=at["f1"], val_count_mae=at["count_mae"])
    ctx.log(f"threshold {th} for every seed (cross-validated F1 by threshold, mean of the seeds: "
            + ", ".join(f"{t} {v:.2f}" for t, v in mean.items()) + ")")


def _share_gpu() -> None:
    """Start of every seed process: take GPU memory as needed, else the first process takes all of it."""
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def _pretrained_weights(hardware: str) -> None:
    import tensorflow as tf

    from ..hardware import get as get_hardware
    hw = get_hardware(hardware)
    if hw.backbone != "mobilenetv2":  # build_model says what is missing
        return
    tf.keras.applications.MobileNetV2(input_shape=(hw.input_size, hw.input_size, 3), alpha=hw.alpha,
                                      include_top=False, weights="imagenet")


def _fit_seed_from_file(args: dict, seed: int, path: str) -> dict:
    """Entry point of a seed process: the arrays the parent wrote, then one seed."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return _fit_seed(args, seed, data)


def _fit_seed(args: dict, seed: int, data) -> dict:
    from ..hardware import get as get_hardware
    from .. import student as tu

    labels, hw, epochs = args["labels"], get_hardware(args["hardware"]), args["epochs"]
    d = Path(args["out"]) / f"seed_{seed}"
    d.mkdir(parents=True)
    if args["count"]:
        if args.get("head"):
            blob, blob_float_io, hist = tu.train_count_head(*data, labels, hw, seed, args.get("real_weight", 4),
                                                            args.get("orientation"))
        else:
            blob, blob_float_io, hist = tu.train_count(*data, labels, hw, epochs, seed, args.get("orientation"))
        v = hist["val_count"]
        info = {"val_f1": v["f1"], "val_count_mae": v["count_mae"], "threshold": hist["threshold"],
                "recipe": hist.get("recipe", "fine_tune"), "kb": round(len(blob) / 1024, 1),
                **({"cv_curve": hist["cv_curve"]} if "cv_curve" in hist else {})}
        msg = (f"seed {seed}: val F1 {v['f1']} at threshold {hist['threshold']}, count error {v['count_mae']} "
               f"per image, {info['kb']} KB")
    elif args.get("head"):
        blob, blob_float_io, hist = tu.train_head(*data, labels, hw, seed, args.get("orientation"))
        # The cross-validated accuracy stands in for validation (export picks the seed by it).
        info = {"val_acc": hist.get("cv_accuracy", 0.0), "recipe": "head", "kb": round(len(blob) / 1024, 1)}
        msg = (f"seed {seed}: last layer only, cross-validated acc {hist.get('cv_accuracy', 'n/a')} "
               f"{hist.get('cv_per_class', '')}, {info['kb']} KB")
    else:
        blob, blob_float_io, hist = tu.train(*data, labels, hw, epochs, seed, args.get("orientation"))
        info = {"val_acc": round(max(hist["val_accuracy"]), 4), "recipe": "fine_tune",
                "kb": round(len(blob) / 1024, 1)}
        msg = f"seed {seed}: val acc {info['val_acc']}, {info['kb']} KB"
    (d / "model_int8.tflite").write_bytes(blob)
    (d / "model_float_io.tflite").write_bytes(blob_float_io)
    (d / "history.json").write_text(json.dumps(hist))
    Context.log(msg)
    return info
