"""Package the model: the int8 .tflite for the device, a portable copy with float32 input and output
(phones, PCs, any TFLite runtime), the labels, a C array when the device runs TFLite Micro, and a report.

The shipped model is the seed with the best validation accuracy (the real test set does not choose it).
The report lists the 3D models used and their authors, as their CC-BY licenses require.
Both .tflite files get a fixed batch of 1: Keras exports a dynamic one (-1), which runtimes that cannot resize
inputs, such as LiteRT.js in the web app's live test, refuse.
Output: export/{model_int8.tflite, model.tflite, labels.txt, report.json}, export/model_data.h (TFLite Micro)
"""
from __future__ import annotations

import json
import shutil
import struct

from .base import Context, Stage


class Export(Stage):
    name = "export"
    version = "8"  # 4: fixed batch 1 in the .tflite files (live test in the browser); 5: shipped_test; 6: advice; 7: crowd advice; 8: density output
    runtime = "cpu"
    results = ("*",)
    after = ("train", "evaluate", "qa_assets")
    uses = ("hardware", "classes", "background_class", "task")

    def run(self, ctx: Context) -> dict:
        hw = ctx.hardware
        summary = json.loads((ctx.job.stage_dir("train") / "summary.json").read_text())
        counting = ctx.spec.task == "count"
        val = "val_f1" if counting else "val_acc"
        seed = max(summary, key=lambda s: summary[s][val])
        src = ctx.job.stage_dir("train") / f"seed_{seed}"
        blob = fixed_batch((src / "model_int8.tflite").read_bytes())
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (out / "model_int8.tflite").write_bytes(blob)
        (out / "model.tflite").write_bytes(fixed_batch((src / "model_float_io.tflite").read_bytes()))
        (out / "labels.txt").write_text("\n".join(ctx.spec.labels) + "\n")
        if hw.runtime == "tflite-micro":
            (out / "model_data.h").write_text(c_array(blob, ctx.spec.labels, hw.input_size))

        kb = round(len(blob) / 1024, 1)
        if kb > hw.max_model_kb:
            ctx.log(f"WARNING: model is {kb} KB, over the {hw.max_model_kb} KB budget of {hw.name}")
        ev = ctx.job.stage_dir("evaluate") / "summary.json"
        qa = ctx.job.stage_dir("qa_assets") / "assets.json"
        credits = sorted({(a["name"], a["author"], a["license"], a["source"]) for a in json.loads(qa.read_text())}) \
            if qa.exists() else []
        report = {
            "job": ctx.spec.name, "hardware": hw.name, "labels": ctx.spec.labels, "background": ctx.spec.background_class, "input": [hw.input_size] * 2 + [3],
            "quantization": hw.quantization, "model_path": "model_int8.tflite", "model_kb": kb,
            # model.tflite: float32 RGB 0-255 in (no normalization, the model does it), probabilities out.
            "portable_model_path": "model.tflite", "portable_input": "float32 RGB 0-255, no normalization",
            "c_array_path": "model_data.h" if hw.runtime == "tflite-micro" else None,
            "seed": int(seed), "task": ctx.spec.task, "metric": "f1" if counting else "accuracy",
            "val_score": summary[seed][val], "val_accuracy": None if counting else summary[seed]["val_acc"],
            "real_test": json.loads(ev.read_text()) if ev.exists() else None,
            **({"output": count_output(hw.input_size, summary[seed].get("threshold"))} if counting else {}),
            "asset_credits": [{"name": n, "author": a, "license": lic, "source": s} for n, a, lic, s in credits],
        }
        report["shipped_test"] = shipped_test(ctx, seed)
        report["previous"] = compare_previous(ctx, seed)
        learned = ctx.job.stage_dir("label") / "teacher.json"
        taught = json.loads(learned.read_text()) if learned.exists() else {}
        # The likeliest reasons for a low count score: say them next to the score.
        advice = [taught[k]["note"] for k in ("small_objects", "crowded") if taught.get(k)]
        if advice:
            report["advice"] = advice
        (out / "report.json").write_text(json.dumps(report, indent=2))
        ctx.log(f"exported seed {seed}: {kb} KB, labels {ctx.spec.labels}")
        return {"seed": int(seed), "model_kb": kb}


def shipped_test(ctx: Context, seed) -> dict | None:
    """The shipped model's own score on the real test photos: the headline, since it is the model the user gets.
    real_test holds the mean and spread of all seeds."""
    f = ctx.job.stage_dir("evaluate") / f"seed_{seed}.json"
    if not f.exists():
        return None
    now = json.loads(f.read_text())
    return {"score": now.get("f1", now.get("accuracy")),
            **({"count_mae": now["count_mae"]} if "count_mae" in now else {})}


def compare_previous(ctx: Context, seed) -> dict | None:
    """The model this run replaced against the shipped one, both on the same test photos (evaluate)."""
    ev = ctx.job.stage_dir("evaluate")
    if not (ev / "previous.json").exists():
        return None
    prev = json.loads((ev / "previous.json").read_text())
    now = json.loads((ev / f"seed_{seed}.json").read_text()) if (ev / f"seed_{seed}.json").exists() else {}
    score = now.get("f1", now.get("accuracy"))
    return {**prev, "now": score, **({"now_count_mae": now["count_mae"]} if "count_mae" in now else {})}


def fixed_batch(blob: bytes) -> bytes:
    """The model with batch 1 instead of -1 in every tensor's shape signature. A same-size edit of the
    flatbuffer (Model.subgraphs[].tensors[].shape_signature[0]), so no TensorFlow is needed here."""
    if blob[4:8] != b"TFL3":  # not a TFLite flatbuffer: leave it alone
        return blob
    buf = bytearray(blob)
    u32 = lambda at: struct.unpack_from("<I", buf, at)[0]  # noqa: E731

    def field(table: int, i: int) -> int | None:
        vt = table - struct.unpack_from("<i", buf, table)[0]
        if 4 + 2 * i >= struct.unpack_from("<H", buf, vt)[0]:
            return None
        off = struct.unpack_from("<H", buf, vt + 4 + 2 * i)[0]
        return table + off if off else None

    def vector(table: int, i: int) -> list[int]:  # element positions
        f = field(table, i)
        if f is None:
            return []
        v = f + u32(f)
        return [v + 4 + 4 * k for k in range(u32(v))]

    model = u32(0)
    for sg in vector(model, 2):                    # Model.subgraphs
        for t in vector(sg + u32(sg), 0):           # SubGraph.tensors
            dims = vector(t + u32(t), 7)            # Tensor.shape_signature
            if dims and struct.unpack_from("<i", buf, dims[0])[0] == -1:
                struct.pack_into("<i", buf, dims[0], 1)
    return bytes(buf)


def c_array(blob: bytes, labels: list[str], size: int) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in blob[i:i + 16]) for i in range(0, len(blob), 16)]
    names = ", ".join(f'"{c}"' for c in labels)
    return (
        "// Generated by EyeForge (eyeforge.app). int8 TFLite model for TFLite Micro.\n"
        "#pragma once\n#include <stdint.h>\n\n"
        f"#define SIM2CHIP_INPUT_SIZE {size}\n#define SIM2CHIP_NUM_CLASSES {len(labels)}\n"
        f"static const char *const sim2chip_labels[] = {{{names}}};\n"
        f"alignas(16) static const uint8_t sim2chip_model[{len(blob)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"
    )


def count_output(size: int, threshold: float | None) -> dict:
    from ..counting import STRIDE, THRESHOLD, grid_size
    g, th = grid_size(size), threshold or THRESHOLD
    return {"shape": [g, g, "classes"], "stride_px": STRIDE, "threshold": th,
            "density": {"shape": [g, g, "classes - 1"]},
            "how_to_read": f"Two outputs, told apart by their channels. Points: a {g}x{g} grid of class "
                           "probabilities, labels.txt order, background first. A cell is an object when its best "
                           f"other class reaches {th}; neighbouring object cells of the same class are one object, "
                           f"at their centre. Density: a {g}x{g} grid with one channel per class (labels.txt order "
                           "without the background), the number of objects in each cell. Its sum per class, "
                           "rounded, is the count: steadier than the number of points in crowds."}
