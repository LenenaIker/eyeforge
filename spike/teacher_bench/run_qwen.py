"""Qwen3-VL-8B, two roles. Env: teacher-bench.

    python spike/teacher_bench/run_qwen.py detect            -> qwen.json (Qwen boxes the pots itself)
    python spike/teacher_bench/run_qwen.py verify locate     -> locate+qwen.json (keeps the boxes Qwen confirms)

The verifier sees each candidate in context (the box drawn in red on a crop around it) with the user's own
description of the object, so it can reject the plant above a pot or a pot-shaped thing that is not one.
"""
from __future__ import annotations

import json
import re
import sys
import time

import torch
from PIL import ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

import common

MODEL = "Qwen/Qwen3-VL-8B-Instruct"

DETECT = """Find every {what} in this photo, including small, distant and partly hidden ones. Mark each one
separately, even when several stand side by side. Object: {descr}
Answer only with JSON: [{{"bbox_2d": [x0, y0, x1, y1]}}]"""

# The grounding prompt of the Qwen3-VL cookbook.
DETECT_V2 = """Locate every instance that belongs to the following categories: "{cats}". Report bbox coordinates in JSON format."""

# Asked yes/no, the model says yes to almost anything (probe_verify.py: 27 of 31 wrong boxes). Naming what the
# box frames and judging the name as text separates them much better.
NAME = """Name in 1 to 4 words the single thing that the red box frames. Answer only with the name."""

JUDGE = """We look for: {what}. The user describes it as: {descr}
Another model saw a box and named what it frames: "{name}".
Is that one of the objects we look for, as the user describes them? Answer only with yes or no."""


def main():
    role = sys.argv[1]
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    what, descr = " / ".join(common.queries()), common.description()

    @torch.no_grad()
    def ask(im, text, max_new=1024):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @torch.no_grad()
    def ask_text(text):
        inp = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": text}]}], tokenize=True,
                                       add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_new_tokens=4, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    preds, t0 = {}, time.time()
    if role in ("detect", "detect_v2"):
        raw = {}
        for name in common.truth():
            text = DETECT.format(what=what, descr=descr) if role == "detect" else \
                DETECT_V2.format(cats=", ".join(common.queries()))
            ans = ask(common.load(name), text, 4096)
            raw[name] = ans
            # Qwen3-VL gives coordinates in 0..1000 of the image it saw.
            boxes = [[int(v) / 1000 for v in m.groups()] for m in
                     re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", ans)]
            preds[name] = [{"box": b, "score": 1.0} for b in boxes]
            print(f"{name}: {len(boxes)} boxes, {len(common.truth()[name])} true", flush=True)
        common.save("qwen" if role == "detect" else "qwen_v2", preds, {"model": MODEL, "raw": raw,
                                    "sec_per_photo": round((time.time() - t0) / len(preds), 2)})
        return

    src = sys.argv[2]
    cand = json.loads((common.OUT / f"{src}.json").read_text())["preds"]
    n = 0
    for name, boxes in cand.items():
        im = common.load_full(name)  # the original pixels: a pot is 20 px in the 1024 px copy
        w, h = im.size
        kept = []
        for b in boxes:
            (x0, x1), (y0, y1) = sorted((b["box"][0] * w, b["box"][2] * w)), sorted((b["box"][1] * h, b["box"][3] * h))
            bw, bh = x1 - x0, y1 - y0
            # Context: the box plus its own size around it, at least 160 px, so the model sees what is next to it.
            m = max(bw, bh, 160 * w / common.SIDE)
            cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
            crop = im.crop((int(cx0), int(cy0), int(min(w, x1 + m)), int(min(h, y1 + m)))).copy()
            ImageDraw.Draw(crop).rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(255, 0, 0), width=max(3, w // 400))
            named = ask(crop, NAME, 16).strip()
            judged = ask_text(JUDGE.format(what=what, descr=descr, name=named))
            n += 1
            kept.append({**b, "verified": judged.lower().startswith("yes"), "named": named})
        preds[name] = [b for b in kept if b["verified"]]
        rejected = [b for b in kept if not b["verified"]]
        print(f"{name}: kept {len(preds[name])} of {len(boxes)}", flush=True)
        preds[name + "#rejected"] = rejected
    rejected = {k[:-9]: v for k, v in preds.items() if k.endswith("#rejected")}
    preds = {k: v for k, v in preds.items() if not k.endswith("#rejected")}
    common.save(f"{src}+qwen", preds, {"model": MODEL, "source": src, "rejected": rejected,
                                       "sec_per_box": round((time.time() - t0) / max(1, n), 2)})


if __name__ == "__main__":
    main()
