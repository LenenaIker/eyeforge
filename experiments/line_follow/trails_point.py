"""A pointing VLM (Molmo2, Apache 2.0) as trail-direction teacher: it points where the trail goes, and the x of
the point is turned into left / straight / right like the SAM 3 rules (trails_eval.py, threshold k).

    $MOLMO_PYTHON experiments/line_follow/trails_point.py FRAMES OUT.json [--model allenai/Molmo2-8B] [--limit N]

Needs transformers 4.57.1 (its own env: Crucial/envs/eyeforge-molmo). Writes {frame: x in 0-1 or null} and the
raw answers, resuming from OUT.json.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPTS = {"trail": ("This photo comes from a camera worn by a hiker. Point to the farthest visible point of the "
                     "trail ahead, where the hiker should walk to keep following the trail."),
           "road": ("This photo comes from a car's dashboard camera. Point to the farthest visible point of the "
                    "car's road ahead, where the car should drive to keep following its road.")}
# Molmo2 answers <points coords="1 1 552 473">: image index, then point id, x and y scaled by 1000 (its README's
# regex: the coordinates have 3 or 4 digits); older Molmo answers <point x="62.5" y="41.0"> in %.
COORDS = re.compile(r'coords="([0-9\t:;, .]+)"')
TRIPLE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
XATTR = re.compile(r'x\d*="([0-9.]+)"')


def x_of(text: str) -> float | None:
    m = COORDS.search(text)
    if m:
        pts = TRIPLE.findall(m.group(1))
        if pts:
            return sum(int(p[1]) for p in pts) / len(pts) / 1000
    xs = XATTR.findall(text)
    if xs:
        return sum(float(v) for v in xs) / len(xs) / 100
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="allenai/Molmo2-8B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scene", choices=sorted(PROMPTS), default="trail")
    ap.add_argument("--reparse", action="store_true", help="only recompute x from the saved answers")
    a = ap.parse_args()
    if a.reparse:
        done = json.loads(a.out.read_text())
        done["x"] = {n: x_of(t) for n, t in done["raw"].items()}
        a.out.write_text(json.dumps(done))
        return
    names = sorted(json.loads((a.frames / "labels.json").read_text()))
    if a.limit:
        names = names[::max(1, len(names) // a.limit)][:a.limit]
    done = json.loads(a.out.read_text()) if a.out.exists() else {"x": {}, "raw": {}}
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                        device_map="cuda")
    t0 = time.time()
    for i, n in enumerate(n for n in names if n not in done["x"]):
        img = Image.open(a.frames / n).convert("RGB")
        msgs = [{"role": "user", "content": [dict(type="text", text=PROMPTS[a.scene]), dict(type="image", image=img)]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = proc.tokenizer.decode(ids[0, inputs["input_ids"].size(1):], skip_special_tokens=True)
        done["x"][n], done["raw"][n] = x_of(text), text
        if i % 25 == 0:
            a.out.write_text(json.dumps(done))
            print(f"{i} {time.time() - t0:.0f}s {n} {text[:120]!r}", flush=True)
    a.out.write_text(json.dumps(done))
    print("done", len(done["x"]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
