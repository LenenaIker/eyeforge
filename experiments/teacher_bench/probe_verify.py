"""Which verifier prompt tells the pots from the rest: run on GLM's false boxes and as many true ones."""
import json
import random

import torch
from PIL import ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

import common
from report import match

PROMPTS = {
    "choice": """Look at the red box. The object we want: {what}. The user describes it as: {descr}
What does the red box hold? Pick one:
A) exactly one such object (the container), whole or mostly whole
B) only the plant or leaves, not the container
C) a window, railing, wall or other part of the building
D) several such objects together in one box
E) something else or nothing
Answer only with the letter.""",
    "name": """Name in 1 to 4 words the single thing that the red box frames. Answer only with the name.""",
}

proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16,
                                                    device_map="cuda").eval()
what, descr = " / ".join(common.queries()), common.description()
glm = json.loads((common.OUT / "glm.json").read_text())["preds"]
items = []
for photo, pts in common.truth().items():
    found, used = match(glm[photo], pts)
    items += [(photo, b, ok) for b, ok in zip(glm[photo], used)]
bad = [i for i in items if not i[2]]
random.seed(0)
good = random.sample([i for i in items if i[2]], len(bad))


@torch.no_grad()
def ask(im, text):
    msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                   return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


rows = []
for photo, b, ok in bad + good:
    im = common.load_full(photo)  # the original pixels: a pot is 20 px in the 1024 px copy
    w, h = im.size
    x0, y0, x1, y1 = b["box"][0] * w, b["box"][1] * h, b["box"][2] * w, b["box"][3] * h
    m = max(x1 - x0, y1 - y0, 160 * w / common.SIDE)
    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
    crop = im.crop((int(cx0), int(cy0), int(min(w, x1 + m)), int(min(h, y1 + m)))).copy()
    ImageDraw.Draw(crop).rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(255, 0, 0), width=max(3, w // 400))
    r = {k: ask(crop, p.format(what=what, descr=descr)) for k, p in PROMPTS.items()}
    rows.append({"photo": photo, "box": b["box"], "true": ok, **r})
    print(("TRUE " if ok else "FALSE"), r, flush=True)
(common.OUT / "probe_verify_full.json").write_text(json.dumps(rows, indent=1))
for label, sel in (("false boxes", [r for r in rows if not r["true"]]), ("true boxes", [r for r in rows if r["true"]])):
    from collections import Counter
    print(label, Counter(r["choice"][:1] for r in sel))
