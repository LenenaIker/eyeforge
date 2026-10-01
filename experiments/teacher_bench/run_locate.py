"""LocateAnything-3B (NVIDIA) as a detector. Env: teacher-bench.

    python experiments/teacher_bench/run_locate.py [--tiles]
"""
from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import common

MODEL = "nvidia/LocateAnything-3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", action="store_true", help="also look at 2x2 tiles (small, distant pots)")
    ap.add_argument("--v2", action="store_true", help="model card usage: </c> between categories, 8192 tokens, "
                                                      "1536 px photo (it takes up to 2.5K; 2048 runs out of memory on a 24 GB card)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    sep, side, tokens = ("</c>", 1536, 8192) if args.v2 else (", ", common.SIDE, 2048)
    question = f"Locate all the instances that matches the following description: {sep.join(common.queries())}."

    @torch.no_grad()
    def ask(im):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": question}]}]
        text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(msgs)
        inp = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        out = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16), input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"], image_grid_hws=inp.get("image_grid_hws"),
                             tokenizer=tok, max_new_tokens=tokens, use_cache=True, generation_mode="hybrid",
                             temperature=0.7, do_sample=True, top_p=0.9)  # model card settings: greedy loops
        ans = out[0] if isinstance(out, (tuple, list)) else out
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans):
            b = [int(g) / 1000 for g in m.groups()]
            if b not in boxes:  # a looping answer repeats the same box
                boxes.append(b)
        return ans, boxes

    torch.manual_seed(0)
    preds, raw, t0 = {}, {}, time.time()
    for name in common.truth():
        im = common.load(name, side)
        ans, boxes = ask(im)
        found = [{"box": b, "score": 1.0} for b in boxes]
        raw[name] = ans
        if args.tiles:
            w, h = im.size
            for tx in (0, 1):
                for ty in (0, 1):
                    # Tiles overlap by 10% so a pot on a seam is whole in one of them.
                    x0, y0 = max(0, tx * 0.45), max(0, ty * 0.45)
                    crop = im.crop((int(x0 * w), int(y0 * h), int((x0 + 0.55) * w), int((y0 + 0.55) * h)))
                    _, tb = ask(crop)
                    found += [{"box": [x0 + b[0] * 0.55, y0 + b[1] * 0.55, x0 + b[2] * 0.55, y0 + b[3] * 0.55],
                               "score": 0.9} for b in tb]
            found = common.nms(found)
        preds[name] = found
        print(f"{name}: {len(found)} boxes, {len(common.truth()[name])} true", flush=True)
    secs = (time.time() - t0) / len(preds)
    common.save("locate" + ("_v2" if args.v2 else "") + ("_tiles" if args.tiles else ""), preds,
                {"model": MODEL, "question": question, "sec_per_photo": round(secs, 2), "raw": raw})


if __name__ == "__main__":
    main()
