"""SAM 3 (Meta) as a detector: one text prompt per query, merged. Env: teacher-sam3.

    python spike/teacher_bench/run_sam3.py [--threshold 0.4]
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import Sam3Model, Sam3Processor

import common

MODEL = "facebook/sam3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3, help="low: the score view picks the cut later")
    args = ap.parse_args()
    model = Sam3Model.from_pretrained(MODEL).cuda().eval()
    proc = Sam3Processor.from_pretrained(MODEL)
    preds, t0 = {}, time.time()
    for name in common.truth():
        im = common.load(name)
        w, h = im.size
        found = []
        for q in common.queries():
            inp = proc(images=im, text=q, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
            r = proc.post_process_instance_segmentation(out, threshold=args.threshold, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                found.append({"box": [b[0] / w, b[1] / h, b[2] / w, b[3] / h], "score": s, "query": q})
        preds[name] = common.nms(found)
        print(f"{name}: {len(preds[name])} boxes, {len(common.truth()[name])} true", flush=True)
    common.save("sam3", preds, {"model": MODEL, "queries": common.queries(), "threshold": args.threshold,
                                "sec_per_photo": round((time.time() - t0) / len(preds), 2)})


if __name__ == "__main__":
    main()
