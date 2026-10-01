"""SAM 3 over a list of photos, in the Python that has it (torch + transformers >= 5). The label stage runs it.

    python -m sim2chip.sam3_worker < request.json

Request: {"model", "side", "floor", "phrases": {class: [noun phrase, ...]}, "items": [{"path", "out"}]}.
Each photo is shrunk to `side` pixels (as the teacher always saw them), segmented once per phrase, and every
instance scoring at least `floor` is written to its `out` file: {"detections": [{"class", "box", "score"}]},
boxes in 0..1. Keeping the low scores lets the label stage choose the threshold without running SAM 3 again.
A photo that fails gets {"error": ...}. One progress line per photo on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def nms(dets: list[dict], iou: float = 0.5) -> list[dict]:
    """Same-class boxes from several phrases of one class are one object: keep the most confident."""
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        x0, y0, x1, y1 = d["box"]
        for k in keep:
            if k["class"] != d["class"]:
                continue
            a0, b0, a1, b1 = k["box"]
            inter = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
            union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if union > 0 and inter / union > iou:
                break
        else:
            keep.append(d)
    return keep


def main() -> int:
    import torch
    from PIL import Image, ImageOps
    from transformers import Sam3Model, Sam3Processor

    req = json.load(sys.stdin)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(req["model"]).to(device).eval()
    proc = Sam3Processor.from_pretrained(req["model"])
    print(f"SAM 3 on {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}", flush=True)
    for i, item in enumerate(req["items"]):
        out = Path(item["out"])
        try:
            with Image.open(item["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((req["side"], req["side"]))
            w, h = im.size
            dets = []
            for cls, phrases in req["phrases"].items():
                for phrase in phrases:
                    inp = proc(images=im, text=phrase, return_tensors="pt").to(device)
                    with torch.no_grad():
                        res = model(**inp)
                    r = proc.post_process_instance_segmentation(
                        res, threshold=req["floor"], mask_threshold=0.5,
                        target_sizes=inp.get("original_sizes").tolist())[0]
                    for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                        x0, x1 = sorted((min(max(b[0], 0), w), min(max(b[2], 0), w)))
                        y0, y1 = sorted((min(max(b[1], 0), h), min(max(b[3], 0), h)))
                        if x1 - x0 >= 1 and y1 - y0 >= 1:
                            dets.append({"class": cls, "score": round(s, 4), "box": [
                                round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]})
            answer = {"detections": nms(dets)}
        except Exception as e:  # a broken photo costs that photo only
            answer = {"error": f"{type(e).__name__}: {e}"[:300]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(answer))
        print(f"{i + 1}/{len(req['items'])} {Path(item['path']).name}: "
              f"{answer.get('error') or len(answer['detections'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
