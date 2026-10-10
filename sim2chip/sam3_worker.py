"""SAM 3 over a list of photos, in the Python that has it (torch + transformers >= 5). The label stage runs it.

    python -m sim2chip.sam3_worker < request.json

Request: {"model", "side", "floor", "items": [{"path", "outs": {noun phrase: out file}}]}.
Each photo is shrunk to `side` pixels (as the teacher always saw them), segmented once per phrase of its `outs`,
and every instance scoring at least `floor` is written to that phrase's file: {"detections": [{"box", "score"}]},
boxes in 0..1. One file per photo and phrase, so a job asks only for the phrases no earlier job asked for
(stages.label puts each class's phrases together). Keeping the low scores lets the label stage choose the
threshold without running SAM 3 again. A photo that fails gets {"error": ...} in each file. One progress line
per photo on stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(model_path: str):
    """SAM 3 on the GPU (or the CPU): what run() needs. The warm GPU keeps it loaded (teacher_server.py)."""
    import torch
    from transformers import Sam3Model, Sam3Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(model_path).to(device).eval()
    return model, Sam3Processor.from_pretrained(model_path), device


def run(req: dict, loaded, say=print) -> None:
    import torch
    from PIL import Image, ImageOps
    model, proc, device = loaded
    say(f"SAM 3 on {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")
    for i, item in enumerate(req["items"]):
        outs = {phrase: Path(o) for phrase, o in item["outs"].items()}
        try:
            with Image.open(item["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((req["side"], req["side"]))
            w, h = im.size
            answers = {}
            # The photo through the vision encoder once, then every phrase on it: the encoder is most of the work,
            # and a job with 8 classes of 2 phrases ran it 16 times per photo (44 s for 6 photos, 2026-10-09).
            pix = proc(images=im, return_tensors="pt").to(device)
            with torch.no_grad():
                vision = model.get_vision_features(pixel_values=pix["pixel_values"])
            for phrase in outs:
                text = proc(text=phrase, return_tensors="pt").to(device)
                with torch.no_grad():
                    res = model(vision_embeds=vision, input_ids=text["input_ids"],
                                attention_mask=text["attention_mask"])
                r = proc.post_process_instance_segmentation(
                    res, threshold=req["floor"], mask_threshold=0.5,
                    target_sizes=pix.get("original_sizes").tolist())[0]
                dets = []
                for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                    x0, x1 = sorted((min(max(b[0], 0), w), min(max(b[2], 0), w)))
                    y0, y1 = sorted((min(max(b[1], 0), h), min(max(b[3], 0), h)))
                    if x1 - x0 >= 1 and y1 - y0 >= 1:
                        dets.append({"score": round(s, 4), "box": [
                            round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]})
                answers[phrase] = {"detections": dets}
        except Exception as e:  # a broken photo costs that photo only
            answers = dict.fromkeys(outs, {"error": f"{type(e).__name__}: {e}"[:300]})
        for phrase, out in outs.items():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(answers[phrase]))
        found = next((a["error"] for a in answers.values() if "error" in a), None)
        say(f"{i + 1}/{len(req['items'])} {Path(item['path']).name}: "
            f"{found or sum(len(a['detections']) for a in answers.values())}")


def main() -> int:
    req = json.load(sys.stdin)
    run(req, load(req["model"]), say=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
