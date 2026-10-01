"""SAM 3 with visual exemplars from the calibration photos. Env: teacher-sam3.

    python spike/teacher_bench/run_sam3_exemplars.py      -> sam3_ex.json (+ split.json)

SAM 3 takes exemplar boxes only inside the image it segments, so the exemplars travel with the photo: crops of
the calibration photos (same scale as the photo) go in a strip on its right, with positive boxes on the user's
objects and negative boxes on what SAM 3 confused with them there. Detections in the strip are dropped.
The calibration photos are the first CALIBRATION of the folder spread evenly, as the app samples them; they are
not scored. Each one is also segmented with the exemplars of the others, to choose the threshold on them.
Positives: SAM 3 text boxes (sam3.json) that hold one of the user's points; negatives: confident ones that hold none.
"""
from __future__ import annotations

import json
import time

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

import common
from report import match

MODEL = "facebook/sam3"
CALIBRATION = 6  # photos the user checks before the first run (server.CALIBRATION_PHOTOS)
POS, NEG = 4, 4  # exemplars of each kind
STRIP = 0.3  # strip width, share of the photo width


def split() -> tuple[list[str], list[str]]:
    photos = list(common.truth())
    step = len(photos) / CALIBRATION
    calib = [photos[int(i * step)] for i in range(CALIBRATION)]
    return calib, [p for p in photos if p not in calib]


def exemplars(calib: list[str]) -> list[dict]:
    """Candidate exemplars: {"photo", "box" (0..1), "label" 1 or 0, "score"}."""
    sam = json.loads((common.OUT / "sam3.json").read_text())["preds"]
    truth, out = common.truth(), []
    for photo in calib:
        boxes = sam[photo]
        found, used = match(boxes, truth[photo])
        for b, ok in zip(boxes, used):
            if ok or b["score"] >= 0.5:
                out.append({"photo": photo, "box": b["box"], "label": int(ok), "score": b["score"]})
    return out


def pick(cands: list[dict]) -> list[dict]:
    """The most confident positives and negatives, at most one of each kind per photo first, for variety."""
    chosen = []
    for label, n in ((1, POS), (0, NEG)):
        pool = sorted((c for c in cands if c["label"] == label), key=lambda c: -c["score"])
        seen = set()
        for c in pool:
            if len([x for x in chosen if x["label"] == label]) >= n:
                break
            if c["photo"] not in seen:
                chosen.append(c)
                seen.add(c["photo"])
        for c in pool:  # fill up when there are fewer photos than exemplars
            if len([x for x in chosen if x["label"] == label]) >= n:
                break
            if c not in chosen:
                chosen.append(c)
    return chosen


def mosaic(photo: str, ex: list[dict]) -> tuple[Image.Image, list[list[float]], list[int], int]:
    """The photo with the exemplar crops in a strip on its right; exemplar boxes in pixels of the mosaic."""
    im = common.load(photo)
    w, h = im.size
    sw = int(w * STRIP)
    crops = []
    for e in ex:
        src = common.load(e["photo"])
        pw, ph = src.size
        x0, y0, x1, y1 = e["box"][0] * pw, e["box"][1] * ph, e["box"][2] * pw, e["box"][3] * ph
        m = max(x1 - x0, y1 - y0) * 0.5  # some context around it
        cx0, cy0, cx1, cy1 = max(0, x0 - m), max(0, y0 - m), min(pw, x1 + m), min(ph, y1 + m)
        crop = src.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
        scale = min(1.0, sw / crop.width, h / len(ex) / crop.height)  # shrink only when it does not fit
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        crops.append((crop, [(x0 - cx0) * scale, (y0 - cy0) * scale, (x1 - cx0) * scale, (y1 - cy0) * scale],
                      e["label"]))
    out = Image.new("RGB", (w + sw, h), (0, 0, 0))
    out.paste(im, (0, 0))
    boxes, labels, x, y, col_w = [], [], w, 0, 0
    for crop, b, label in crops:  # columns of crops, top to bottom
        if y + crop.height > h:
            x, y, col_w = x + col_w, 0, 0
        if x + crop.width > w + sw:
            break
        out.paste(crop, (x, y))
        boxes.append([x + b[0], y + b[1], x + b[2], y + b[3]])
        labels.append(label)
        y, col_w = y + crop.height, max(col_w, crop.width)
    return out, boxes, labels, w


def main():
    calib, test = split()
    cands = exemplars(calib)
    model = Sam3Model.from_pretrained(MODEL).cuda().eval()
    proc = Sam3Processor.from_pretrained(MODEL)

    def detect(photo, ex):
        im, boxes, labels, w = mosaic(photo, ex)
        h = im.height
        found = []
        for q in common.queries():
            inp = proc(images=im, text=q, input_boxes=[boxes], input_boxes_labels=[labels],
                       return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**inp)
            r = proc.post_process_instance_segmentation(out, threshold=0.2, mask_threshold=0.5,
                                                        target_sizes=inp.get("original_sizes").tolist())[0]
            for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                if (b[0] + b[2]) / 2 < w:  # not in the exemplar strip
                    found.append({"box": [b[0] / w, b[1] / h, min(b[2], w) / w, b[3] / h], "score": s})
        return common.nms(found), im

    preds, t0 = {}, time.time()
    for photo in calib:  # each calibration photo with the exemplars of the others, to choose the threshold
        preds[photo], _ = detect(photo, pick([c for c in cands if c["photo"] != photo]))
    ex = pick(cands)
    for i, photo in enumerate(test):
        preds[photo], im = detect(photo, ex)
        if i == 0:
            im.save(common.OUT / "sam3_ex_mosaic_example.jpg", quality=85)
        print(f"{photo}: {len([b for b in preds[photo] if b['score'] >= 0.5])} boxes >= 0.5, "
              f"{len(common.truth()[photo])} true", flush=True)
    (common.OUT / "split.json").write_text(json.dumps({"calibration": calib, "test": test}, indent=1))
    common.save("sam3_ex", preds, {"model": MODEL, "exemplars": ex, "threshold": 0.2,
                                   "sec_per_photo": round((time.time() - t0) / len(preds), 2)})


if __name__ == "__main__":
    main()
