"""Baseline: the current teacher (vision model on Token Factory) with COUNT_PROMPT minus the plant phrase and
without the learned rules (they came from these same photos). Env: edge-sdg.

    python spike/teacher_bench/run_glm.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2chip.llm import WORKERS, Client, ask_json, image_part  # noqa: E402

import common  # noqa: E402

# The label stage's prompt and parser before SAM 3 replaced this teacher (2026-09-28).
COUNT_PROMPT = """You mark objects in photos to train a small camera model that counts them. Objects to find:
{classes}

Find every one of them in the photo, including small, distant and partly hidden ones. Mark each object
separately, even when several stand side by side. A box holds the whole visible object as described above,
with everything that belongs to it (a pot and the plant growing in it), not just its main part.

Answer only with JSON: {{"objects": [{{"class": "<one of: {ids}>", "box": [x0, y0, x1, y1]}}]}} with each box
in pixel coordinates of this image, which is {w} x {h} pixels. Use an empty list when there are none."""


def objects_from(answer: dict, ids: list[str], w: int, h: int) -> list[dict]:
    """The teacher's boxes (pixels of the image it saw) as points in 0..1, keeping the box for review."""
    out = []
    for o in answer["objects"]:
        cls = o.get("class", ids[0] if len(ids) == 1 else None)
        x0, y0, x1, y1 = (float(v) for v in o["box"])
        x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
        y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
        if cls not in ids or x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append({"x": round((x0 + x1) / 2 / w, 4), "y": round((y0 + y1) / 2 / h, 4), "class": cls,
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]})
    return out


PROMPT = COUNT_PROMPT.replace(" (a pot and the plant growing in it)", "")
assert PROMPT != COUNT_PROMPT, "COUNT_PROMPT changed: update the fix above"

# Short, SAM 3 style: only the names, no description. GLM publishes no grounding format, so both coordinate
# conventions are tried: pixels of the image it saw, and 0..1000 (the one GLM-4.5V was trained on).
SHORT = """Locate every instance that belongs to the following categories: {cats}.
Answer only with JSON: {{"objects": [{{"class": "maceta", "box": [x0, y0, x1, y1]}}]}} with {coords}. Use an empty
list when there are none."""
COORDS = {"px": "each box in pixel coordinates of this image, which is {w} x {h} pixels",
          "norm": "each box in coordinates normalized to 0..1000 of this image"}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "long"  # long, px or norm
    client = Client(usage_log=common.OUT / "llm_usage.jsonl")
    model = client.model_for("vision")
    tmp = Path(tempfile.mkdtemp())

    def one(name):
        im = common.load(name)
        w, h = im.size
        p = tmp / name.replace("/", "__")
        im.save(p, quality=90)
        if variant == "long":
            text = PROMPT.format(classes=f"- maceta: {common.description()}", ids="maceta", w=w, h=h)
        else:
            text = SHORT.format(cats=", ".join(common.queries()), coords=COORDS[variant].format(w=w, h=h))
        # objects_from reads pixels: 0..1000 answers are read as pixels of a 1000 x 1000 image.
        sw, sh = (1000, 1000) if variant == "norm" else (w, h)
        msg = [{"role": "user", "content": [{"type": "text", "text": text}, image_part(p)]}]
        v = ask_json(client, msg, model, "bench", lambda v: objects_from(v, ["maceta"], sw, sh) is not None,
                     temperature=0.0, max_tokens=8192)
        objs = objects_from(v, ["maceta"], sw, sh) if v else []
        return name, objs

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(one, common.truth()))
    print("example object:", next((o for _, objs in answers for o in objs), None))
    preds = {n: [{"box": o["box"], "score": 1.0} for o in objs] for n, objs in answers}
    common.save("glm" if variant == "long" else f"glm_short_{variant}", preds, {"model": model, "sec_per_photo_parallel": round((time.time() - t0) / len(preds), 2)})


if __name__ == "__main__":
    main()
