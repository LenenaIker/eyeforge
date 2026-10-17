"""S.6 of the sphere route, support: what holds each target up in the user's photos, so the spheres can show the
targets as the user's photos do (a bin on its wheels, a bucket hung on a rail, a lamp on a wall) without a rule
per class. The vision model names the support and the holder (flow_things.py's "on" field, here per target);
the geometry of context.py checks it: a target said to stand on the ground must have its base on the ground plane,
and one said to hang must not.

Asked on the whole photo, the local model called every bin on a hanging rack standing (2026-10-17). Here it gets
one target at a time: a crop around its box with the box drawn in, observation first and decision after, three
answers voted (flow_observe.py); no majority leaves the target unsure.

    python -m sim2chip.photo2scene.sphere.support --photos data/real/container \
        --context data/sphere_test/bins/context.json --out data/sphere_test/bins/support.json [--sheet <jpg>]

Writes {photo: [per target: class, box, support ("ground" or "hangs"), holder, votes, held_by, base_m, mounted,
agree]}.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ...llm import LOCAL_VLM, Client, ask_json, map_within

SUPPORTS = ["ground", "wall", "post_or_rail", "another_object"]
HANGS = {"hangs"}
MARGIN = 1.0       # of the box's size on every side: the crop shows what is around and behind the target
CROP_LONG = 768
SAMPLES = 5
MAX_TOKENS = 512  # a usable answer runs about 160 tokens; the rare runaway one (2048 tokens, 54 s) is cut and asked again
PROMPT = """The red box marks one {name} in a photo taken in the street. Look at the bottom of that object and \
at what is under, behind and above it. Answer with ONE JSON object, nothing else:
{{
  "observation": "<1-2 sentences: what is right under the bottom of the object in the red box, and what is behind \
and above it. Describe only.>",
  "bottom_on_ground": <true if the bottom of the object in the red box rests on the ground or floor, false if \
there is a gap of air under it>,
  "held_by": "ground|wall|post_or_rail|another_object",
  "holder": "<1-3 common words naming what holds it, as an open-vocabulary segmenter would understand: e.g. \
sidewalk, grass, metal rail, wooden post, brick wall>"
}}
held_by: "ground" if it stands on its own base or wheels; "wall" if it is fixed straight to a building wall; \
"post_or_rail" if it hangs from a post, pole, rail, bracket or hook (even one fixed to a wall); \
"another_object" if it rests on or in another movable thing."""


def marked_crop(img: Image.Image, box, margin: float = MARGIN, long: int = CROP_LONG) -> Image.Image:
    """The box (normalised x0, y0, x1, y1) with `margin` of its size around it, the box drawn in red."""
    W, H = img.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    c = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))
    crop = img.crop(c).copy()
    s = long / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))))
    d = ImageDraw.Draw(crop)
    d.rectangle([(x0 - c[0]) * s, (y0 - c[1]) * s, (x1 - c[0]) * s, (y1 - c[1]) * s], outline=(255, 0, 0),
                width=max(2, long // 200))
    return crop


def valid(v) -> bool:
    """A usable answer, its support set: "ground" when the bottom rests on it, else "hangs"."""
    if not (isinstance(v, dict) and v.get("held_by") in SUPPORTS and isinstance(v.get("holder"), str)
            and isinstance(v.get("observation"), str) and isinstance(v.get("bottom_on_ground"), bool)):
        return False
    # Whether air is under it decides; held_by stays informative only: on the bench of 2026-10-17 the model
    # called buckets hung on a rail "another_object" while saying their bottom was off the ground.
    v["support"] = "ground" if v["bottom_on_ground"] else "hangs"
    return True


def vote(samples: list[dict]) -> dict:
    """The support by majority (None without a strict majority), the holder most named among the answers that gave
    it."""
    c = Counter(s["support"] for s in samples)
    top, n = c.most_common(1)[0] if c else (None, 0)
    support = top if n > len(samples) / 2 else None
    holders = Counter(s["holder"].strip().lower() for s in samples if s["support"] == support)
    return {"support": support, "holder": holders.most_common(1)[0][0] if holders else None,
            "votes": dict(c), "held_by": dict(Counter(s["held_by"] for s in samples)),
            "observation": next((s["observation"] for s in samples if s["support"] == support), "")}


def agree(support: str | None, mounted: bool | None) -> bool | None:
    """Whether the model's support fits the measured base: on the ground stands, with air under it hangs. None
    when either is unknown."""
    if support is None or mounted is None:
        return None
    return (support in HANGS) == mounted


def ask(client: Client, crop: Image.Image, name: str, model: str = LOCAL_VLM, samples: int = SAMPLES,
        temperature: float = 0.6) -> dict | None:
    b = io.BytesIO()
    crop.save(b, "JPEG", quality=90)
    msg = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(name=name)},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}]}]
    got = [ask_json(client, msg, model, purpose="target_support", check=valid, tries=3, temperature=temperature,
                    max_tokens=MAX_TOKENS) for _ in range(samples)]
    got = [g for g in got if g]
    return {**vote(got), "samples": got} if got else None


def run(photos: Path, measured: dict, out: Path, model: str = LOCAL_VLM) -> dict:
    client = Client()
    jobs = [(name, i, m) for name, marks in sorted(measured.items()) if isinstance(marks, list) for i, m in enumerate(marks)]

    def one(job):
        name, i, m = job
        img = ImageOps.exif_transpose(Image.open(photos / name)).convert("RGB")
        return ask(client, marked_crop(img, m["box"]), m["class"].replace("_", " "), model)
    answers = map_within(one, jobs, seconds=3600, workers=1)  # one GPU, one answer at a time
    res = {}
    for (name, i, m), a in zip(jobs, answers):
        a = a or {"support": None, "holder": None, "votes": {}, "observation": "", "samples": []}
        res.setdefault(name, []).append({
            "class": m["class"], "box": m["box"], "base_m": m.get("base_m"), "mounted": m.get("mounted"),
            "wall_gap_m": m.get("wall_gap_m"), **{k: a.get(k) for k in ("support", "holder", "votes", "held_by", "observation")},
            "agree": agree(a["support"], m.get("mounted"))})
    out.write_text(json.dumps(res, indent=1))
    return res


def sheet(photos: Path, res: dict, out: Path, cell: int = 300, cols: int = 8) -> None:
    """Every target's crop with the model's support and holder over the measured base, disagreements in red."""
    items = [(name, t) for name, ts in sorted(res.items()) for t in ts]
    rows = -(-len(items) // cols)
    canvas = Image.new("RGB", (cols * cell, rows * (cell + 44)), (24, 24, 24))
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    for k, (name, t) in enumerate(items):
        img = ImageOps.exif_transpose(Image.open(photos / name)).convert("RGB")
        c = marked_crop(img, t["box"], long=cell)
        x, y = (k % cols) * cell, (k // cols) * (cell + 44)
        canvas.paste(c, (x + (cell - c.width) // 2, y + (cell - c.height) // 2))
        d = ImageDraw.Draw(canvas)
        colour = {True: (120, 220, 120), False: (255, 90, 90), None: (220, 220, 220)}[t["agree"]]
        base = "?" if t["base_m"] is None else f"{t['base_m']:.2f} m"
        d.text((x + 4, y + cell + 2), f"{t['class'].split('_')[0]} {t['support'] or 'unsure'} {dict(t['votes'])}",
               fill=colour, font=font)
        d.text((x + 4, y + cell + 22), f"{(t['holder'] or '')[:22]} | base {base}", fill=colour, font=font)
    canvas.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--context", type=Path, required=True, help="context.py's measure() output")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--model", default=LOCAL_VLM)
    args = ap.parse_args()
    res = run(args.photos, json.loads(args.context.read_text()), args.out, args.model)
    ts = [t for v in res.values() for t in v]
    known = [t for t in ts if t["agree"] is not None]
    print(f"{len(ts)} targets, {sum(t['support'] is None for t in ts)} unsure, "
          f"agree with the base on {sum(t['agree'] for t in known)} of {len(known)}")
    print(Counter((t["class"], t["support"]) for t in ts))
    if args.sheet:
        sheet(args.photos, res, args.sheet)


if __name__ == "__main__":
    main()
