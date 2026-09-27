"""Teacher: label the user's raw photos with a vision model (Nemotron on Token Factory).

Classification: each photo gets one class of the spec (background included) or "unsure". Photos below the
confidence threshold are set aside for human review instead of being used.
Count task: the teacher boxes every object in the photo; its centre becomes the object's point. Photos
without a valid answer go to "unsure"; photos with no objects are kept (they teach what "none" looks like).
Teacher answers are kept in the shared cache by photo content, model and prompt: the same photo gets the
same answer in every job (so test scores compare across jobs) and is never paid for twice. The user's
corrections from review (input/corrections.json) override the teacher.
Output: label/<class>/<photo> (copies) or, for count, label/images/<photo>; label/unsure/, label/verdicts.json
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from .. import review
from ..workspace import content_key
from .base import Context, Stage
from .ingest import IMAGE_EXT

PROMPT = """You label photos for training a small camera classifier. Classes:
{classes}

Look at the photo and pick the class of the main object in it. Use "{bg}" when none of the other classes is
present. Use "unsure" if the photo is ambiguous, too dark or blurry to tell, or shows several different classes.

Answer only with JSON: {{"class": "<one of: {ids}, unsure>", "confidence": <0 to 1>, "reason": "<short>"}}"""

COUNT_PROMPT = """You mark objects in photos to train a small camera model that counts them. Objects to find:
{classes}

Find every one of them in the photo, including small, distant and partly hidden ones. Mark each object
separately, even when several stand side by side.

Answer only with JSON: {{"objects": [{{"class": "<one of: {ids}>", "box": [x0, y0, x1, y1]}}]}} with each box
in pixel coordinates of this image, which is {w} x {h} pixels. Use an empty list when there are none."""


def copy_name(key: str) -> str:
    """Name of a photo's copy in label/<class>/: subfolder names are kept as a prefix."""
    p = key.split("/")
    return p[-1] if len(p) == 1 else f"{p[-2]}_{p[-1]}"


class Label(Stage):
    name = "label"
    runtime = "cpu"
    needs_llm = True
    reads_input = True
    rereads = ("verdicts.json",)  # the user's corrections and earlier answers
    uses_cache = True
    uses = ("user_data", "classes", "background_class", "task")

    def enabled(self, spec) -> bool:
        return bool(spec.user_data.raw)

    def run(self, ctx: Context) -> dict:
        from PIL import Image, ImageOps

        from ..llm import Client, image_part, parse_json

        ud, spec = ctx.spec.user_data, ctx.spec
        src = ctx.job.resolve_input(ud.raw)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise FileNotFoundError(f"user_data.raw: no photos in {src}")
        out = ctx.job.stage_dir(self.name)
        client = Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
        model = client.model_for("vision")
        descr = {c.id: c.description or c.id for c in spec.classes}
        counting = spec.task == "count"
        if counting:
            ids = [c.id for c in spec.classes]
            prompt = COUNT_PROMPT.format(classes="\n".join(f"- {k}: {v}" for k, v in descr.items()),
                                         ids=", ".join(ids), w="{w}", h="{h}")  # size filled per photo
        else:
            classes = "\n".join([f"- {spec.background_class}: none of the classes below, only background"] +
                                [f"- {k}: {v}" for k, v in descr.items()])
            prompt = PROMPT.format(classes=classes, bg=spec.background_class, ids=", ".join(spec.labels))
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        previous = json.loads((out / "verdicts.json").read_text()) if (out / "verdicts.json").exists() else {}
        # Corrected photos keep the teacher's own verdict next to the user's: restore it before reusing.
        teacher = {k: (({**v, "objects": v["teacher_objects"]} if "teacher_objects" in v else
                        {**v, "class": v["teacher"], "confidence": v["teacher_confidence"]}) | {"judge": "teacher"}
                       if v.get("judge") == "user" else v) for k, v in previous.items()}
        reuse = {k: v for k, v in teacher.items() if v.get("model") == model and v.get("prompt") == prompt_id
                 and v.get("reason") != "no valid answer"}
        shared = ctx.cache.root / "labels" / prompt_id / re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        shared.mkdir(parents=True, exist_ok=True)
        keys = {f: content_key(f) for f in files}
        for f in files:
            k, hit = str(f.relative_to(src)), shared / f"{keys[f]}.json"
            if hit.exists():  # the shared answer wins over this job's older one: one truth per photo
                reuse[k] = json.loads(hit.read_text())
        shutil.rmtree(out, ignore_errors=True)
        small = out / "_small"
        small.mkdir(parents=True)
        todo = [f for f in files if str(f.relative_to(src)) not in reuse]
        ctx.log(f"labeling {len(todo)} photos with {model} ({len(files) - len(todo)} already labeled)")

        def one(item):
            i, f = item
            # Downscale before sending: the model does not need more pixels than a cheap camera has.
            tmp = small / f"{i:06d}.jpg"
            # Counting needs the small, distant objects too, so it sends more pixels.
            side = 1024 if counting else 512
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((side, side))
                im.save(tmp, quality=90)
                w, h = im.size
            text = prompt.replace("{w}", str(w)).replace("{h}", str(h))
            msg = [{"role": "user", "content": [{"type": "text", "text": text}, image_part(tmp)]}]
            for _ in range(2):
                try:
                    v = parse_json(client.chat(msg, model, purpose="label", temperature=0.0,
                                               max_tokens=8192 if counting else 2048))
                    if counting:
                        return f, {"objects": objects_from(v, ids, w, h)}
                    if v.get("class") in spec.labels + ["unsure"]:
                        return f, v
                except (ValueError, TypeError, KeyError, AttributeError):
                    pass
            if counting:
                return f, {"objects": None, "reason": "no valid answer"}
            return f, {"class": "unsure", "confidence": 0, "reason": "no valid answer"}

        with ThreadPoolExecutor(max_workers=8) as pool:
            fresh = dict(pool.map(one, enumerate(todo)))
        for f, v in fresh.items():
            if v.get("reason") != "no valid answer":
                (shared / f"{keys[f]}.json").write_text(json.dumps({**v, "model": model, "prompt": prompt_id}))
        corrections = review.load_corrections(ctx.job)
        fixes, fix_points = corrections["labels"], corrections.get("points", {})
        results = []
        for f in files:
            key = str(f.relative_to(src))
            v = fresh.get(f) or reuse[key]
            v = {**v, "model": model, "prompt": prompt_id}
            if counting and key in fix_points:  # the user's marks win over the teacher's
                v = {**v, "teacher_objects": v["objects"], "judge": "user",
                     "objects": [{"x": x, "y": y, "class": c} for x, y, c in fix_points[key] if c in ids]}
            elif not counting and key in fixes:
                v = {**v, "teacher": v["class"], "teacher_confidence": v.get("confidence", 0),
                     "class": fixes[key], "confidence": 1.0, "judge": "user"}
            results.append((f, v))
        shutil.rmtree(small, ignore_errors=True)
        counts: dict[str, int] = {}
        verdicts = {}
        for f, v in results:
            if counting:
                cls = "unsure" if v["objects"] is None else "images"
            else:
                cls = v["class"] if float(v.get("confidence", 0)) >= ud.label_min_confidence else "unsure"
            d = out / cls
            d.mkdir(exist_ok=True)
            shutil.copy2(f, d / copy_name(str(f.relative_to(src))))
            counts[cls] = counts.get(cls, 0) + 1
            verdicts[str(f.relative_to(src))] = {**v, "assigned": cls}
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        if counting:
            n = sum(len(v["objects"]) for v in verdicts.values() if v["objects"] is not None)
            ctx.log(f"{counts.get('images', 0)} photos with {n} objects marked, {counts.get('unsure', 0)} unsure")
            return {"labels": counts, "objects": n}
        ctx.log(f"labels {counts}")
        return {"labels": counts}


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
