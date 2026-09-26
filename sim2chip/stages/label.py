"""Teacher: label the user's raw photos with a vision model (Nemotron on Token Factory).

Each photo gets one class of the spec (background included) or "unsure". Photos below the confidence
threshold are set aside for human review instead of being used. On a rerun, photos already judged with the
same prompt and model are not sent again, and the user's corrections from review (input/corrections.json)
override the teacher.
Output: label/<class>/<photo> (copies), label/unsure/, label/verdicts.json
"""
from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

from .. import review
from .base import Context, Stage
from .ingest import IMAGE_EXT

PROMPT = """You label photos for training a small camera classifier. Classes:
{classes}

Look at the photo and pick the class of the main object in it. Use "{bg}" when none of the other classes is
present. Use "unsure" if the photo is ambiguous, too dark or blurry to tell, or shows several different classes.

Answer only with JSON: {{"class": "<one of: {ids}, unsure>", "confidence": <0 to 1>, "reason": "<short>"}}"""


def copy_name(key: str) -> str:
    """Name of a photo's copy in label/<class>/: subfolder names are kept as a prefix."""
    p = key.split("/")
    return p[-1] if len(p) == 1 else f"{p[-2]}_{p[-1]}"


class Label(Stage):
    name = "label"
    runtime = "cpu"
    needs_llm = True
    reads_input = True
    uses = ("user_data", "classes", "background_class")

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
        classes = "\n".join([f"- {spec.background_class}: none of the classes below, only background"] +
                            [f"- {k}: {v}" for k, v in descr.items()])
        prompt = PROMPT.format(classes=classes, bg=spec.background_class, ids=", ".join(spec.labels))
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        previous = json.loads((out / "verdicts.json").read_text()) if (out / "verdicts.json").exists() else {}
        # Corrected photos keep the teacher's own verdict next to the user's: restore it before reusing.
        teacher = {k: ({**v, "class": v["teacher"], "confidence": v["teacher_confidence"], "judge": "teacher"}
                       if v.get("judge") == "user" else v) for k, v in previous.items()}
        reuse = {k: v for k, v in teacher.items() if v.get("model") == model and v.get("prompt") == prompt_id
                 and v.get("reason") != "no valid answer"}
        shutil.rmtree(out, ignore_errors=True)
        small = out / "_small"
        small.mkdir(parents=True)
        todo = [f for f in files if str(f.relative_to(src)) not in reuse]
        ctx.log(f"labeling {len(todo)} photos with {model} ({len(files) - len(todo)} already labeled)")

        def one(item):
            i, f = item
            # Downscale before sending: the model does not need more pixels than a cheap camera has.
            tmp = small / f"{i:06d}.jpg"
            with Image.open(f) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((512, 512))
                im.save(tmp, quality=90)
            msg = [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(tmp)]}]
            for _ in range(2):
                try:
                    v = parse_json(client.chat(msg, model, purpose="label", temperature=0.0, max_tokens=2048))
                    if v.get("class") in spec.labels + ["unsure"]:
                        return f, v
                except ValueError:
                    pass
            return f, {"class": "unsure", "confidence": 0, "reason": "no valid answer"}

        with ThreadPoolExecutor(max_workers=8) as pool:
            fresh = dict(pool.map(one, enumerate(todo)))
        fixes = review.load_corrections(ctx.job)["labels"]
        results = []
        for f in files:
            key = str(f.relative_to(src))
            v = fresh.get(f) or reuse[key]
            v = {**v, "model": model, "prompt": prompt_id}
            if key in fixes:  # the user's word wins over the teacher's
                v = {**v, "teacher": v["class"], "teacher_confidence": v.get("confidence", 0),
                     "class": fixes[key], "confidence": 1.0, "judge": "user"}
            results.append((f, v))
        shutil.rmtree(small, ignore_errors=True)
        counts: dict[str, int] = {}
        verdicts = {}
        for f, v in results:
            cls = v["class"] if float(v.get("confidence", 0)) >= ud.label_min_confidence else "unsure"
            d = out / cls
            d.mkdir(exist_ok=True)
            shutil.copy2(f, d / copy_name(str(f.relative_to(src))))
            counts[cls] = counts.get(cls, 0) + 1
            verdicts[str(f.relative_to(src))] = {**v, "assigned": cls}
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        ctx.log(f"labels {counts}")
        return {"labels": counts}
