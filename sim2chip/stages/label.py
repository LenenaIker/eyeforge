"""Teacher: label the user's raw photos with a vision model (Nemotron on Token Factory).

Each photo gets one class of the spec (background included) or "unsure". Photos below the confidence
threshold are set aside for human review instead of being used.
Output: label/<class>/<photo> (copies), label/unsure/, label/verdicts.json
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor

from .base import Context, Stage
from .ingest import IMAGE_EXT

PROMPT = """You label photos for training a small camera classifier. Classes:
{classes}

Look at the photo and pick the class of the main object in it. Use "{bg}" when none of the other classes is
present. Use "unsure" if the photo is ambiguous, too dark or blurry to tell, or shows several different classes.

Answer only with JSON: {{"class": "<one of: {ids}, unsure>", "confidence": <0 to 1>, "reason": "<short>"}}"""


class Label(Stage):
    name = "label"
    runtime = "cpu"
    needs_llm = True
    reads_input = True
    uses = ("user_data", "classes", "background_class")

    def enabled(self, spec) -> bool:
        return bool(spec.user_data.raw)

    def run(self, ctx: Context) -> dict:
        from PIL import Image

        from ..llm import Client, image_part, parse_json

        ud, spec = ctx.spec.user_data, ctx.spec
        src = ctx.job.resolve_input(ud.raw)
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise FileNotFoundError(f"user_data.raw: no photos in {src}")
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        small = out / "_small"
        small.mkdir(parents=True)
        client = Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
        model = client.model_for("vision")
        descr = {c.id: c.description or c.id for c in spec.classes}
        classes = "\n".join([f"- {spec.background_class}: none of the classes below, only background"] +
                            [f"- {k}: {v}" for k, v in descr.items()])
        prompt = PROMPT.format(classes=classes, bg=spec.background_class, ids=", ".join(spec.labels))
        ctx.log(f"labeling {len(files)} photos with {model}")

        def one(item):
            i, f = item
            # Downscale before sending: the model does not need more pixels than a cheap camera has.
            tmp = small / f"{i:06d}.jpg"
            with Image.open(f) as im:
                im = im.convert("RGB")
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
            results = list(pool.map(one, enumerate(files)))
        shutil.rmtree(small, ignore_errors=True)
        counts: dict[str, int] = {}
        verdicts = {}
        for f, v in results:
            cls = v["class"] if float(v.get("confidence", 0)) >= ud.label_min_confidence else "unsure"
            d = out / cls
            d.mkdir(exist_ok=True)
            name = f"{f.parent.name}_{f.name}" if f.parent != src else f.name
            shutil.copy2(f, d / name)
            counts[cls] = counts.get(cls, 0) + 1
            verdicts[str(f.relative_to(src))] = {**v, "assigned": cls}
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        ctx.log(f"labels {counts}")
        return {"labels": counts}
