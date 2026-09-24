"""Asset quality control: decide which converted models are used and with which class.

Modes:
  none    keep everything
  manual  apply a human review (exclude list, class overrides, models without color)
  auto    a vision model (Nemotron on Token Factory) judges a contact sheet of 4 views per asset;
          any manual lists in the spec are applied on top, so a human can still correct it
Output: qa_assets/assets.json  [{..., class, usd, dims, no_color}]
        qa_assets/verdicts.json (auto): the model's answer per asset
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from .base import Context, Stage

PROBLEMS = ["plinth_or_base", "diorama_or_scene", "multiple_objects", "part_only", "not_upright", "wrong_object",
            "fantasy_or_scifi", "unrecognizable"]

PROMPT = """You are the quality gate of a synthetic-data generator. The image shows 4 renders (a 2x2 grid) of ONE
3D model downloaded from the internet. It will be placed in scenes to train a camera that must recognize
these classes:
{classes}

The model was listed under the category "{category}" with the name "{name}" and is currently assigned to
class "{cls}".

Decide if the model is usable as a realistic example of a class, as seen by a low-resolution camera.
- KEEP stylized, low-poly or simplified models if a person would still recognize the class at a glance.
- DISCARD: models on a plinth, base or ground slab; dioramas or whole scenes; several objects; only a part
  (a head, a wheel); upside down or lying on its side; a different object than the class; fantasy or
  sci-fi designs that do not exist in the real world.
- If it is a valid example of ANOTHER class of the list, keep it and give that class.
- no_color is true when the model is untextured (plain white or gray everywhere).

Answer only with JSON:
{{"decision": "keep" | "discard", "class": "<one of: {ids}>", "problems": [subset of {problems}],
  "no_color": true | false, "reason": "<one short sentence>"}}"""


class QAAssets(Stage):
    name = "qa_assets"
    runtime = "cpu"
    uses = ("qa", "classes")
    after = ("convert_assets", "preview_assets")

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        assets = json.loads((ctx.job.stage_dir("convert_assets") / "assets.json").read_text())
        qa = ctx.spec.qa
        ids = [k.id for k in ctx.spec.classes]
        exclude, override, no_color = set(), {}, set()
        if qa.mode == "auto":
            verdicts = self.judge(ctx, assets)
            for a in assets:
                v = verdicts.get(a["uid"])
                if v is None or v["decision"] == "discard":
                    exclude.add(a["uid"])
                    continue
                if v.get("class") in ids and v["class"] != a["class"]:
                    override[a["uid"]] = v["class"]
                if v.get("no_color"):
                    no_color.add(a["uid"])
        if qa.mode in ("manual", "auto"):
            if qa.exclude:
                exclude |= set(ctx.job.resolve_input(qa.exclude).read_text().split())
            if qa.class_override:
                override |= {u: v["class"] for u, v in
                             json.loads(ctx.job.resolve_input(qa.class_override).read_text()).items()}
            if qa.no_color:
                no_color |= set(json.loads(ctx.job.resolve_input(qa.no_color).read_text()))
        bad = sorted({c for c in override.values() if c not in ids})
        if bad:
            raise ValueError(f"qa.class_override uses classes that are not in the spec: {bad}")
        kept = [{**a, "class": override.get(a["uid"], a["class"]), "no_color": a["uid"] in no_color}
                for a in assets if a["uid"] not in exclude]
        per_class = {}
        for a in kept:
            per_class[a["class"]] = per_class.get(a["class"], 0) + 1
        empty = [c.id for c in ctx.spec.classes if c.assets and not per_class.get(c.id)]
        if empty:
            raise ValueError(f"no usable 3D models left for classes {empty}")
        (ctx.out(self.name) / "assets.json").write_text(json.dumps(kept, indent=1))
        ctx.log(f"{len(kept)} of {len(assets)} assets kept (mode {qa.mode}): {per_class}")
        return {"kept": per_class, "dropped": len(assets) - len(kept)}

    def judge(self, ctx: Context, assets: list[dict]) -> dict:
        from PIL import Image

        from ..llm import Client, image_part, parse_json

        prev = ctx.job.stage_dir("preview_assets")
        views = json.loads((prev / "views.json").read_text())
        out = ctx.out(self.name)
        sheets = out / "sheets"
        sheets.mkdir(exist_ok=True)
        client = Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
        model = client.model_for("vision")
        classes = "\n".join(f"- {c.id}: {c.description or c.id}" for c in ctx.spec.classes)
        ids = ", ".join(c.id for c in ctx.spec.classes)
        ctx.log(f"judging {len(assets)} assets with {model}")

        def one(a):
            files = views.get(a["uid"], [])
            if len(files) < 4:
                return a["uid"], None
            ims = [Image.open(prev / f).convert("RGB") for f in files[:4]]
            w, h = ims[0].size
            sheet = Image.new("RGB", (w * 2, h * 2))
            for k, im in enumerate(ims):
                sheet.paste(im, ((k % 2) * w, (k // 2) * h))
            path = sheets / f"{a['uid']}.jpg"
            sheet.save(path, quality=90)
            text = PROMPT.format(classes=classes, category=a.get("category", ""), name=a.get("name", ""),
                                 cls=a["class"], ids=ids, problems=PROBLEMS)
            msg = [{"role": "user", "content": [{"type": "text", "text": text}, image_part(path)]}]
            for attempt in range(2):
                try:
                    v = parse_json(client.chat(msg, model, purpose="qa_assets", temperature=0.0, max_tokens=4096))
                    if v.get("decision") in ("keep", "discard"):
                        return a["uid"], v
                except ValueError:
                    pass
            return a["uid"], None

        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = dict(pool.map(one, assets))
        failed = [u for u, v in verdicts.items() if v is None]
        if failed:
            ctx.log(f"{len(failed)} assets without a valid verdict are discarded")
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        return verdicts
