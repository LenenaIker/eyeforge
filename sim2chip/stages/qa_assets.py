"""Asset quality control: decide which converted models are used and with which class.

Modes:
  none    keep everything
  manual  apply a human review (exclude list, class overrides, models without color)
  auto    models on Token Factory judge a contact sheet of 4 views per asset. qa.strategy "vision": a vision
          model decides; "describe_then_decide": a vision model describes the asset and Nemotron decides.
          Any manual lists in the spec are applied on top, so a human can still correct it
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


DESCRIBE = """The image shows 4 renders (a 2x2 grid) of ONE 3D model downloaded from the internet. Describe it
factually; do not judge whether it is useful. Answer only with JSON:
{"object": "<what it is, a few words>", "style": "realistic" | "stylized" | "fantasy_or_scifi",
 "upright": true | false, "on_plinth_or_base": true | false, "scene_or_diorama": true | false,
 "object_count": <integer>, "only_a_part": true | false, "textured": true | false,
 "details": "<one sentence: shape, parts, anything unusual>"}"""

DECIDE = """You are the quality gate of a synthetic-data generator. A 3D model downloaded from the internet will be
placed in scenes to train a low-resolution camera that must recognize these classes:
{classes}

The model was listed under the category "{category}" with the name "{name}" and is currently assigned to
class "{cls}". A vision model looked at 4 renders of it and described it as:
{description}

Decide if the model is usable as an example of a class, as seen by a low-resolution camera.
- KEEP stylized, low-poly or simplified models if a person would still recognize the class at a glance.
- KEEP close variants of a class (a different type, size or configuration of the same kind of thing) under the
  most similar class of the list, as long as the camera could not tell them apart at low resolution.
- KEEP unusual, old or concept designs if they still look like something that could exist in the real world.
- A thing that is commonly seen on its own counts as complete, even if it is sometimes part of something
  bigger. "Only a part" means a fragment that never appears alone: a head, a wheel, a door, a torso.
- DISCARD: models on a plinth, base or ground slab; dioramas or whole scenes; several separate objects;
  fragments; not upright; a clearly different kind of object; impossible designs (flying, robotic, magical).
- The flags in the description are reliable: discard when style is "fantasy_or_scifi", on_plinth_or_base,
  scene_or_diorama or only_a_part is true, upright is false or object_count is above 1.
- If the listing and the description both point to ANOTHER class of the list, keep it and give that class.
  If they point to different kinds of object (listed as one thing, described as another), the model or its
  listing is unreliable: discard it.

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
        model = client.model_for("describe" if ctx.spec.qa.strategy == "describe_then_decide" else "vision")
        classes = "\n".join(f"- {c.id}: {c.description or c.id}" for c in ctx.spec.classes)
        ids = ", ".join(c.id for c in ctx.spec.classes)
        ctx.log(f"judging {len(assets)} assets with {model}")

        strategy = ctx.spec.qa.strategy
        text_model = client.model_for("text") if strategy == "describe_then_decide" else None
        if text_model:
            ctx.log(f"decisions by {text_model}")

        def sheet_for(a):
            files = views.get(a["uid"], [])
            if len(files) < 4:
                return None
            ims = [Image.open(prev / f).convert("RGB") for f in files[:4]]
            w, h = ims[0].size
            sheet = Image.new("RGB", (w * 2, h * 2))
            for k, im in enumerate(ims):
                sheet.paste(im, ((k % 2) * w, (k // 2) * h))
            path = sheets / f"{a['uid']}.jpg"
            sheet.save(path, quality=90)
            return path

        def ask(msg, mdl, purpose, check):
            for _ in range(2):
                try:
                    v = parse_json(client.chat(msg, mdl, purpose=purpose, temperature=0.0, max_tokens=4096))
                    if check(v):
                        return v
                except ValueError:
                    pass
            return None

        def one(a):
            path = sheet_for(a)
            if path is None:
                return a["uid"], None
            fields = dict(classes=classes, category=a.get("category", ""), name=a.get("name", ""), cls=a["class"],
                          ids=ids, problems=PROBLEMS)
            is_verdict = lambda v: v.get("decision") in ("keep", "discard")  # noqa: E731
            if strategy == "vision":
                msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(**fields)}, image_part(path)]}]
                return a["uid"], ask(msg, model, "qa_assets", is_verdict)
            desc = ask([{"role": "user", "content": [{"type": "text", "text": DESCRIBE}, image_part(path)]}], model,
                       "qa_describe", lambda v: "object" in v)
            if desc is None:
                return a["uid"], None
            msg = [{"role": "user", "content": DECIDE.format(description=json.dumps(desc), **fields)}]
            v = ask(msg, text_model, "qa_decide", is_verdict)
            return a["uid"], ({**v, "description": desc} if v else None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = dict(pool.map(one, assets))
        failed = [u for u, v in verdicts.items() if v is None]
        if failed:
            ctx.log(f"{len(failed)} assets without a valid verdict are discarded")
        (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        return verdicts
