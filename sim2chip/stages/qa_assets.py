"""Asset quality control: decide which converted models are used and with which class.

Modes:
  none    keep everything
  manual  apply a human review (exclude list, class overrides, models without color)
  auto    models on Token Factory judge a contact sheet of 4 views per asset. qa.strategy "vision": a vision
          model decides; "describe_then_decide": a vision model describes the asset and Nemotron decides.
          Any manual lists in the spec are applied on top, so a human can still correct it
Background props (role "prop") go through the same review with the prop listed next to the classes: a
prop model judged as a target class is dropped, so the camera never learns to ignore a target.
Output: qa_assets/assets.json  [{..., class, usd, dims, no_color, role?}]
        qa_assets/verdicts.json (auto): the model's answer per asset
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from ..asset_db import review_concept
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
 "era": "current" | "vintage", "looks_like": "photo" | "video_game" | "toy_or_cartoon",
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

# qa.realistic: the models go onto real photos, where a stylized one stands out from everything around it.
REALISTIC = """
The models will be composited onto real photos of today's places, next to real things: DISCARD stylized,
cartoon, toy-like, video-game or low-poly models, untextured ones and designs from another era (antique or
vintage), even when the class is recognizable. Keep only models that could pass for a photo of the real thing
seen there today; when unsure, discard."""


class QAAssets(Stage):
    name = "qa_assets"
    runtime = "cpu"
    needs_llm = True
    reads_input = True
    uses_db = True
    uses = ("qa", "classes", "props")
    after = ("convert_assets", "preview_assets")
    skippable = "with photos"

    def enabled(self, spec) -> bool:
        return spec.synthetic  # the 3D models placed in the walk scenes (scene_render)

    def calls_llm(self, spec) -> bool:
        return spec.qa.mode == "auto"

    def run(self, ctx: Context) -> dict:
        assets = json.loads((ctx.job.stage_dir("convert_assets") / "assets.json").read_text())
        qa = ctx.spec.qa
        ids = [k.id for k in ctx.spec.classes]
        prop_ids = [p.id for p in ctx.spec.props]
        exclude, override, no_color = set(), {}, set()
        if qa.mode == "auto":
            verdicts = self.review(ctx, assets)
            for a in assets:
                v = verdicts.get(a["uid"])
                if v is None or v["decision"] == "discard":
                    exclude.add(a["uid"])
                    continue
                judged = v.get("class") or a["class"]
                if a.get("role") == "prop" and judged != a["class"] and judged in ids + prop_ids:
                    exclude.add(a["uid"])  # looks like a target (or another prop): not a safe background
                    continue
                if a.get("role") != "prop" and judged in prop_ids:
                    exclude.add(a["uid"])  # a target model that looks like background clutter
                    continue
                if a.get("role") != "prop" and judged in ids and judged != a["class"]:
                    override[a["uid"]] = judged
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
        # Live-search classes and props were oversampled: keep the first `count` survivors (search relevance order).
        cap = {c.id: c.assets.count for c in ctx.spec.classes if c.assets and c.assets.source == "search"}
        cap |= {p.id: p.count for p in ctx.spec.props}
        per_class = {}
        capped = []
        for a in kept:
            # Models generated from the user's photos were not oversampled: they are never capped.
            if a.get("generated") or per_class.get(a["class"], 0) < cap.get(a["class"], len(kept)):
                per_class[a["class"]] = per_class.get(a["class"], 0) + 1
                capped.append(a)
        kept = capped
        # A class without a usable model learns from the user's photos only (the hybrid mode): the internet has
        # no 3D model of everything (none of a yellow or a brown recycling bin, 2026-10-09). Only a job where no
        # class kept a model has nothing to render.
        wanted = [c.id for c in ctx.spec.classes if c.assets]
        empty = [c for c in wanted if not per_class.get(c)]
        if empty and len(empty) == len(wanted):
            raise ValueError(f"no usable 3D models left for classes {empty}")
        if empty:
            ctx.log(f"WARNING: no usable 3D models for {empty}: they learn from your photos only, "
                    f"the synthetic images show the other classes")
        no_props = [p.id for p in ctx.spec.props if not per_class.get(p.id)]
        if no_props:
            ctx.log(f"WARNING: no usable 3D models for props {no_props}, scenes will lack them")
        (ctx.out(self.name) / "assets.json").write_text(json.dumps(kept, indent=1))
        ctx.log(f"{len(kept)} of {len(assets)} assets kept (mode {qa.mode}): {per_class}")
        return {"kept": per_class, "dropped": len(assets) - len(kept)}

    def review(self, ctx: Context, assets: list[dict]) -> dict:
        """Verdicts from the asset knowledge base when known for the asset's concept, the vision model otherwise.
        New verdicts go back into the knowledge base."""
        from ..asset_db import AssetDB

        from .. import review as review_mod

        db = AssetDB(ctx.cache.db)
        mine = review_mod.load_corrections(ctx.job)["assets"]  # this job's review wins over the shared base
        verdicts, todo = {}, []
        for a in assets:
            r = {"decision": mine[a["uid"]], "reason": "set by the user in review"} if a["uid"] in mine \
                else db.lookup(a["uid"], review_concept(a, ctx.spec.qa.realistic))
            if r:
                verdicts[a["uid"]] = {"decision": r["decision"], "no_color": r.get("no_color", False),
                                      "reason": r.get("reason", ""), "from_db": True}
            else:
                todo.append(a)
        ctx.log(f"asset knowledge base: {len(verdicts)} verdicts reused, {len(todo)} assets to judge")
        judged = self.judge(ctx, todo) if todo else {}
        verdicts |= judged
        records = []
        for a in todo:
            v = judged.get(a["uid"])
            if v is None:
                continue  # no valid answer (transient): judge again next time
            # A model moved to another class of this spec is not a good example of the concept it was found for.
            keep = v["decision"] == "keep" and v.get("class") in (None, a["class"])
            records.append({**a, "concept": review_concept(a, ctx.spec.qa.realistic), "decision": "keep" if keep else "discard",
                            "no_color": v.get("no_color", False), "problems": v.get("problems"),
                            "reason": v.get("reason") if keep or v["decision"] == "discard"
                            else f"judged as {v.get('class')}: {v.get('reason', '')}", "judge": v.get("judge")})
        db.add(records, ctx.job.name)
        (ctx.out(self.name) / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
        return verdicts

    def judge(self, ctx: Context, assets: list[dict]) -> dict:
        from PIL import Image

        from ..llm import REASONING_TOKENS, WORKERS, Client, LLMError, ask_json, image_part

        prev = ctx.job.stage_dir("preview_assets")
        views = json.loads((prev / "views.json").read_text())
        out = ctx.out(self.name)
        sheets = out / "sheets"
        sheets.mkdir(exist_ok=True)
        client = Client(usage_log=ctx.job.logs / "llm_usage.jsonl")
        model = client.model_for("describe" if ctx.spec.qa.strategy == "describe_then_decide" else "vision")
        classes = "\n".join(f"- {c.id}: {c.description or c.id}" for c in ctx.spec.classes)
        if ctx.spec.props:
            classes += ("\nBackground objects placed around them (the camera must ignore these, so they must not"
                        " look like any class above):\n")
            classes += "\n".join(f"- {p.id}: {p.description or p.queries[0]}" for p in ctx.spec.props)
        ids = ", ".join([c.id for c in ctx.spec.classes] + [p.id for p in ctx.spec.props])
        ctx.log(f"judging {len(assets)} assets with {model}")

        strategy = ctx.spec.qa.strategy
        text_model = client.model_for("text") if strategy == "describe_then_decide" else None
        if text_model:
            ctx.log(f"decisions by {text_model}")

        def sheet_for(a):
            files = views.get(a["uid"], [])
            if len(files) < 4:
                return None
            ims = []
            for f in files[:4]:
                with Image.open(prev / f) as im:
                    ims.append(im.convert("RGB"))
            w, h = ims[0].size
            sheet = Image.new("RGB", (w * 2, h * 2))
            for k, im in enumerate(ims):
                sheet.paste(im, ((k % 2) * w, (k // 2) * h))
            path = sheets / f"{a['uid']}.jpg"
            sheet.save(path, quality=90)
            return path

        def ask(msg, mdl, purpose, check):
            return ask_json(client, msg, mdl, purpose, check, temperature=0.0, max_tokens=REASONING_TOKENS)

        errors = []

        def one(a):
            try:
                return judge_one(a)
            except (OSError, LLMError) as e:  # one broken sheet or failed call costs that asset only
                ctx.log(f"  {a['uid']}: {e}")
                errors.append(a["uid"])
                return a["uid"], None

        def judge_one(a):
            path = sheet_for(a)
            if path is None:
                return a["uid"], None
            fields = dict(classes=classes, category=a.get("category", ""), name=a.get("name", ""), cls=a["class"],
                          ids=ids, problems=PROBLEMS)
            extra = REALISTIC if ctx.spec.qa.realistic else ""
            is_verdict = lambda v: v.get("decision") in ("keep", "discard")  # noqa: E731
            if strategy == "vision":
                msg = [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(**fields) + extra},
                                                    image_part(path)]}]
                v = ask(msg, model, "qa_assets", is_verdict)
                return a["uid"], ({**v, "judge": model} if v else None)
            desc = ask([{"role": "user", "content": [{"type": "text", "text": DESCRIBE}, image_part(path)]}], model,
                       "qa_describe", lambda v: "object" in v)
            if desc is None:
                return a["uid"], None
            if ctx.spec.qa.realistic and (desc.get("style") != "realistic" or desc.get("textured") is False
                                          or desc.get("era") == "vintage" or desc.get("looks_like") != "photo"):
                return a["uid"], {"decision": "discard", "class": a["class"], "problems": [], "no_color": False,
                                  "reason": f"not realistic ({desc.get('style')}, {desc.get('looks_like')}, "
                                            f"{desc.get('era')}, textured {desc.get('textured')})",
                                  "description": desc, "judge": model}
            msg = [{"role": "user", "content": DECIDE.format(description=json.dumps(desc), **fields) + extra}]
            v = ask(msg, text_model, "qa_decide", is_verdict)
            return a["uid"], ({**v, "description": desc, "judge": f"{model} + {text_model}"} if v else None)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            verdicts = dict(pool.map(one, assets))
        if assets and len(errors) == len(assets):
            raise RuntimeError(f"the vision model failed on all {len(assets)} assets, see the lines above")
        failed = [u for u, v in verdicts.items() if v is None]
        if failed:
            ctx.log(f"{len(failed)} assets without a valid verdict are discarded")
        return verdicts
