"""Asset quality control: decide which converted models are used and with which class.

Modes: "none" keeps everything; "manual" applies a human review (exclude list, class overrides and
models without color). Week 2 adds an automatic mode with Nemotron VL.
Output: qa_assets/assets.json  [{..., class, usd, dims, no_color}]
"""
from __future__ import annotations

import json

from .base import Context, Stage


class QAAssets(Stage):
    name = "qa_assets"
    runtime = "cpu"
    uses = ("qa", "classes")
    after = ("convert_assets",)

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        assets = json.loads((ctx.job.stage_dir("convert_assets") / "assets.json").read_text())
        qa = ctx.spec.qa
        exclude, override, no_color = set(), {}, set()
        if qa.mode == "manual":
            if qa.exclude:
                exclude = set(ctx.job.resolve_input(qa.exclude).read_text().split())
            if qa.class_override:
                override = {u: v["class"] for u, v in json.loads(ctx.job.resolve_input(qa.class_override).read_text()).items()}
            if qa.no_color:
                no_color = set(json.loads(ctx.job.resolve_input(qa.no_color).read_text()))
        bad = sorted({c for c in override.values() if c not in [k.id for k in ctx.spec.classes]})
        if bad:
            raise ValueError(f"qa.class_override uses classes that are not in the spec: {bad}")
        kept = []
        for a in assets:
            if a["uid"] in exclude:
                continue
            kept.append({**a, "class": override.get(a["uid"], a["class"]), "no_color": a["uid"] in no_color})
        per_class = {}
        for a in kept:
            per_class[a["class"]] = per_class.get(a["class"], 0) + 1
        empty = [c.id for c in ctx.spec.classes if c.assets and not per_class.get(c.id)]
        if empty:
            raise ValueError(f"no usable 3D models left for classes {empty}")
        (ctx.out(self.name) / "assets.json").write_text(json.dumps(kept, indent=1))
        ctx.log(f"{len(kept)} of {len(assets)} assets kept (mode {qa.mode}): {per_class}")
        return {"kept": per_class, "dropped": len(assets) - len(kept)}
