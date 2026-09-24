"""Record the asset reviews of existing jobs in the asset knowledge base.

    python scripts/seed_asset_db.py data/jobs/live_search_check              # auto QA verdicts
    python scripts/seed_asset_db.py data/jobs/lora_alarm_traffic --human     # manual review lists of the spec

Auto: qa_assets/verdicts.json (judge from the job's usage log). Human: the spec's qa.exclude,
qa.class_override and qa.no_color lists. A model moved to another class counts as a discard for the
concept it was found for, as in the QA stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sim2chip import spec as spec_mod
from sim2chip.asset_db import AssetDB
from sim2chip.stages.qa_assets import concept_of
from sim2chip.workspace import Cache, Job, default_home


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job", type=Path)
    p.add_argument("--human", action="store_true", help="use the spec's manual review lists")
    p.add_argument("--home", type=Path, default=default_home())
    args = p.parse_args()
    job = Job(args.job)
    spec = spec_mod.load(job.spec_path)
    assets = json.loads((job.stage_dir("convert_assets") / "assets.json").read_text())
    records = []
    if args.human:
        qa = spec.qa
        exclude = set(job.resolve_input(qa.exclude).read_text().split()) if qa.exclude else set()
        override = {u: v["class"] for u, v in json.loads(job.resolve_input(qa.class_override).read_text()).items()} \
            if qa.class_override else {}
        no_color = set(json.loads(job.resolve_input(qa.no_color).read_text())) if qa.no_color else set()
        for a in assets:
            moved = override.get(a["uid"], a["class"]) != a["class"]
            keep = a["uid"] not in exclude and not moved
            reason = "human review" + (f": moved to {override[a['uid']]}" if moved else "")
            records.append({**a, "concept": concept_of(a), "decision": "keep" if keep else "discard",
                            "no_color": a["uid"] in no_color, "reason": reason, "judge": "human"})
    else:
        verdicts = json.loads((job.stage_dir("qa_assets") / "verdicts.json").read_text())
        for a in assets:
            v = verdicts.get(a["uid"])
            if not v or v.get("from_db"):
                continue
            keep = v["decision"] == "keep" and v.get("class") in (None, a["class"])
            records.append({**a, "concept": concept_of(a), "decision": "keep" if keep else "discard",
                            "no_color": v.get("no_color", False), "problems": v.get("problems"),
                            "reason": v.get("reason"), "judge": v.get("judge") or "vision model"})
    out = AssetDB(Cache(args.home / "cache").db).add(records, f"seed-{job.name}")
    keep = sum(r["decision"] == "keep" for r in records)
    print(f"{len(records)} records ({keep} keep) -> {out}")


if __name__ == "__main__":
    main()
