"""Human review of what the models decided, after they decided it.

The teacher labels the user's photos and asset QA judges the 3D models; the user only fixes what looks
wrong, never labels from scratch. This module gathers samples for a front end (photos with their labels,
3D models with their verdicts, random synthetic frames, test predictions) and applies corrections:
- photo labels (count task: the points of every object in a photo) go to input/corrections.json, which the
  label stage applies on top of its (reused) verdicts;
- asset decisions go to the asset knowledge base with judge "user", so later jobs reuse them too.
Then the job reruns from the first stage the corrections affect.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .workspace import Cache, Job

CORRECTIONS = "corrections.json"
# Folders of a job whose images the review may show.
MEDIA_DIRS = ("input", "label", "preview_assets", "qa_assets", "generate_assets", "render", "ingest")
RENDER_SAMPLE = 12


def load_corrections(job: Job) -> dict:
    f = job.input / CORRECTIONS
    fixes = json.loads(f.read_text()) if f.exists() else {}
    return {"labels": {}, "assets": {}, "points": {}, **fixes}


def gather(job: Job, cache: Cache, labels: list[str], seed: int = 0, task: str = "roi_classification") -> dict:
    """Everything a reviewer looks at, with image paths relative to the job folder. For the count task, photos
    and test predictions carry points [x, y, class] in 0..1 of the (upright) image."""
    rel = lambda p: str(Path(p).relative_to(job.root))  # noqa: E731
    from .workspace import copy_name
    fixes = load_corrections(job)
    out: dict = {"task": task, "labels": labels, "photos": [], "assets": [], "renders": [], "test": [],
                 "teacher_rules": []}
    lessons = job.stage_dir("label") / "lessons.json"
    if lessons.exists():  # what the teacher learned from earlier corrections
        out["teacher_rules"] = json.loads(lessons.read_text()).get("rules", [])

    verdicts = job.stage_dir("label") / "verdicts.json"
    if verdicts.exists():
        for key, v in sorted(json.loads(verdicts.read_text()).items()):
            path = job.stage_dir("label") / v["assigned"] / copy_name(key)
            if "objects" in v:  # count task
                pts = lambda objs: [[o["x"], o["y"], o["class"]] for o in objs or []]  # noqa: E731
                out["photos"].append({"key": key, "points": pts(v["objects"]), "unsure": v["objects"] is None,
                                      "teacher_points": pts(v.get("teacher_objects", v["objects"])),
                                      "judge": v.get("judge", "teacher"), "corrected": key in fixes["points"],
                                      "reason": v.get("reason", ""), "image": rel(path) if path.exists() else None})
                continue
            out["photos"].append({"key": key, "label": v["assigned"], "teacher": v.get("teacher", v.get("class")),
                                  "confidence": v.get("confidence"), "reason": v.get("reason", ""),
                                  "judge": v.get("judge", "teacher"), "corrected": key in fixes["labels"],
                                  "image": rel(path) if path.exists() else None})

    from .asset_db import AssetDB
    from .asset_db import concept_of
    converted = job.stage_dir("convert_assets") / "assets.json"
    if converted.exists():
        db = AssetDB(cache.db)
        kept_file = job.stage_dir("qa_assets") / "assets.json"
        kept = {a["uid"] for a in json.loads(kept_file.read_text())} if kept_file.exists() else set()
        for a in json.loads(converted.read_text()):
            r = db.lookup(a["uid"], concept_of(a)) or {}
            image = next((p for p in (job.stage_dir("qa_assets") / "sheets" / f"{a['uid']}.jpg",
                                      job.stage_dir("preview_assets") / f"{a['uid']}_0.png",
                                      job.stage_dir("generate_assets") / a["class"] / f"{a['uid']}.png")
                          if p.exists()), None)
            out["assets"].append({"uid": a["uid"], "class": a["class"], "role": a.get("role", "target"),
                                  "name": a.get("name"), "generated": bool(a.get("generated")),
                                  "kept": a["uid"] in kept, "decision": r.get("decision"),
                                  "reason": r.get("reason", ""), "judge": r.get("judge"),
                                  "image": rel(image) if image else None})

    frames = sorted(job.stage_dir("render").rglob("rgb_*.png"))
    out["renders"] = [rel(p) for p in sorted(random.Random(seed).sample(frames, min(RENDER_SAMPLE, len(frames))))]

    export = job.stage_dir("export") / "report.json"
    if export.exists():
        best = json.loads(export.read_text())["seed"]
        ev = job.stage_dir("evaluate") / f"seed_{best}.json"
        if ev.exists():
            # Count predictions point into the original photo (upright, any shape), not the square model input.
            out["test"] = [{**p, "image": p["source"] if "source" in p else f"ingest/test/{p['truth']}/{p['file']}"}
                           for p in json.loads(ev.read_text()).get("predictions", [])]
    return out


def apply(job: Job, cache: Cache, labels: dict[str, str], assets: dict[str, str],
          points: dict[str, list] | None = None, share: bool = True) -> str | None:
    """Save corrections and return the stage to rerun from (None when nothing changed). `points` maps a photo
    to all its objects as [x, y, class] (the photo's full, corrected list). Asset decisions always apply to
    this job; with `share` they also go to the shared asset knowledge base, which every later job trusts
    (off on a public server, where one visitor must not decide for the others)."""
    fixes = load_corrections(job)
    changed_labels = {k: v for k, v in labels.items() if fixes["labels"].get(k) != v}
    fixes["labels"] |= changed_labels
    changed_points = {k: [[round(float(x), 4), round(float(y), 4), str(c)] for x, y, c, *_ in v]
                      for k, v in (points or {}).items()}
    changed_points = {k: v for k, v in changed_points.items() if fixes["points"].get(k) != v}
    fixes["points"] |= changed_points
    fixes["assets"] |= assets
    job.input.mkdir(parents=True, exist_ok=True)
    (job.input / CORRECTIONS).write_text(json.dumps(fixes, indent=1))
    if assets and share:
        from .asset_db import AssetDB
        from .asset_db import concept_of
        by_uid = {a["uid"]: a for a in json.loads((job.stage_dir("convert_assets") / "assets.json").read_text())}
        AssetDB(cache.db).add([{**by_uid[u], "concept": concept_of(by_uid[u]), "decision": d, "judge": "user",
                                "reason": "set by the user in review"}
                               for u, d in assets.items() if u in by_uid], job.name)
    if changed_labels or changed_points:
        return "label"
    return "qa_assets" if assets else None
