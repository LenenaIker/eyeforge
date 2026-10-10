"""Human review of what the models decided, after they decided it.

The teacher labels the user's photos and asset QA judges the 3D models; the user only fixes what looks
wrong, never labels from scratch. This module gathers samples for a front end (photos with their labels,
3D models with their verdicts, random synthetic frames, test predictions) and applies corrections:
- photo labels (count task: the points of every object in a photo) go to input/corrections.json, which the
  label stage applies on top of its (reused) verdicts. A count photo's points cover every class, or only those
  in "checked" (a check that asked about the classes no earlier job checked: "check");
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
    return {"labels": {}, "assets": {}, "points": {}, "checked": {}, **fixes}


def checked(fixes: dict, key: str, ids) -> set[str]:
    """The classes the user's points on a count photo stand for: those in fixes["checked"], or every class when
    it does not name them (the photo was fixed with all its classes in view)."""
    if key not in fixes["points"]:
        return set()
    named = (fixes.get("checked") or {}).get(key)
    return set(ids) if named is None else set(named) & set(ids)


class StackedMarks(ValueError):
    """Corrections that mark one object as several classes (the server answers 400)."""


SAME_SPOT = 0.01  # points of two classes closer than this (share of the photo's side) mark one object


def stacked(points: list) -> set[str]:
    """The classes of a count photo's points [x, y, class] that share a spot with a point of another class: one
    object marked as several classes. A check that showed SAM 3 finding one bin under every bin phrase saved
    such stacks as the user's truth (2026-10-09), so the bin was every color at once."""
    out = set()
    for i, (x, y, c, *_) in enumerate(points):
        for x2, y2, c2, *_ in points[i + 1:]:
            if c != c2 and abs(x - x2) < SAME_SPOT and abs(y - y2) < SAME_SPOT:
                out |= {c, c2}
    return out


def unstack(fixes: dict, ids) -> dict:
    """The corrections without stacked marks: on a photo whose object was marked as several classes, those
    classes count as not checked (their points go, so the teacher's marks stand and a check asks again)."""
    for k, pts in list(fixes["points"].items()):
        bad = stacked(pts)
        if not bad:
            continue
        left = checked(fixes, k, ids) - bad
        if left:
            fixes["points"][k] = [p for p in pts if p[2] in left]
            fixes.setdefault("checked", {})[k] = sorted(left)
        else:
            fixes["points"].pop(k)
            (fixes.get("checked") or {}).pop(k, None)
    return fixes


def gather(job: Job, cache: Cache, labels: list[str], seed: int = 0, task: str = "roi_classification") -> dict:
    """Everything a reviewer looks at, with image paths relative to the job folder. For the count task, photos
    and test predictions carry points [x, y, class] in 0..1 of the (upright) image."""
    rel = lambda p: str(Path(p).relative_to(job.root))  # noqa: E731
    from .workspace import copy_name
    fixes = load_corrections(job)
    out: dict = {"task": task, "labels": labels, "photos": [], "assets": [], "renders": [], "test": [],
                 "teacher_rules": [], "check": fixes.get("check") or []}
    learned = job.stage_dir("label") / "teacher.json"
    if learned.exists():  # the threshold the teacher chose from the user's corrections
        out["teacher_rules"] = [json.loads(learned.read_text())["note"]]

    verdicts = job.stage_dir("label") / "verdicts.json"
    if verdicts.exists():
        for key, v in sorted(json.loads(verdicts.read_text()).items()):
            path = job.stage_dir("label") / v["assigned"] / (v.get("copy") or copy_name(key))
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
          points: dict[str, list] | None = None, share: bool = True, classes: list[str] | None = None) -> str | None:
    """Save corrections and return the stage to rerun from (None when nothing changed). `points` maps a photo
    to all its objects as [x, y, class] (the photo's full, corrected list); the user checked them for `classes`
    (on top of those checked before) or, when None, for every class. Asset decisions always apply to
    this job; with `share` they also go to the shared asset knowledge base, which every later job trusts
    (off on a public server, where one visitor must not decide for the others)."""
    fixes = load_corrections(job)
    changed_labels = {k: v for k, v in labels.items() if fixes["labels"].get(k) != v}
    fixes["labels"] |= changed_labels
    changed_points = {k: [[round(float(x), 4), round(float(y), 4), str(c)] for x, y, c, *_ in v]
                      for k, v in (points or {}).items()}
    changed_points = {k: v for k, v in changed_points.items() if fixes["points"].get(k) != v}
    if bad := {k: sorted(s) for k, v in changed_points.items() if (s := stacked(v))}:
        raise StackedMarks("one object is marked as several classes ("
                         + "; ".join(f"{k}: {', '.join(s)}" for k, s in bad.items())
                         + "): keep only the class it is")
    for k in changed_points:
        everything = k in fixes["points"] and k not in fixes["checked"]  # checked before for every class
        if classes is None or everything:
            fixes["checked"].pop(k, None)
        else:
            fixes["checked"][k] = sorted(set(fixes["checked"].get(k, [])) | set(classes))
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
