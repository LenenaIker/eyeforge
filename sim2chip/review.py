"""Human review of what the models decided, after they decided it.

The teacher labels the user's photos and asset QA judges the 3D models; the user only fixes what looks
wrong, never labels from scratch. This module gathers samples for a front end (photos with their labels,
3D models with their verdicts, random synthetic frames, test predictions) and applies corrections:
- on the web server, photo marks go to the job's labeling in the library (server.save_review): the photos the
  user reviewed there are marked as corrected here. A job of the CLI keeps them in input/corrections.json
  (apply), which the label stage reads when the job has no labeling;
- asset decisions go to input/corrections.json and to the asset knowledge base with judge "user", so later
  jobs reuse them too.
Then the job reruns from the first stage the corrections affect.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .workspace import Cache, Job

CORRECTIONS = "corrections.json"
# Folders of a job whose images the review may show.
MEDIA_DIRS = ("input", "label", "preview_assets", "qa_assets", "generate_assets", "render", "ingest",
              "scene_render")
RENDER_SAMPLE = 12
SYNTH_SAMPLE = 8  # pictures per group on the job page: the synthetic that went in, and what each filter left out


def load_corrections(job: Job) -> dict:
    f = job.input / CORRECTIONS
    fixes = json.loads(f.read_text()) if f.exists() else {}
    return {"labels": {}, "assets": {}, "points": {}, **fixes}


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


def gather(job: Job, cache: Cache, labels: list[str], seed: int = 0, task: str = "roi_classification") -> dict:
    """Everything a reviewer looks at, with image paths relative to the job folder. For the count task, photos
    and test predictions carry points [x, y, class] in 0..1 of the (upright) image."""
    rel = lambda p: str(Path(p).relative_to(job.root))  # noqa: E731
    from .workspace import copy_name
    fixes = load_corrections(job)
    from .library import KEEP
    snap = json.loads((job.input / KEEP).read_text()) if (job.input / KEEP).exists() else None
    # The photos the user reviewed, with their marks: in the job's labeling, else in its corrections (CLI).
    if snap is not None:
        mine = {k: p["marks"] for k, p in snap["photos"].items() if p["reviewed"]}
        fixed_points = {k: [[m["x"], m["y"], m["class"]] for m in v if m.get("x") is not None] for k, v in mine.items()}
        fixed_labels = {k: v[0]["class"] for k, v in mine.items() if v}
    else:
        fixed_points, fixed_labels = fixes["points"], fixes["labels"]
    out: dict = {"task": task, "labels": labels, "photos": [], "assets": [], "renders": [], "test": [],
                 "teacher_rules": []}
    learned = job.stage_dir("label") / "teacher.json"
    if learned.exists():  # the threshold the teacher chose from the user's corrections
        out["teacher_rules"] = [json.loads(learned.read_text())["note"]]

    verdicts = job.stage_dir("label") / "verdicts.json"
    if verdicts.exists():
        for key, v in sorted(json.loads(verdicts.read_text()).items()):
            path = job.stage_dir("label") / v["assigned"] / (v.get("copy") or copy_name(key))
            if "objects" in v:  # count task
                pts = lambda objs: [[o["x"], o["y"], o["class"]] for o in objs or []]  # noqa: E731
                mine = key in fixed_points
                out["photos"].append({"key": key, "points": fixed_points[key] if mine else pts(v["objects"]),
                                      "unsure": v["objects"] is None and not mine,
                                      "teacher_points": pts(v.get("teacher_objects", v["objects"])),
                                      "judge": "user" if mine else v.get("judge", "teacher"), "corrected": mine,
                                      "reason": v.get("reason", ""), "image": rel(path) if path.exists() else None})
                continue
            mine = key in fixed_labels
            out["photos"].append({"key": key, "label": fixed_labels[key] if mine else v["assigned"],
                                  "teacher": v.get("teacher", v.get("class")),
                                  "confidence": v.get("confidence"), "reason": v.get("reason", ""),
                                  "judge": "user" if mine else v.get("judge", "teacher"), "corrected": mine,
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
                                  "name": a.get("name"), "author": a.get("author"),
                                  "generated": bool(a.get("generated")),
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
    """Save corrections in input/corrections.json and return the stage to rerun from (None when nothing
    changed). `points` maps a photo to all its objects as [x, y, class] (the photo's full, corrected list; the
    web server keeps photo marks in the library instead, server.save_review). Asset decisions always apply to
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


def synthetic(job: Job, seed: int = 0) -> dict:
    """What the job page shows of the synthetic images: how many went into training and how many the likeness to
    the user's photos left out, with a few pictures of each. Every frame is made in a walk scene, past what the
    scene's photo shows ("extrapolation"; "imitation" and "review" stay for the page's layout and are 0). Paths
    relative to the job folder. When the setup itself rules the synthetic out, only why: the folders hold an
    earlier run's frames."""
    from .spec import load as load_spec
    from .stages.render import no_synthetic
    none = {"went_in": 0, "split": {"imitation": 0, "extrapolation": 0}, "threshold": None,
            "left_out": {"likeness": 0, "review": 0}, "reviews": [],
            "samples": {"went_in": [], "likeness": [], "review": []}, "no_synthetic": None}
    try:
        spec = load_spec(job.spec_path if job.spec_path.exists() else job.root / "request.json")
    except (OSError, ValueError):
        return none
    why = no_synthetic(spec, job)
    if not spec.synthetic or job.dropped("scene_render") or job.dropped("render"):
        return {**none, "no_synthetic": why}
    rel = lambda p: str(Path(p).relative_to(job.root))  # noqa: E731
    load = lambda p: json.loads(p.read_text()) if p.exists() else None  # noqa: E731
    render = job.stage_dir("render")
    like = load(render / "likeness.json")
    went = [lab for lab in load(render / "labels.json") or [] if lab.get("scene") is not None]
    rng = random.Random(seed)
    pick = lambda xs: sorted(rng.sample(xs, min(SYNTH_SAMPLE, len(xs))), key=lambda x: x["image"])  # noqa: E731
    kept = [{"image": rel(render / f"rgb_{lab['frame']:04d}.png"), "scene": lab["scene"], "phase": "extrapolation"}
            for lab in went]
    unlike = [{"image": rel(render / f"rgb_{f['frame']:04d}.png"), "similarity": f["similarity"]}
              for f in (like or {}).get("frames", []) if not f["kept"] and (render / f"rgb_{f['frame']:04d}.png").exists()]
    return {"went_in": len(went), "split": {"imitation": 0, "extrapolation": len(went)},
            "threshold": (like or {}).get("threshold"), "left_out": {"likeness": len(unlike), "review": 0},
            "reviews": [], "samples": {"went_in": pick(kept), "likeness": pick(unlike), "review": []},
            "no_synthetic": why}
