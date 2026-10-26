"""Synthetic dataset: the frames scene_render made in the walk scenes the user chose, on the CPU, minus those that do
not look like the user's training photos (likeness.py); a job with none trains on the photos alone and no_synthetic
says why. Each frame's boxes come in its label ("boxes": [[class, x0, y0, x1, y1], ...], pixels).
Output: render/rgb_*.png, render/labels.json, render/likeness.json
"""
from __future__ import annotations

import json
import os
import shutil

from .base import Context, Stage


class Render(Stage):
    name = "render"
    version = "9"  # v9: frames of the walk scenes (scene_render); v8: frames unlike the user's photos left out
    # (likeness.py); v7 and before: the scene template and the old scenes rebuilt from a photo (removed 2026-10-26)
    uses = ("generation", "classes", "background_class", "task")
    after = ("scene_render", "ingest")  # ingest: the training photos the frames are compared with
    skippable = "with photos"
    gpu = "any"       # the likeness model (SigLIP 2, in the teacher's own Python like the label stage)
    uses_cache = True

    def enabled(self, spec) -> bool:
        return spec.synthetic

    def run(self, ctx: Context) -> dict:
        out = ctx.job.stage_dir(self.name)
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        labels = like_photos(ctx, out, scene_frames(ctx, out, 0))
        (out / "labels.json").write_text(json.dumps(labels, indent=1))
        if not labels:
            ctx.log("no synthetic frames left: the model learns from the photos alone")
        return {"frames": len(labels)}


def scene_frames(ctx: Context, out, first: int) -> list[dict]:
    """The frames scene_render made, numbered from `first` on."""
    src = ctx.job.stage_dir("scene_render")
    # A run that turned the scenes off leaves the last run's frames in the folder: not this spec's.
    if not ctx.spec.synthetic or not (src / "frames.json").exists():
        return []
    labels = []
    for i, fr in enumerate(json.loads((src / "frames.json").read_text())):
        f = first + i
        shutil.copyfile(src / fr["file"], out / f"rgb_{f:04d}.png")
        cls = fr["class"] if "class" in fr else fr["boxes"][0][0] if fr["boxes"] else ctx.spec.background_class
        labels.append({"frame": f, "class": cls, "context": "scene", "scene": fr["scene"], "boxes": fr["boxes"]})
    if labels:
        ctx.log(f"{len(labels)} frames from the walk scenes ({len({lab['scene'] for lab in labels})} scenes)")
    return labels


def no_synthetic(spec, job=None) -> str | None:
    """Why the model learns from the user's photos alone, in their words, or None when synthetic images go into
    training. With a job, also what its run found: a step left out, or no frame kept."""
    if spec.generation.frames <= 0:
        return "the setup asks for your photos only"
    if not spec.generation.scenes:
        return "no scene was chosen for synthetic images"
    if job is None:
        return None
    step = "crops" if spec.task == "roi_classification" else "points"
    for name, what in (("fetch_assets", "finding 3D models failed"), ("convert_assets", "preparing 3D models failed"),
                       ("qa_assets", "checking 3D models failed"), ("scene_render", "rendering in your scenes failed"),
                       ("render", "the synthetic images could not be compared with your photos"),
                       (step, "preparing the synthetic images failed")):
        if left := job.dropped(name):
            return f"{what} ({left})"
    labels = job.stage_dir("render") / "labels.json"
    if labels.exists() and not json.loads(labels.read_text()):
        like = job.stage_dir("render") / LIKENESS
        if like.exists() and (n := json.loads(like.read_text())["left_out"]):
            return f"none of the {n} synthetic images looked like your photos"
        return "no synthetic image was made in your scenes"
    return None


LIKENESS = "likeness.json"  # in render/: how alike each frame is to the user's photos, and which went in


def like_photos(ctx: Context, out, labels: list[dict]) -> list[dict]:
    """The frames that look like the user's training photos (likeness.py); the others stay on disk, out of the
    labels. Under two training photos there is no threshold and every frame goes in."""
    from .. import likeness
    from .ingest import train_photos
    if not labels:
        return labels
    if os.environ.get("SIM2CHIP_LIKENESS") == "0":  # to measure what the frames do without it (roadmap V.6)
        ctx.log(f"likeness filter off (SIM2CHIP_LIKENESS=0): all {len(labels)} synthetic frames go in")
        return labels
    real = train_photos(ctx)
    if len(real) < 2:
        ctx.log(f"only {len(real)} training photos to compare with: every synthetic frame goes in")
        return labels
    frames = {lab["frame"]: out / f"rgb_{lab['frame']:04d}.png" for lab in labels}
    timeline = lambda step, seconds: ctx.job.timeline(step, seconds, os.environ.get("SIM2CHIP_RUNS_ON", "local"))  # noqa: E731
    emb = likeness.embeddings(real + list(frames.values()), ctx.cache.root, ctx.log, timeline)
    real_emb = [emb[f] for f in real if f in emb]
    thr = likeness.threshold(real_emb)
    if thr is None:
        ctx.log(f"only {len(real_emb)} training photos to compare with: every synthetic frame goes in")
        return labels
    shown = [lab for lab in labels if frames[lab["frame"]] in emb]
    sims = likeness.nearest([emb[frames[lab["frame"]]] for lab in shown], real_emb) if shown else []
    kept = [lab for lab, s in zip(shown, sims) if s >= thr]
    (out / LIKENESS).write_text(json.dumps({
        "threshold": round(thr, 4), "photos": len(real_emb), "kept": len(kept), "left_out": len(labels) - len(kept),
        "frames": [{"frame": lab["frame"], "similarity": round(s, 4), "kept": s >= thr} for lab, s in zip(shown, sims)]},
        indent=1))
    ctx.log(f"{len(kept)} of {len(labels)} synthetic frames look like the {len(real_emb)} training photos (SigLIP 2 "
            f"similarity to the nearest one at least {thr:.3f}, the least alike pair of neighbours among the photos)")
    return kept
