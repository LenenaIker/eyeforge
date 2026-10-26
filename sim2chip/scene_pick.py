"""The photos of a dataset proposed for a walk scene (the dataset page's "Extract" cards): what the vision model
said of each photo when it profiled the dataset (scene_profile, Nemotron VL) ranks them, SigLIP 2 keeps them apart
(no two near copies of one spot). A good scene photo is outdoors, taken by hand at eye height looking ahead, with
few big things in it (they are erased, and each one erased is guessed ground), in daylight; and it has the focal
length in its EXIF, which the route needs.
"""
from __future__ import annotations

from pathlib import Path

PROPOSALS = 6
BIG_M = 1.5           # an object at least this long is big: the route erases it
SAME = 0.92           # SigLIP cosine similarity above which two photos show one spot
EYE = (0.8, 3.0)      # metres: a hand-held camera
LOOK_DOWN_MAX = 35.0  # degrees: further down, the photo shows the ground, not a place to walk


def score(d: dict) -> float | None:
    """How well a photo described by the profile suits a walk scene, higher better; None when it does not."""
    if d.get("context") != "ground" or d.get("light") in ("night", "indoor"):
        return None
    pitch, height = d.get("camera_pitch_deg"), d.get("camera_height_m")
    if isinstance(pitch, (int, float)) and pitch > LOOK_DOWN_MAX:
        return None
    if isinstance(height, (int, float)) and not EYE[0] <= height <= EYE[1]:
        return None
    big = sum(int(o.get("count") or 1) for o in d.get("objects") or []
              if isinstance(o.get("size_m"), (int, float)) and o["size_m"] >= BIG_M)
    s = 1.0 / (1 + big)
    if isinstance(pitch, (int, float)):
        s *= 1 - min(abs(pitch), LOOK_DOWN_MAX) / (2 * LOOK_DOWN_MAX)  # level looks see more of the place
    return s * (1.0 if d.get("light") in ("sun", "overcast") else 0.8)


def propose(photos: list[dict], cache_root: Path, log=print) -> list[dict]:
    """photos: [{"id", "file" (Path), "description" (profile dict)}] -> the PROPOSALS best, apart from each other:
    [{"photo_id", "score"}]."""
    from .scene_job import has_focal
    ranked = sorted(((s, p) for p in photos if (s := score(p["description"])) is not None and has_focal(p["file"])),
                    key=lambda sp: -sp[0])
    emb = {}
    if len(ranked) > PROPOSALS:
        try:
            from . import likeness
            emb = likeness.embeddings([p["file"] for _, p in ranked], cache_root, log)
        except Exception as e:  # without the teacher the best ones go, near copies or not
            log(f"scene proposals without SigLIP: {type(e).__name__}: {e}")
    picked: list[tuple[float, dict]] = []
    for s, p in ranked:
        if len(picked) == PROPOSALS:
            break
        e = emb.get(p["file"])
        if e is not None and any(_cos(e, emb[q["file"]]) > SAME for _, q in picked if q["file"] in emb):
            continue
        picked.append((s, p))
    return [{"photo_id": p["id"], "score": round(s, 3)} for s, p in picked]


def _cos(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
