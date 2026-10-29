"""Does a synthetic frame look like the user's photos? Every frame is compared with the user's training photos through
SigLIP 2 embeddings (the classification teacher, teacher.py); what lies farther from all of them than the real photos
lie from each other stays out of training. The same for every task.

The threshold comes from the photos themselves: each training photo's similarity to its nearest other training
photo, and the lowest of those. A frame no real photo is as close to as the two least alike neighbours among the
real ones is outside what the camera sees.
"""
from __future__ import annotations

import json
from pathlib import Path

from .workspace import content_key


def threshold(real: list[list[float]]) -> float | None:
    """The lowest nearest-neighbour cosine similarity among the real photos (unit vectors); None under two photos."""
    import numpy as np
    if len(real) < 2:
        return None
    r = np.asarray(real, dtype=np.float32)
    sim = r @ r.T
    np.fill_diagonal(sim, -1.0)
    return float(sim.max(axis=1).min())


def nearest(synth: list[list[float]], real: list[list[float]]) -> list[float]:
    """Each synthetic frame's similarity to its closest real photo."""
    import numpy as np
    return (np.asarray(synth, dtype=np.float32) @ np.asarray(real, dtype=np.float32).T).max(axis=1).tolist()


def embeddings(files: list[Path], cache: Path, log=print, timeline=None) -> dict[Path, list[float]]:
    """SigLIP 2 embeddings of the files, by content in the cache (the label stage's own files): computed for those
    missing. Raises teacher.TeacherError when the teacher cannot run; a broken file is just missing."""
    from . import teacher
    out = {f: cache / "labels" / "siglip" / teacher.SIGLIP_MODEL.split("/")[-1] / f"{content_key(f)}.json"
           for f in files}
    todo = [f for f in files if not out[f].exists()]
    if todo:
        teacher.embed([(f, out[f]) for f in todo], [], cache / "labels" / "siglip_text" / "_none.json", log=log,
                      timeline=timeline)
    emb = {}
    for f in files:
        a = json.loads(out[f].read_text()) if out[f].exists() else {}
        if "embedding" in a:
            emb[f] = a["embedding"]
        else:
            out[f].unlink(missing_ok=True)
    return emb
