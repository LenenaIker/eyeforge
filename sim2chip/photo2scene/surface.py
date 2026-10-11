"""Surface route of photo2scene (roadmap B.5): classes that are surfaces or areas (lawn, border, pavement), not things
to move around. Imitation only so far: each training photo re-framed in 3D from its own depth (surf_views.py), new
angles, heights and turns within the measured camera range, the user's decision (M.3, 2026-10-11).

A view's label comes from the surfaces SAM 3 finds in it (surf_masks.py), the user's other decision: a small rule
learnt from the training photos (their share of each surface, whole and in the middle, against their labels) says
whether the view still looks like its photo. If it does, the view keeps the photo's label; if the rule sees another
class, the view is left out. On the lawn the rule alone matched the user's own labels in 53 of 69 photos (the teacher
in 56), too loose to relabel a view, enough to notice a border re-framed out of sight.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PHRASES_ASK = """These are the classes of a camera that tells surfaces or areas apart, with what each means:
{classes}
List the surfaces and materials a segmentation model should look for to tell them apart: what the classes name
and what their descriptions mention (grass, concrete, gravel, soil, metal edging). Only things one can point at in a
photo, never a negation like "no grass". English nouns, at most 8. Answer only with JSON: ["...", ...]"""
MAX_VIEWS = 8  # re-framings per photo


def phrases(spec, cache, log=print) -> list[str]:
    """The surfaces SAM 3 looks for: from the text model, by the classes and their descriptions (cached)."""
    from ..llm import Client, parse_json
    from ..workspace import cache_key
    meaning = {c.id: c.description or c.id.replace("_", " ") for c in spec.classes}
    f = cache.root / "labels" / "surface_phrases" / f"{cache_key(meaning)}.json"
    if f.exists():
        return json.loads(f.read_text())
    client = Client()
    out = parse_json(client.chat([{"role": "user", "content": PHRASES_ASK.format(classes=json.dumps(meaning))}],
                                 client.model_for("text"), purpose="surface_phrases", temperature=0.0, max_tokens=512))
    found = list(dict.fromkeys(p.strip().lower() for p in out if isinstance(p, str) and p.strip()))[:8] \
        if isinstance(out, list) else []
    if not found:
        raise RuntimeError("the text model named no surfaces for the classes")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(found))
    log(f"surfaces to look for: {', '.join(found)}")
    return found


def fit(X, y: list[int], k: int, l2: float = 0.1, steps: int = 3000):
    """Softmax regression on standardised features; returns predict(X) -> class indices."""
    import numpy as np
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W, b, Y = np.zeros((Z.shape[1], k)), np.zeros(k), np.eye(k)[y]
    for _ in range(steps):
        P = np.exp(Z @ W + b - (Z @ W + b).max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        W -= 0.5 * (Z.T @ (P - Y) / len(Z) + l2 * W)
        b -= 0.5 * (P - Y).mean(0)
    return lambda Xn: (((np.asarray(Xn, float) - mu) / sd) @ W + b).argmax(1)


def build(flow, photos: list[tuple[Path, str]], surfaces: list[str], frames: int, pitch, height, seed: int,
          log=print) -> list[dict]:
    """Re-framed views of the training photos [(file, label)] in flow.d. Returns [{"file", "source", "class"}]
    (file relative to flow.d), the views the rule sees as another class left out."""
    d = flow.d
    (d / "photos.txt").write_text("\n".join(str(f) for f, _ in photos))
    per = max(1, min(MAX_VIEWS, math.ceil(frames / max(len(photos), 1))))
    flow.run("U.1_masks", "sam3", "surf_masks.py", ["--photos", d / "photos.txt", "--phrases", ",".join(surfaces),
                                                    "--out", d / "masks"])
    args = ["--photos", d / "photos.txt", "--masks", d / "masks", "--out", d / "views", "--views", per, "--seed", seed]
    if pitch:
        args += ["--pitch", f"{pitch[0]},{pitch[1]}"]
    if height:
        args += ["--height", f"{height[0]},{height[1]}"]
    flow.run("U.2_views", "mesh", "surf_views.py", args, ["views/views.json"])
    made = json.loads((d / "views" / "views.json").read_text())
    label = {Path(f).stem: cls for f, cls in photos}
    classes = sorted(set(label.values()))
    seen = [s for s in made["photos"] if s in label]
    if len(classes) < 2 or len(seen) < 2 * len(classes):
        log(f"too few photos per class to learn how the surfaces tell the classes apart ({len(seen)} photos)")
        return []
    X = [made["photos"][s]["feats"] for s in seen]
    y = [classes.index(label[s]) for s in seen]
    rule = fit(X, y, len(classes))
    ok = sum(int(fit([x for j, x in enumerate(X) if j != i], [v for j, v in enumerate(y) if j != i],
                     len(classes))([X[i]])[0] == y[i]) for i in range(len(X)))
    log(f"the surfaces tell the classes apart in {ok} of {len(X)} training photos (each left out in turn)")
    views = [v for v in made["views"] if v["source"] in label]
    if not views:
        return []
    src = rule([made["photos"][v["source"]]["feats"] for v in views])
    new = rule([v["feats"] for v in views])
    out = [{"file": f"views/{v['file']}", "source": v["source"], "class": label[v["source"]]}
           for v, a, b in zip(views, src, new) if a == b]
    log(f"{len(out)} of {len(views)} re-framed views keep their photo's class ({per} per photo); "
        f"{len(views) - len(out)} looked like another class and stay out")
    return out
