"""Surface route of photo2scene (roadmap B.5): classes that are surfaces or areas (lawn, border, pavement), not things
to move around. Imitation: each training photo re-framed in 3D from its own depth (surf_views.py), new angles,
heights and turns within the measured camera range (the user's decision, M.3, 2026-10-11). Extrapolation, a tenth of
the views (decision 0.1): two photos of different classes joined on one flat ground along a random line
(surf_mix.py), new borders between the user's own surfaces.

Every view is labelled like one of the user's photos: by the classification teacher with what it learnt from their
corrections (label/teacher.json's centres), nothing made per case (the user, 2026-10-11). A re-framed view keeps its
photo's class only when the teacher gives it that class too; a joined ground takes the teacher's class. Below the
job's label_min_confidence a view stays out. A first try labelled the views by a rule learnt from SAM 3 surface masks:
it knew border in 1 of the lawn's 5 border photos, so no new border ever got that label.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MAX_VIEWS = 8    # re-framings per photo
MIX_SHARE = 0.1  # of the views, joined grounds (extrapolation)


def build(flow, photos: list[tuple[Path, str]], frames: int, pitch, height, seed: int, classify, min_conf: float,
          log=print) -> list[dict]:
    """Views of the training photos [(file, label)] in flow.d, labelled by classify(files) -> [(class, confidence)]
    (the teacher). Returns [{"file", "source", "class"}] with file relative to flow.d."""
    d = flow.d
    (d / "photos.txt").write_text("\n".join(str(f) for f, _ in photos))
    label = {Path(f).stem: cls for f, cls in photos}
    per = max(1, min(MAX_VIEWS, math.ceil(frames / max(len(photos), 1))))
    args = ["--photos", d / "photos.txt", "--out", d / "views", "--views", per, "--seed", seed]
    if pitch:
        args += ["--pitch", f"{pitch[0]},{pitch[1]}"]
    if height:
        args += ["--height", f"{height[0]},{height[1]}"]
    flow.run("U.1_views", "mesh", "surf_views.py", args, ["views/views.json"])
    views = [v for v in json.loads((d / "views" / "views.json").read_text())["views"] if v["source"] in label]
    said = classify([d / "views" / v["file"] for v in views]) if views else []
    out = [{"file": f"views/{v['file']}", "source": v["source"], "class": c}
           for v, (c, conf) in zip(views, said) if c == label[v["source"]] and conf >= min_conf]
    log(f"{len(out)} of {len(views)} re-framed views keep their photo's class for the teacher ({per} per photo)")
    mix = round(len(out) * MIX_SHARE / (1 - MIX_SHARE))
    if mix and len(set(label.values())) > 1:
        (d / "labels.json").write_text(json.dumps(label))
        args = ["--photos", d / "photos.txt", "--labels", d / "labels.json", "--out", d / "mix", "--views", 2 * mix,
                "--seed", seed] + (["--pitch", f"{pitch[0]},{pitch[1]}"] if pitch else [])
        flow.run("U.2_mix", "mesh", "surf_mix.py", args, ["mix/mix.json"])
        joined = json.loads((d / "mix" / "mix.json").read_text())
        said = classify([d / "mix" / v["file"] for v in joined]) if joined else []
        sure = [(v, c) for v, (c, conf) in zip(joined, said) if conf >= min_conf][:mix]
        out += [{"file": f"mix/{v['file']}", "source": f"{v['a']}+{v['b']}", "class": c} for v, c in sure]
        found: dict[str, int] = {}
        for _, c in sure:
            found[c] = found.get(c, 0) + 1
        log(f"{len(sure)} new grounds joining two of your surfaces ({len(joined)} made, {mix} wanted): "
            + (", ".join(f"{c} {n}" for c, n in sorted(found.items())) or "none sure enough"))
    return out
