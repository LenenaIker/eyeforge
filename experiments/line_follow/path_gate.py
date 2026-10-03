"""Zero-shot "is there a path or road to follow?" with SigLIP 2, before Molmo2 points: no user corrections, the
photo is compared with a few sentences per answer and the nearest wins (teacher.embed / teacher.nearest).

    python experiments/line_follow/path_gate.py OUT NAME=FOLDER [NAME=FOLDER ...]

Writes OUT/gate.json {set: {photo: [answer, margin]}}, prints the share of "path" per set and the time per photo.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim2chip import teacher  # noqa: E402

SENTENCES = {
    "path": ["a photo taken on a road, looking along it",
             "a photo taken on a sidewalk, a bike lane or a footpath, looking along it",
             "a photo of a trail or a dirt path ahead"],
    "none": ["a photo of a lawn or a field of grass with no path",
             "a photo of bushes, trees or a wall, with no road or path ahead",
             "a close photo of the ground with no path to follow"],
}


def main():
    out = Path(sys.argv[1])
    sets = dict(a.split("=", 1) for a in sys.argv[2:])
    out.mkdir(parents=True, exist_ok=True)
    texts = [s for v in SENTENCES.values() for s in v]
    items, where = [], {}
    for name, folder in sets.items():
        for p in sorted(Path(folder).glob("*.jpg")):
            o = out / "emb" / name / (p.stem + ".json")
            items.append((p, o))
            where[o] = (name, p.name)
    t0 = time.time()
    todo = [(p, o) for p, o in items if not o.exists()]
    teacher.embed(todo, texts, out / "text.json", log=lambda s: None)
    took = time.time() - t0
    t = json.loads((out / "text.json").read_text())["embeddings"]
    centre, i = {}, 0
    for k, v in SENTENCES.items():
        centre[k] = teacher._unit([sum(c) / len(v) for c in zip(*t[i:i + len(v)])])
        i += len(v)
    res: dict[str, dict] = {}
    for p, o in items:
        e = json.loads(o.read_text()).get("embedding")
        if e is None:
            continue
        s = {k: teacher._dot(e, c) for k, c in centre.items()}
        name, photo = where[o]
        res.setdefault(name, {})[photo] = ["path" if s["path"] > s["none"] else "none",
                                           round(s["path"] - s["none"], 4)]
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    print(f"{len(todo)} photos embedded in {took:.0f} s (model load included)")
    for name, r in res.items():
        print(f"{name:10} {len(r):4} photos, path {sum(v[0] == 'path' for v in r.values()) / len(r):.3f}")


if __name__ == "__main__":
    main()
