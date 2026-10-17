"""The waste bins job for the sphere route's measure (S.8 on bins, 2026-10-17): the user's 40 street photos of bins
(data/real/container, phone, 2026-10-09), bins counted by colour by the SAM 3 teacher, nothing reviewed, as C.0 did
for roads_cars. Classes as the product's interview wrote them for this dataset (e2e contenedores_de_basura): visual
names, one per colour. Photos are hard-linked; the library is not touched.

    python experiments/sphere/bins_job.py [--home runs/eyeforge/p2s_test]
"""
import argparse
import json
import os
from pathlib import Path

from sim2chip.workspace import content_key

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/real/container"
NAME = "bins_count"
FOLDER = "labeling_bins0000"
COLOURS = ("blue", "yellow", "brown", "gray", "green")


def classes() -> list[dict]:
    return [{"id": f"{c}_bin", "description": f"{c} trash bin seen from the side or front",
             "size_m": {"measure": "height", "range": [0.8, 1.6]},
             "assets": {"source": "search", "categories": [], "count": 12, "photos": None, "generate": None,
                        "models": [], "queries": [f"{c} trash bin", f"{c} waste container", f"{c} wheelie bin"]},
             "names": [f"{c} bin", f"{c} trash container"], "repaint": None, "metallic": None} for c in COLOURS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", type=Path, default=ROOT / "runs/p2s_test")
    args = ap.parse_args()
    dst = args.home / "jobs" / NAME
    (dst / "input" / FOLDER).mkdir(parents=True, exist_ok=True)
    photos = {}
    for i, f in enumerate(sorted(SRC.glob("*.jpg")), 1):
        t = dst / "input" / FOLDER / f.name
        if not t.exists():
            os.link(f.resolve(), t)
        photos[f.name] = {"photo_id": i, "hash": content_key(f), "reviewed": False, "marks": []}
    snap = {"labeling": "bins0000", "name": "Bins by colour", "task": "count", "threshold": None,
            "classes": [{"key": c["id"], "description": c["description"], "phrases": c["names"]} for c in classes()],
            "photos": photos}
    (dst / "input/labeling.json").write_text(json.dumps(snap, indent=1))
    spec = json.loads((args.home / "jobs/roads_cars/spec.json").read_text())
    spec.update(name=NAME, task="count", background_class="empty", classes=classes())
    spec["camera"].update(height_m=[1.2, 1.7], distance_m=[1.5, 20.0], pitch_deg=[1, 25])
    spec["scene"].update(ground_tags=["pavement"])
    spec["user_data"].update(raw=FOLDER, labeling="bins0000")
    spec["generation"].update(photo_scenes=False)
    spec["training"]["seeds"] = [0, 1, 2, 3, 4]
    (dst / "spec.json").write_text(json.dumps(spec, indent=1))
    print(len(photos), "photos ->", dst)


if __name__ == "__main__":
    main()
