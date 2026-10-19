"""The waste bins job for the sphere route's measure (S.8 on bins, 2026-10-17): the user's 40 street photos of bins
(data/real/container, phone, 2026-10-09), bins counted by colour by the SAM 3 teacher, nothing reviewed, as C.0 did
for roads_cars. Classes as the product's interview wrote them for this dataset (e2e contenedores_de_basura): visual
names, one per colour. Photos are hard-linked; the library is not touched.

--test-from splits by place (docs/sintetico_completar_2026-10-18.md, point 1): the photos are one walk with no
usable GPS (every photo has the same fix), so the photos taken from that time on (HHMMSS) go to test and the earlier
ones, other streets, to training. It writes the job's ingest/split.json, which ingest keeps.

    python experiments/sphere/bins_job.py [--home runs/eyeforge/p2s_test] [--name bins_place --test-from 150100]
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
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--test-from", help="HHMMSS: photos taken from then on are the test, earlier ones train")
    args = ap.parse_args()
    dst = args.home / "jobs" / args.name
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
    base = args.home / "jobs/roads_cars/spec.json"
    # roads_cars went with the old jobs (2026-10-18): the spec's defaults then, PicoDet on the generic profile.
    spec = json.loads(base.read_text()) if base.exists() else {
        "hardware": "generic", "camera": {}, "scene": {}, "user_data": {}, "generation": {}, "training": {}}
    spec.update(name=args.name, task="count", background_class="empty", classes=classes())
    spec["camera"].update(height_m=[1.2, 1.7], distance_m=[1.5, 20.0], pitch_deg=[1, 25])
    spec["scene"].update(ground_tags=["pavement"])
    spec["user_data"].update(raw=FOLDER, labeling="bins0000")
    spec["generation"].update(photo_scenes=False)
    spec["training"]["seeds"] = [0, 1, 2, 3, 4]
    (dst / "spec.json").write_text(json.dumps(spec, indent=1))
    if args.test_from:
        sides = {p["hash"]: "test" if f.split("_")[2][:6] >= args.test_from else "train" for f, p in photos.items()}
        (dst / "ingest").mkdir(exist_ok=True)
        (dst / "ingest/split.json").write_text(json.dumps(sides, indent=1, sort_keys=True))
        print(sum(v == "test" for v in sides.values()), "test photos from", args.test_from)
    print(len(photos), "photos ->", dst)


if __name__ == "__main__":
    main()
