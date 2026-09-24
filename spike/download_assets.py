"""Descarga una muestra de modelos 3D de Objaverse por clase (categorías LVIS).

Solo se guardan modelos con licencia que permita uso comercial y derivados.
"""
import argparse
import json
import random
from pathlib import Path

import objaverse

# Clase del detector -> categorías LVIS que la componen
CLASSES = {
    "small_vehicle": ["car_(automobile)", "cab_(taxi)", "police_cruiser", "jeep", "minivan", "pickup_truck"],
    "large_vehicle": ["truck", "trailer_truck", "bus_(vehicle)", "garbage_truck", "tow_truck", "camper_(vehicle)"],
    "motorcycle": ["motorcycle", "motor_scooter"],
    "bicycle": ["bicycle"],
}

OK_LICENSES = {"by", "by-sa", "cc0"}


def license_ok(ann: dict) -> bool:
    lic = (ann.get("license") or "").lower()
    return lic in OK_LICENSES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-category", type=int, default=12)
    p.add_argument("--extra", nargs="*", default=[], help="clase=N para pedir más modelos de una clase")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "objaverse")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    extra = {k: int(v) for k, v in (e.split("=") for e in args.extra)}
    lvis = objaverse.load_lvis_annotations()
    candidates = {cat: list(lvis[cat]) for cats in CLASSES.values() for cat in cats}
    all_uids = [u for us in candidates.values() for u in us]
    anns = objaverse.load_annotations(all_uids)

    chosen: dict[str, str] = {}  # uid -> clase
    manifest = []
    for cls, cats in CLASSES.items():
        for cat in cats:
            ok = [u for u in candidates[cat] if license_ok(anns.get(u, {}))]
            random.shuffle(ok)
            n = extra.get(cls, args.per_category)
            for u in ok[:n]:
                chosen[u] = cls
                a = anns[u]
                manifest.append({"uid": u, "class": cls, "lvis": cat, "name": a.get("name"),
                                 "license": a.get("license"), "author": a.get("user", {}).get("username"),
                                 "source": a.get("viewerUrl")})
            print(f"{cls:14s} {cat:20s} {len(ok):4d} con licencia ok, tomados {min(len(ok), n)}")

    args.out.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(args.out)
    objaverse._VERSIONED_PATH = str(args.out / "hf-objaverse-v1")
    paths = objaverse.load_objects(uids=list(chosen), download_processes=8)
    for m in manifest:
        m["path"] = paths.get(m["uid"])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(paths)} modelos descargados en {args.out}")


if __name__ == "__main__":
    main()
