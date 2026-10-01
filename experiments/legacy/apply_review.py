"""Aplica las decisiones de la página de revisión (criba) a data/usd/exclude.txt.

Las decisiones se descargan del db del artifact como JSON (un fichero por asset, id = uid)
en --reviews-dir. La decisión humana manda sobre la de Claude; lo no revisado se queda como estaba.
Guarda copia del exclude anterior y un resumen con las notas en data/review/.

Uso: python experiments/legacy/apply_review.py --reviews-dir data/review/db/reviews
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reviews-dir", type=Path, required=True)
    p.add_argument("--exclude", type=Path, default=ROOT / "data" / "usd" / "exclude.txt")
    args = p.parse_args()

    excluded = set(args.exclude.read_text().split())
    before = set(excluded)
    notes = []
    for f in sorted(args.reviews_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        body = doc.get("data", doc)  # admite el volcado con o sin envoltorio
        uid, decision = f.stem, body.get("decision")
        if decision == "discard":
            excluded.add(uid)
        elif decision == "keep":
            excluded.discard(uid)
        if body.get("note"):
            notes.append({"uid": uid, "decision": decision, "note": body["note"]})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "data" / "review" / f"exclude_{stamp}.txt"
    shutil.copy(args.exclude, backup)
    args.exclude.write_text("\n".join(sorted(excluded)) + "\n")
    (ROOT / "data" / "review" / f"notes_{stamp}.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False))
    print(f"excluidos {len(before)} -> {len(excluded)}  (+{len(excluded - before)} / -{len(before - excluded)})")
    print(f"copia anterior: {backup}  notas: {len(notes)}")


if __name__ == "__main__":
    main()
