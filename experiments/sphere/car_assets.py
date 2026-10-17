"""The sphere route's car assets: the product's asset stages (fetch, convert, preview, QA) on a spec of passenger
cars, with qa.realistic on (cartoon cars stand out on a real panorama; the S.5 replica, 2026-10-17). The stages are
off in the product since the scene template was retired (stages/base.py TEMPLATE), so they are forced on here. The
QA runs on the local vision model unless SIM2CHIP_DESCRIBE_MODEL and SIM2CHIP_TEXT_MODEL say otherwise.

    python experiments/sphere/car_assets.py [specs/benchmarks/sphere_cars.json [stage ...]]

Writes the job in data/jobs/<spec name>/ (qa_assets/assets.json is the QA's kept list, pool.json what also passes
the look gate) and a contact sheet of the kept and discarded models next to it.
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from sim2chip import backends, spec as spec_mod
from sim2chip.llm import LOCAL_VLM
from sim2chip.pipeline import Pipeline
from sim2chip.stages import STAGES
from sim2chip.workspace import Cache, default_home, open_job

ROOT = Path(__file__).resolve().parents[2]
ASSET_STAGES = ("fetch_assets", "convert_assets", "preview_assets", "qa_assets")


def sheet(job: Path, out: Path, cols: int = 10, thumb: int = 160) -> None:
    verdicts = json.loads((job / "qa_assets" / "verdicts.json").read_text())
    sheets = job / "qa_assets" / "sheets"
    items = sorted(verdicts.items(), key=lambda kv: (kv[1] or {}).get("decision") != "keep")
    rows = (len(items) + cols - 1) // cols
    img = Image.new("RGB", (cols * thumb, rows * (thumb + 14)), "white")
    draw = ImageDraw.Draw(img)
    for k, (uid, v) in enumerate(items):
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 14)
        if (sheets / f"{uid}.jpg").exists():
            img.paste(Image.open(sheets / f"{uid}.jpg").convert("RGB").resize((thumb, thumb)), (x, y + 14))
        keep = (v or {}).get("decision") == "keep"
        draw.text((x + 2, y), f"{'KEEP' if keep else 'drop'} {uid[:6]}", fill=(0, 120, 0) if keep else (180, 0, 0),
                  font_size=11)
    img.save(out, quality=85)


# SigLIP 2 against the kept previews, in pairs with the class's own sentence: what reads more as one of these than
# as a modern car goes. On the 96 kept of 2026-10-17 it caught the antique cars, buses and golf carts the local
# describer had called current; "toy" and "cartoon" won on every render alike (a render on a grey floor reads as
# one), so they are no gate.
MODERN = "a photo of a modern passenger car"
NOT_IT = ("a photo of an antique car from the early 1900s", "a photo of a bus", "a photo of a golf cart")


def look_gate(job: Path) -> list[dict]:
    """The kept assets that read as a modern car to SigLIP 2 rather than as any of NOT_IT, minus the hand-made
    exclude list. Writes <job>/pool.json ({"kept": [...], "gated": {uid: sentence}})."""
    import numpy as np

    from sim2chip import teacher
    kept = json.loads((job / "qa_assets" / "assets.json").read_text())
    sheets, out = job / "qa_assets" / "sheets", job / "siglip"
    out.mkdir(exist_ok=True)
    items = [(sheets / f"{a['uid']}.jpg", out / f"{a['uid']}.json") for a in kept if (sheets / f"{a['uid']}.jpg").exists()]
    teacher.embed(items, [MODERN, *NOT_IT], out / "_texts.json")
    T = np.array(json.loads((out / "_texts.json").read_text())["embeddings"])
    exclude = set((ROOT / "data/usd/exclude.txt").read_text().split())
    gated, pool = {}, []
    for a in kept:
        f = out / f"{a['uid']}.json"
        if a["uid"] in exclude:
            gated[a["uid"]] = "hand-made exclude list"
            continue
        if f.exists():
            s = T @ np.array(json.loads(f.read_text())["embedding"])
            if s[1:].max() > s[0]:
                gated[a["uid"]] = NOT_IT[int(s[1:].argmax())]
                continue
        pool.append(a)
    (job / "pool.json").write_text(json.dumps({"kept": pool, "gated": gated}, indent=1))
    return pool


def main():
    spec = spec_mod.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "specs/benchmarks/sphere_cars.json")
    os.environ.setdefault("SIM2CHIP_DESCRIBE_MODEL", LOCAL_VLM)
    os.environ.setdefault("SIM2CHIP_TEXT_MODEL", LOCAL_VLM)
    for s in STAGES:
        if s.name in ASSET_STAGES:
            s.enabled = lambda _spec: True
    home = default_home()
    job = open_job(home, spec.name)
    pipe = Pipeline(spec, job, Cache(home / "cache"))
    for name in sys.argv[2:] or ASSET_STAGES:
        pipe.run(backends.get("local", verbose=True), only=name)
    sheet(job.root, job.root / "qa_sheet.jpg")
    kept = json.loads((job.root / "qa_assets" / "assets.json").read_text())
    pool = look_gate(job.root)
    print(f"{len(kept)} kept by the QA, {len(pool)} after the look gate -> {job.root / 'pool.json'}; "
          f"sheet {job.root / 'qa_sheet.jpg'}")


if __name__ == "__main__":
    main()
