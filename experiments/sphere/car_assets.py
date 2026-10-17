"""The sphere route's target assets: the product's asset stages (fetch, convert, preview, QA) on a spec of
passenger cars (or of street bins, sphere_bins.json), with qa.realistic on (cartoon cars stand out on a real panorama; the S.5 replica, 2026-10-17). The stages are
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
GATES = {"sphere_cars": (MODERN, NOT_IT)}  # per spec; a spec without one keeps what the QA kept
# Specs whose pool is cut to the targets of a user's job (photo_gate) and sized by the vision model (heights).
USER_JOBS = {"sphere_bins": (Path("/media/lenena-iker/Crucial/runs/eyeforge/p2s_test/jobs/bins_count"),
                             "street waste bin")}


def look_gate(job: Path, gate: tuple[str, tuple[str, ...]] | None) -> list[dict]:
    """The kept assets that read as the gate's own sentence to SigLIP 2 rather than as any of its others (cars:
    MODERN against NOT_IT), minus the hand-made exclude list. Writes <job>/pool.json ({"kept": [...], "gated":
    {uid: sentence}})."""
    import numpy as np

    from sim2chip import teacher
    kept = json.loads((job / "qa_assets" / "assets.json").read_text())
    sheets, out = job / "qa_assets" / "sheets", job / "siglip"
    out.mkdir(exist_ok=True)
    items = [(sheets / f"{a['uid']}.jpg", out / f"{a['uid']}.json") for a in kept if (sheets / f"{a['uid']}.jpg").exists()]
    if gate:
        teacher.embed(items, [gate[0], *gate[1]], out / "_texts.json")
        T = np.array(json.loads((out / "_texts.json").read_text())["embeddings"])
    exclude = set((ROOT / "data/usd/exclude.txt").read_text().split())
    gated, pool = {}, []
    for a in kept:
        f = out / f"{a['uid']}.json"
        if a["uid"] in exclude:
            gated[a["uid"]] = "hand-made exclude list"
            continue
        if gate and f.exists():
            s = T @ np.array(json.loads(f.read_text())["embedding"])
            if s[1:].max() > s[0]:
                gated[a["uid"]] = gate[1][int(s[1:].argmax())]
                continue
        pool.append(a)
    (job / "pool.json").write_text(json.dumps({"kept": pool, "gated": gated}, indent=1))
    return pool


def crops(job_dir: Path, out: Path) -> list[Path]:
    """Grey crops (10 % margin) of the targets the teacher marked on the user's photos (a count job's
    label/verdicts.json)."""
    from PIL import ImageOps
    v = json.loads((job_dir / "label" / "verdicts.json").read_text())
    photos = next((job_dir / "input").glob("labeling_*"))
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, x in sorted(v.items()):
        if not x["objects"]:
            continue
        im = ImageOps.exif_transpose(Image.open(photos / name)).convert("L").convert("RGB")
        for k, o in enumerate(x["objects"]):
            b = o["box"]
            w, h = b[2] - b[0], b[3] - b[1]
            box = ((b[0] - .1 * w) * im.width, (b[1] - .1 * h) * im.height, (b[2] + .1 * w) * im.width,
                   (b[3] + .1 * h) * im.height)
            f = out / f"{Path(name).stem}_{k}.jpg"
            im.crop(tuple(int(c) for c in box)).save(f)
            files.append(f)
    return files


PHOTO_SHARE = 0.5   # of the pool kept by the photo gate, the most alike; never fewer than MIN_POOL
MIN_POOL = 12


def photo_gate(job: Path, pool: list[dict], user_job: Path) -> list[dict]:
    """The assets whose shape looks most like the user's own targets: SigLIP 2 on grey renders and grey crops of the
    teacher's marks (grey, since the colour comes from the class at render time), an asset scored by the mean over
    its 4 renders of its 5 nearest crops. The search finds what the internet has (US dumpsters, office baskets);
    the user's photos say which of it their place has (bins, 2026-10-17)."""
    import numpy as np

    from sim2chip import teacher
    out = job / "photo_gate"
    real = crops(user_job, out / "crops")
    views = json.loads((job / "preview_assets" / "views.json").read_text())
    grey, items = {}, []
    for a in pool:
        for f in views.get(a["uid"], [])[:4]:
            g = out / "renders" / f"{Path(f).stem}.jpg"
            g.parent.mkdir(parents=True, exist_ok=True)
            Image.open(job / "preview_assets" / f).convert("L").convert("RGB").save(g)
            grey.setdefault(a["uid"], []).append(g)
    emb = out / "emb"
    emb.mkdir(exist_ok=True)
    items = [(f, emb / f"{f.parent.name}_{f.stem}.json") for f in real + [g for gs in grey.values() for g in gs]]
    teacher.embed([it for it in items if not it[1].exists()], [], out / "_texts.json")
    E = lambda f: np.array(json.loads((emb / f"{f.parent.name}_{f.stem}.json").read_text())["embedding"])  # noqa: E731
    C = np.stack([E(f) for f in real])
    score = {u: float(np.mean([np.sort(C @ E(g))[-5:].mean() for g in gs])) for u, gs in grey.items()}
    ranked = sorted((a for a in pool if a["uid"] in score), key=lambda a: -score[a["uid"]])
    keep = ranked[:max(MIN_POOL, int(round(PHOTO_SHARE * len(ranked))))]
    (out / "scores.json").write_text(json.dumps({a["uid"]: round(score[a["uid"]], 4) for a in ranked}, indent=1))
    return [{**a, "photo_score": round(score[a["uid"]], 4)} for a in keep]


HEIGHT_PROMPT = """The image shows 4 renders of ONE 3D model of a {what}. How tall is the real thing it shows, \
standing as it is used? Answer with one JSON object, nothing else: {{"height_m": <metres>}}"""


def heights(job: Path, pool: list[dict], what: str) -> list[dict]:
    """Each asset's real height, from the local vision model on its QA sheet: one class can hold things of very
    different sizes (a litter bin and a glass igloo are both street bins), and a model's units say nothing."""
    from sim2chip.llm import Client, image_part, parse_json
    client, out = Client(), []
    for a in pool:
        h = None
        for _ in range(2):
            try:
                v = parse_json(client.chat([{"role": "user", "content": [
                    {"type": "text", "text": HEIGHT_PROMPT.format(what=what)},
                    image_part(job / "qa_assets" / "sheets" / f"{a['uid']}.jpg")]}], LOCAL_VLM,
                    purpose="asset_height", temperature=0.0, max_tokens=2048))
                h = float(v["height_m"]) if isinstance(v, dict) and 0.2 < float(v.get("height_m", 0)) < 10 else None
            except (ValueError, TypeError, KeyError):
                continue
            if h:
                break
        out.append({**a, "height_m": h} if h else a)
    return out


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
    if not (job.root / "qa_assets" / "verdicts.json").exists():
        return
    sheet(job.root, job.root / "qa_sheet.jpg")
    kept = json.loads((job.root / "qa_assets" / "assets.json").read_text())
    pool = look_gate(job.root, GATES.get(spec.name)) if not sys.argv[2:] or "qa_assets" in sys.argv[2:] else []
    if spec.name in USER_JOBS:  # a pool for a user's own photos: their targets' look and real sizes
        user_job, what = USER_JOBS[spec.name]
        pool = heights(job.root, photo_gate(job.root, pool, user_job), what)
        (job.root / "pool.json").write_text(json.dumps({"kept": pool, "gated": {}}, indent=1))
    print(f"{len(kept)} kept by the QA, {len(pool)} after the look gate -> {job.root / 'pool.json'}; "
          f"sheet {job.root / 'qa_sheet.jpg'}")


if __name__ == "__main__":
    main()
