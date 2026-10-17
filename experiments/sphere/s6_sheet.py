"""S.6 check (docs/roadmap_photosphere.md): extrapolation. On each sphere, views turned at random around the capture
point with 1 to 5 cars placed where the sphere could hold them (extrapolate.py), rendered by Isaac over the proxy
and composited on the erased panorama; the cars take the blur the replica measured (S.5) and the whole frame the
phone's camera (camera_fx generic). Labels come from Isaac's instance maps. One JPG sheet per sphere with the boxes.

The cars are the reviewed realistic ones (car_assets.py); their fronts are measured first with Orient Anything V2
on one render each (an asset's front is wherever its modeller put it), into the asset job's fronts.json.

    python experiments/sphere/s6_sheet.py data/sphere_test/PXL_* [--views 20] [--hardware generic]

Targets the spheres hold none of (bins, 2026-10-17) are placed by the context measured on the user's photos
(--context, context.py), with their pool (--assets) and label (--label):

    python experiments/sphere/s6_sheet.py data/sphere_test/PXL_* --assets data/jobs/sphere_bins \
        --context data/sphere_test/bins/context.json --label "waste bin" --photos data/real/container --tag _bins
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import STEPS, camera_fx, context, extrapolate, render_plan, replica, review
from sim2chip.workspace import Cache, default_home

ROOT = Path(__file__).resolve().parents[2]
ASSET_JOB = default_home() / "jobs" / "sphere_cars"
RENDER_SIZE = {"esp32cam": (640, 480), "esp32s3": (640, 480)}  # twice the sensor's frame, which it resamples
THUMB = (320, 240)
FRONT_DISTANCES_M = (8.0, 9.0, 7.0, 10.0)


def pool(job: Path = ASSET_JOB) -> list[dict]:
    """The kept assets with absolute USD paths, metric sizes and their measured front."""
    cache = Cache(default_home() / "cache")
    fronts = json.loads((job / "fronts.json").read_text()) if (job / "fronts.json").exists() else {}
    out = []
    for a in json.loads((job / "pool.json").read_text())["kept"]:
        if not all(a.get("dims") or [0]):
            continue  # no measured size: nothing to scale it by
        usd = Path(a["usd"]) if Path(a["usd"]).is_absolute() else cache.root / a["usd"]
        f = fronts.get(a["uid"], {})
        out.append({**a, "usd": str(usd), "width_range_m": (1.65, 1.95),
                    "front_flip_deg": f.get("flip_deg", 0.0), "front_deg": f.get("front_deg", 0.0)})
    return out


def measure_fronts(d: Path, cars: list[dict], rng) -> dict:
    """Each asset rendered alone at a known yaw on free ground of sphere d, then Orient Anything on its cutout:
    flip_deg 180 when its front comes out at the back."""
    ground = render_plan.Ground(d)
    views, intended = [], {}
    for k, a in enumerate(cars):
        size = extrapolate.sized(a, a["height_m"]) if a.get("height_m") else extrapolate.asset_size(a, rng)
        for _ in range(20):
            heading = float(rng.uniform(0, 360))
            yaw = (90 - heading) % 360 + 60  # three quarters: front or back clearly in view
            spot = ground.place(heading, FRONT_DISTANCES_M, size[0], size[1], yaw)
            if spot:
                break
        else:
            continue
        pos, _ = spot
        views.append({"name": f"f{k:03d}", "yaw_deg": round(heading % 360, 2), "pitch_deg": 5.0, "fov_deg": 50.0,
                      "size": [768, 576], "objects": [{"usd": a["usd"], "length_m": max(size[:2]), "pos": pos,
                                                       "yaw_deg": yaw, "label": "car", "uid": a["uid"],
                                                       "key": f"f{k}"}]})
        intended[f"o{k:03d}"] = (a["uid"], yaw)
    render_plan.write(d, views, sun=None, sub="fronts")
    flow = Flow(d)
    flow.run("sphere_fronts", "isaac", str(STEPS / "render.py"), ["--dir", d, "--sub", "fronts", "--path-tracing"])
    cut = d / "fronts" / "cut"
    cut.mkdir(exist_ok=True)
    looked = {}
    for k, v in enumerate(views):
        inst = np.asarray(Image.open(d / "fronts" / f"inst_{v['name']}.png")) == 1
        if inst.sum() < 500:
            continue
        img = np.asarray(Image.open(d / "fronts" / f"raw_{v['name']}.png").convert("RGB"))
        name = f"o{int(v['name'][1:]):03d}"
        replica.cutout(img, inst).save(cut / f"{name}.png")
        looked[name] = v["yaw_deg"]
    flow.run("sphere_orient_fronts", "sam3", str(STEPS / "orient.py"),
             ["--crops", cut, "--out", d / "fronts" / "orient.json"])
    heads = replica.headings(json.loads((d / "fronts" / "orient.json").read_text()), looked)
    fronts = {}
    for name, (uid, yaw) in intended.items():
        h = heads.get(name)
        if h and h["front"]:
            off = (h["yaw_deg"] - yaw) % 360  # where its front is, from its own axis (the model's: x)
            fronts[uid] = {"flip_deg": 180.0 if abs((off + 180) % 360 - 180) > 90 else 0.0,
                           "front_deg": float(round(off / 90) * 90 % 360),
                           "seen_yaw_deg": h["yaw_deg"], "given_yaw_deg": round(yaw % 360, 1)}
    return fronts


def blur_for(places: list[Path]) -> dict:
    """The objects' blur per sphere: its replica's measure, else the median of the measured ones."""
    measured = {d: json.loads((d / "replica" / "measure.json").read_text())["camera"]["blur_px"]
                for d in places if (d / "replica" / "measure.json").exists()}
    fallback = float(np.median(list(measured.values()))) if measured else 0.0
    return {d: measured.get(d, fallback) for d in places}


def likeness(places: list[Path], photos: list[Path]) -> dict:
    """SigLIP 2's similarity filter (sim2chip/likeness.py, A.3): each frame against the user's photos, under the
    threshold the photos set among themselves is out. Per place: {view name: (similarity, kept)}."""
    from sim2chip import likeness as lk
    frames = {d: sorted((d / "render").glob("final_x*.png")) for d in places}
    files = photos + [f for fs in frames.values() for f in fs]
    emb = lk.embeddings(files, Cache(default_home() / "cache").root)
    real = [emb[f] for f in photos if f in emb]
    thr = lk.threshold(real)
    out = {}
    for d, fs in frames.items():
        sims = lk.nearest([emb[f] for f in fs], real) if fs and real else []
        out[d] = {f.stem[len("final_"):]: (round(s, 3), thr is None or s >= thr) for f, s in zip(fs, sims)}
    return {"threshold": thr, "places": out}


def sheet(d: Path, plan: dict, labels: dict, liked: dict | None = None, rated: dict | None = None,
          cols: int = 4) -> Image.Image:
    views = plan["views"]
    rows = math.ceil(len(views) / cols)
    out = Image.new("RGB", (cols * (THUMB[0] + 6), 40 + rows * (THUMB[1] + 22)), "white")
    draw = ImageDraw.Draw(out)
    n = sum(len(b) for b in labels.values())
    liked = liked or {}
    rated = rated or {}
    kept = sum(1 for v in views if liked.get(v["name"], (0, True))[1])
    passed = sum(1 for v in views if rated.get(v["name"], {"keep": True})["keep"])
    draw.text((6, 8), f"{d.name}: {len(views)} views, {n} labelled targets; {kept} look like the user's photos, "
              f"{passed} pass the VLM review (red: out)", fill=(0, 0, 0), font_size=20)
    for k, v in enumerate(views):
        x, y = (k % cols) * (THUMB[0] + 6), 40 + (k // cols) * (THUMB[1] + 22)
        img = Image.open(d / "render" / f"final_{v['name']}.png").convert("RGB")
        sx, sy = THUMB[0] / img.width, THUMB[1] / img.height
        t = img.resize(THUMB, Image.Resampling.LANCZOS)
        dt = ImageDraw.Draw(t)
        for b in labels.get(v["name"], []):
            x0, y0, x1, y1 = b["box"]
            dt.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 220, 0), width=2)
        out.paste(t, (x, y + 20))
        froms = ",".join(sorted({o["heading_from"].split(" ")[0] for o in v["objects"]})) or "-"
        sim, ok = liked.get(v["name"], (None, True))
        r = rated.get(v["name"], {"score": None, "keep": True})
        draw.text((x + 2, y + 2), f"{v['name']}: {len(labels.get(v['name'], []))}/{len(v['objects'])} targets "
                  f"({froms}) sim {sim} vlm {r['score']}", fill=(0, 0, 0) if ok and r["keep"] else (200, 0, 0),
                  font_size=13)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("places", nargs="+", type=Path)
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--hardware", default="generic")
    ap.add_argument("--pitch", type=float, default=5.0, help="the user camera's pitch, degrees down")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--photos", type=Path, default=ROOT / "data/real/photospheres-scenes",
                    help="the user's photos for the similarity filter (photospheres in it are left out)")
    ap.add_argument("--assets", type=Path, default=ASSET_JOB, help="the asset job (car_assets.py) of the targets")
    ap.add_argument("--context", type=Path, help="context.json measured on the user's photos (context.py): "
                    "targets placed by it")
    ap.add_argument("--label", default="car", help="what the targets are, for the review")
    args = ap.parse_args()
    places = [d.resolve() for d in args.places]
    rng = np.random.default_rng(args.seed)
    if not (args.assets / "fronts.json").exists():
        fronts = measure_fronts(places[0], pool(args.assets), rng)
        (args.assets / "fronts.json").write_text(json.dumps(fronts, indent=1))
        print(f"fronts: {len(fronts)} measured, {sum(f['flip_deg'] > 0 for f in fronts.values())} flipped",
              flush=True)
    cars = pool(args.assets)
    prior = context.prior(json.loads(args.context.read_text())) if args.context else None
    sensor = camera_fx.sensor_for(args.hardware)
    blur = blur_for(places)
    done = {}
    for d in places:
        t0 = time.time()
        w, h = RENDER_SIZE.get(args.hardware, (1024, 768))
        views = extrapolate.plan_views(d, cars, args.views, rng, sensor.fov_deg, (w, h), args.label,
                                       pitch_deg=args.pitch, prior=prior)
        plan = render_plan.write(d, views, sun=None)
        Flow(d).run("sphere_render", "isaac", str(STEPS / "render.py"), ["--dir", d, "--path-tracing"])
        report = json.loads((d / "render" / "render.json").read_text())
        labels = extrapolate.labels(report, plan)
        scale = (sensor.size[0] / w) if sensor.size else 1.0
        for v in plan["views"]:
            img = np.asarray(Image.open(d / "render" / f"render_{v['name']}.png").convert("RGB"))
            obj = np.asarray(Image.open(d / "render" / f"obj_{v['name']}.png")) > 127
            img = replica.match_objects(img, obj, blur[d], 0.0, rng)
            camera_fx.apply(Image.fromarray(img), args.hardware, rng).save(d / "render" / f"final_{v['name']}.png")
            for b in labels[v["name"]]:
                b["box"] = [round(c * scale, 1) for c in b["box"]]
        (d / "render" / "labels.json").write_text(json.dumps(labels, indent=1))
        done[d] = (plan, labels)
        placed = sum(len(v["objects"]) for v in plan["views"])
        print(json.dumps({"place": d.name, "views": len(views), "placed": placed,
                          "labelled": sum(len(b) for b in labels.values()), "blur_px": blur[d],
                          "seconds": round(time.time() - t0)}), flush=True)
    photos = sorted(p.resolve() for p in args.photos.glob("*.jpg") if "PHOTOSPHERE" not in p.name)
    liked = likeness(places, photos) if photos else {"threshold": None, "places": {}}
    for d, (plan, labels) in done.items():
        mine = liked["places"].get(d, {})
        (d / "render" / "likeness.json").write_text(json.dumps({"threshold": liked["threshold"], "views": mine},
                                                               indent=1))
        rated = review.review_views(d, args.label)
        path = d.parent / f"sheet_s6_{d.name}{args.tag}.jpg"
        sheet(d, plan, labels, mine, rated).save(path, quality=88)
        print(f"{d.name}: {sum(ok for _, ok in mine.values())}/{len(mine)} views kept by the similarity filter "
              f"(threshold {liked['threshold']}), {sum(r['keep'] for r in rated.values())} by the VLM review "
              f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
