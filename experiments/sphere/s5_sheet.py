"""S.5 check (docs/roadmap_photosphere.md): the replica. On each sphere the erased cars go back where they stood
(replica.py fits a box on the ground to each mask, turned as Orient Anything V2 says it faces, and picks the
Objaverse vehicle closest to its size; the same model on a first render finds each asset's front), Isaac renders
them in a view centred on each over the erased panorama, and the same view of the original panorama is the
reference. One JPG sheet per sphere: per view, the real cut, the replica and the replica with the objects given
the camera's measured blur and noise; the measures go to <place>/replica/measure.json.

    python experiments/sphere/s5_sheet.py data/sphere_test/PXL_* [--realtime]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from s4_sheet import ROOT
from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import STEPS, replica

THUMB = (400, 300)


PASSENGER_LVIS = ("car_(automobile)", "cab_(taxi)", "police_cruiser")


def vehicles() -> list[dict]:
    """The converted passenger cars not excluded by hand. The whole small_vehicle class put lifted pickups and a
    Hummer where low cars stood, with daylight under them (2026-10-17); in the product the targets' assets come
    from fetch_assets and qa_assets."""
    exclude = set((ROOT / "data/usd/exclude.txt").read_text().split())
    return [a for a in json.loads((ROOT / "data/usd/assets.json").read_text())
            if a.get("lvis") in PASSENGER_LVIS and a["uid"] not in exclude and Path(a["usd"]).exists()]


def sheet(d: Path, report: dict) -> Image.Image:
    views = report["views"]
    out = Image.new("RGB", (3 * THUMB[0] + 20, 70 + len(views) * (THUMB[1] + 26)), "white")
    draw = ImageDraw.Draw(out)
    cam, s = report["camera"], report["summary"]
    draw.text((8, 6), f"{d.name}: real | replica | replica, objects with the camera's blur and noise",
              fill=(0, 0, 0), font_size=20)
    draw.text((8, 34), f"median IoU {s['iou']}, brightness replica/real {s['brightness']}, contact shadow real "
              f"{s['shadow_real']} replica {s['shadow_replica']}; camera blur {cam['blur_px']} px, "
              f"noise +{cam['noise']}", fill=(0, 0, 0), font_size=16)
    for r, v in enumerate(views):
        y = 70 + r * (THUMB[1] + 26)
        for c, n in enumerate((f"real_{v['name']}.png", f"render_{v['name']}.png", f"matched_{v['name']}.png")):
            out.paste(Image.open(d / "replica" / n).convert("RGB").resize(THUMB, Image.Resampling.LANCZOS),
                      (c * (THUMB[0] + 10), y + 22))
        sh = v["shadow"] or {}
        draw.text((4, y + 2), f"{v['name']}: IoU {v['iou']}, brightness {v['brightness']}, shadow real "
                  f"{sh.get('real')} replica {sh.get('replica')}", fill=(0, 0, 0), font_size=16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("places", nargs="+", type=Path)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sky", default="uniform", choices=("uniform", "overcast"))
    args = ap.parse_args()
    pools = {"car": vehicles()}
    for d in args.places:
        d = d.resolve()
        t0 = time.time()
        flow = Flow(d)
        rep = d / "replica"

        def orient(name, crops):
            flow.run(f"sphere_orient_{name}", "sam3", str(STEPS / "orient.py"),
                     ["--crops", crops, "--out", rep / f"orient_{name}.json"])
            return json.loads((rep / f"orient_{name}.json").read_text())

        def render():
            flow.run("sphere_replica", "isaac", str(STEPS / "render.py"),
                     ["--dir", d, "--sub", "replica", *([] if args.realtime else ["--path-tracing"])])

        looked = replica.cutouts(d, rep / "cut_real")
        heads = replica.headings(orient("real", rep / "cut_real"), looked)
        plan = replica.plan(d, pools, heads, sky=args.sky)
        if not plan["views"]:
            print(f"{d.name}: no target to replicate", flush=True)
            continue
        render()
        # The assets' fronts: the same model on the first render, then the backwards ones turned round.
        looked = replica.rendered_cutouts(d, rep / "cut_render")
        flips = replica.asset_flips(d, replica.headings(orient("render", rep / "cut_render"), looked))
        (rep / "flips.json").write_text(json.dumps(flips, indent=1))
        if any(flips.values()):
            replica.plan(d, pools, heads, flips, sky=args.sky)
            render()
        report = replica.measure(d, np.random.default_rng(0))
        path = d.parent / f"sheet_s5_{d.name}{args.tag}.jpg"
        sheet(d, report).save(path, quality=88)
        print(json.dumps({"place": d.name, **report["summary"], "camera": report["camera"]}), flush=True)
        print(f"{d.name}: {time.time() - t0:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
