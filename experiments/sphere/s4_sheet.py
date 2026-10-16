"""S.4 check (docs/roadmap_photosphere.md): on each sphere, views turned around the capture point at the user
camera's field of view, one Objaverse car placed on free ground in each, rendered by Isaac over the proxy and
composited on the erased panorama, then passed through the hardware's camera (camera_fx.py). One JPG sheet per
sphere: per view, the background, the composite and the camera's frame.

    python experiments/sphere/s4_sheet.py data/sphere_test/PXL_* [--hardware esp32cam] [--views 4]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import STEPS, build_background, camera_fx, render_plan
from sim2chip.photo2scene.sphere.sun import sun_for

ROOT = Path(__file__).resolve().parents[2]
CAR_CLASS = "small_vehicle"
CAR_LVIS = "car_(automobile)"  # the class also holds golf carts and the like
DISTANCES_M = (9, 8, 7, 6, 10, 11, 12, 5)  # where to look for free ground, nearest to 8 m first
RENDER_WIDTH = {"esp32cam": 640, "esp32s3": 640}  # twice the sensor's output, so the camera resamples it
THUMB = (360, 270)


def cars() -> list[dict]:
    exclude = set((ROOT / "data/usd/exclude.txt").read_text().split()) if (ROOT / "data/usd/exclude.txt").exists() \
        else set()
    return [a for a in json.loads((ROOT / "data/usd/assets.json").read_text())
            if a["class"] == CAR_CLASS and a.get("lvis") == CAR_LVIS and a["uid"] not in exclude and Path(a["usd"]).exists()]


def plan_views(d: Path, n: int, hardware: str, rng) -> list[dict]:
    sensor = camera_fx.sensor_for(hardware)
    w = RENDER_WIDTH.get(hardware, 1024)
    pool = cars()
    ground = render_plan.Ground(d)
    views = []
    for k in range(n):
        heading = 360 * k / n + rng.uniform(-15, 15)
        car = pool[int(rng.integers(len(pool)))]
        axis = -heading + float(rng.uniform(-30, 30))  # long side across the view, give or take 30 degrees
        spot = ground.place(heading, DISTANCES_M, car["size_m"][0], car["size_m"][1], axis)
        objects, D = [], None
        if spot is not None:
            pos, D = spot
            objects.append({"usd": car["usd"], "length_m": car["size_m"][0], "pos": pos,
                            "yaw_deg": axis + car.get("yaw_fix_deg", 0.0), "label": "car", "uid": car["uid"]})
        views.append({"name": f"v{k}", "yaw_deg": round(heading % 360, 2), "pitch_deg": 0.0,
                      "fov_deg": sensor.fov_deg, "size": [w, w * 3 // 4], "objects": objects,
                      "car_distance_m": D})
    return views


def sheet(d: Path, plan: dict, hardware: str) -> Image.Image:
    rows = len(plan["views"])
    out = Image.new("RGB", (3 * THUMB[0] + 20, 40 + rows * (THUMB[1] + 26)), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), f"{d.name}: background | composite | {hardware} camera", fill=(0, 0, 0), font_size=20)
    for r, v in enumerate(plan["views"]):
        y = 40 + r * (THUMB[1] + 26)
        names = (f"bg_{v['name']}.png", f"render_{v['name']}.png", f"final_{v['name']}.png")
        for c, n in enumerate(names):
            out.paste(Image.open(d / "render" / n).convert("RGB").resize(THUMB, Image.Resampling.LANCZOS),
                      (c * (THUMB[0] + 10), y + 22))
        dist = "no free ground" if v["car_distance_m"] is None else f"car at {v['car_distance_m']} m"
        draw.text((4, y + 2), f"{v['name']}: heading {v['yaw_deg']:.0f}, {dist}, captured {v['captured']:.0%}",
                  fill=(0, 0, 0), font_size=16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("places", nargs="+", type=Path)
    ap.add_argument("--hardware", default="esp32cam")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realtime", action="store_true", help="RTX real time instead of path tracing (flatter "
                    "sky light, about twice as fast)")
    ap.add_argument("--tag", default="", help="suffix of the sheet, to compare runs")
    args = ap.parse_args()
    for d in args.places:
        d = d.resolve()  # the steps run from their own folder
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        build_background(Flow(d), ["car"], [])  # done already: only fills what is missing (the erased votes)
        sun = sun_for(json.loads((d / "sphere.json").read_text()))
        plan = render_plan.write(d, plan_views(d, args.views, args.hardware, rng),
                                 sun=None)  # sky only until S.5 fits the sun's strength (these were after sunset)
        Flow(d).run("sphere_render", "isaac", str(STEPS / "render.py"),
                    ["--dir", d, *([] if args.realtime else ["--path-tracing"])])
        for v in plan["views"]:
            img = Image.open(d / "render" / f"render_{v['name']}.png")
            camera_fx.apply(img, args.hardware, rng).save(d / "render" / f"final_{v['name']}.png")
        path = d.parent / f"sheet_s4_{d.name}{args.tag}.jpg"
        sheet(d, plan, args.hardware).save(path, quality=88)
        print(f"{d.name}: sun {sun}; {time.time() - t0:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
