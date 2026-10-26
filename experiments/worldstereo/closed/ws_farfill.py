"""Far fill, test L0 row A: the splat's far zone painted view by view, the centre left as it is. A pixel belongs to
the far zone when its point lies more than MARGIN metres beyond the walks' reach (the farthest training camera from
the panorama's centre, on the ground plane), or when the splat left it empty and the panorama does not call that
direction sky (Isaac's dome shows the sky). Every model sees the whole render, so it can continue the centre; only
the far zone takes its pixels (grown GROW px, feathered FEATHER px):

- LaMa (SIM2CHIP_SAM3_PYTHON env, bg_prepare.inpaint): takes a mask, so it paints only the empty pixels;
- Qwen-Image-Edit 2511 (SIM2CHIP_ERASE_PYTHON env, erase_qwen.load, Lightning 4 steps): no mask input, the empty
  pixels are painted magenta and it is told to remove them (experiments/sphere_walk/fill.py's prompt);
- NVIDIA Fixer (ws_fixer.py's container): no mask input, it cleans the whole render.

    <gsplat env>/bin/python experiments/worldstereo/ws_farfill.py renders --case photo21_np --arm G2
    <SIM2CHIP_SAM3_PYTHON> experiments/worldstereo/ws_farfill.py paint --case photo21_np --model lama
    <SIM2CHIP_ERASE_PYTHON> experiments/worldstereo/ws_farfill.py paint --case photo21_np --model qwen
    python experiments/worldstereo/ws_farfill.py paint --case photo21_np --model fixer
    python experiments/worldstereo/ws_farfill.py sheet --case photo21_np

Step 2, Difix3D+'s recipe on the far zone only: the same paint on pseudo views (--set pseudo: every walk at
PSEUDO_DIST metres, PSEUDO_YAWS headings), then `views --tag FF|FX` writes splat/views_<tag>.npz, the training views
plus the painted pseudo views whose far zone teaches (ws_splat.py PSEUDO) and whose centre teaches nothing (NONE):

    <gsplat env>/bin/python experiments/worldstereo/ws_farfill.py renders --set pseudo
    <SIM2CHIP_ERASE_PYTHON> experiments/worldstereo/ws_farfill.py paint --set pseudo --model qwen
    python experiments/worldstereo/ws_farfill.py paint --set pseudo --model fixer --src qwen   # FF: Qwen, then Fixer
    python experiments/worldstereo/ws_farfill.py paint --set pseudo --model fixer             # FX: Fixer alone
    python experiments/worldstereo/ws_farfill.py views --set pseudo --tag FF --src fixer_qwen
    python experiments/worldstereo/ws_farfill.py views --set pseudo --tag FX --src fixer
    <gsplat env>/bin/python experiments/worldstereo/ws_splat.py train --case photo21_np --arm FF

Writes data/worldstereo_test/<case>/farfill[_<set>]/: in/, zone/, hole/, <model>/ (NN.png), meta.json, w2c.npy,
sheet.jpg."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

FW, FH = 1024, 576    # Fixer's working size, used by every model
F = ws.F * FH / ws.H  # the frames' focal at this size
MARGIN = 1.0          # metres beyond the walks' reach where the far zone starts
MIN_ALPHA = 0.5       # a splat pixel covered less than this is empty
GROW, FEATHER = 3, 2  # px, as fill.py
MAGENTA = (255, 0, 255)
QWEN_PROMPT = ('Remove the magenta areas from the image, filling them with the scene that continues behind them, '
               'while preserving the remaining elements, maintaining realism and original details.')
FIXER_TIMESTEP = 250  # upstream's examples
PSEUDO_DIST = (0, 3, 6, 9, 12)  # metres along each walk
PSEUDO_YAWS = 8                  # headings per position
MIN_ZONE = 0.05                  # a pseudo view with less far zone than this teaches too little to keep
MIN_HOLE = 0.002                 # Qwen only runs where the splat left at least this much empty

# (label, walk yaw in degrees, metres along it, looking back): mesh_M1's preview views
VIEWS = [("walk 0: 0 m", 4, 0, False), ("walk 0: 4 m", 4, 4, False), ("walk 0: 8 m", 4, 8, False),
         ("walk 0: 12 m", 4, 12, False), ("walk 0: 8 m, looking back", 4, 8, True),
         ("walk 1: 8 m", -180, 8, False), ("walk 1: 8 m, looking back", -180, 8, True),
         ("walk 1: 12 m", -180, 12, False)]


def out_dir(case: str, vset: str = "views") -> Path:
    return ws.DATA / case / ("farfill" if vset == "views" else f"farfill_{vset}")


def pseudo_views(case: str) -> list[tuple[str, np.ndarray]]:
    """Every walk of prep.json at PSEUDO_DIST metres, PSEUDO_YAWS level headings each (the centre only once)."""
    walks = json.loads((ws.DATA / case / "prep.json").read_text())["walks"]
    out = []
    for w, (wyaw, _) in enumerate(walks):
        for d in PSEUDO_DIST:
            if d == 0 and w:
                continue
            pos = np.array([math.cos(math.radians(wyaw)), math.sin(math.radians(wyaw)), 0.0]) * d
            for h in range(PSEUDO_YAWS):
                yaw = wyaw + 360 * h / PSEUDO_YAWS
                out.append((f"walk {w}: {d} m, yaw {yaw:.0f}", ws.w2c(pos, yaw, 0.0)))
    return out


def fname(k: int, vset: str) -> str:
    return f"{k:02d}.png" if vset == "views" else f"{k:03d}.png"


def view_w2c(yaw: float, dist: float, back: bool) -> np.ndarray:
    pos = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0]) * dist
    return ws.w2c(pos, yaw + (180 if back else 0), 0.0)


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1))) > 127


def renders(case: str, arm: str, vset: str):
    """The arm's splat at VIEWS (or the pseudo views), the far zone and the empty non-sky pixels."""
    import torch
    from ws_nbv import lift, load_splat, render
    dev = torch.device("cuda")
    w2c_train = np.load(ws.DATA / case / "splat" / "views.npz")["w2c"]
    reach = float(np.max(np.linalg.norm(np.linalg.inv(w2c_train)[:, :2, 3], axis=1)))
    z = np.load(ws.DATA / case / "distance.npz")
    sky_pano = ~z["mask"]
    K = np.array([[F, 0, FW / 2], [0, F, FH / 2], [0, 0, 1]], np.float32)
    if vset == "views":
        labels, w2cs = [v[0] for v in VIEWS], [view_w2c(y, d, b) for _, y, d, b in VIEWS]
    else:
        labels, w2cs = map(list, zip(*pseudo_views(case)))
    w2cs = np.stack(w2cs).astype(np.float32)
    sp = load_splat(ws.DATA / case / "splat" / "arms" / arm / "splat.ply", dev)
    rgb, dep, alpha = render(sp, w2cs, K, FW, FH)
    dist = torch.linalg.norm(lift(dep, K, w2cs)[..., :2], dim=-1).cpu().numpy()  # on the ground plane
    rgb, alpha = rgb.cpu().numpy(), alpha.cpu().numpy()
    # Sky by direction: the panorama's sky mask at each pixel's ray (sky is at infinity, so the camera's offset
    # from the panorama's centre does not matter).
    j, i = np.meshgrid(np.arange(FW) + 0.5, np.arange(FH) + 0.5)
    cam = np.stack([(j - FW / 2) / F, (i - FH / 2) / F, np.ones_like(j)], -1)
    out = out_dir(case, vset)
    for d in ("in", "zone", "hole"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shares = []
    for k, m in enumerate(w2cs):
        ray = cam @ np.linalg.inv(m)[:3, :3].T
        yaw, pitch = np.arctan2(ray[..., 1], ray[..., 0]), np.arcsin(ray[..., 2] / np.linalg.norm(ray, axis=-1))
        ph, pw = sky_pano.shape
        u = np.clip(((0.5 - yaw / (2 * np.pi)) * pw).astype(int), 0, pw - 1)
        v = np.clip(((0.5 - pitch / np.pi) * ph).astype(int), 0, ph - 1)
        empty = alpha[k] < MIN_ALPHA
        hole = empty & ~sky_pano[v, u]
        far = ~empty & (dist[k] > reach + MARGIN)
        zone = far | hole
        Image.fromarray(rgb[k]).save(out / "in" / fname(k, vset))
        Image.fromarray((zone * 255).astype(np.uint8)).save(out / "zone" / fname(k, vset))
        Image.fromarray((hole * 255).astype(np.uint8)).save(out / "hole" / fname(k, vset))
        shares.append({"zone": round(float(zone.mean()), 3), "hole": round(float(hole.mean()), 3)})
    np.save(out / "w2c.npy", w2cs)
    (out / "meta.json").write_text(json.dumps({"case": case, "arm": arm, "reach_m": round(reach, 2),
                                               "margin_m": MARGIN, "views": labels, "shares": shares}, indent=1))
    print(f"{len(w2cs)} views, reach {reach:.2f} m, far zone from {reach + MARGIN:.2f} m, mean zone "
          f"{np.mean([x['zone'] for x in shares]):.2f}, mean holes {np.mean([x['hole'] for x in shares]):.3f}")


def paste(img: Image.Image, painted: Image.Image, zone: np.ndarray) -> Image.Image:
    """The painted pixels inside the grown, feathered zone; the render elsewhere."""
    zone = grow(zone, GROW)
    a = np.asarray(Image.fromarray((zone * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(FEATHER)),
                   np.float32)[..., None] / 255
    a = np.maximum(a, zone[..., None])
    res = np.asarray(painted.resize(img.size, Image.Resampling.LANCZOS), np.float32) * a + \
        np.asarray(img, np.float32) * (1 - a)
    return Image.fromarray(np.clip(res + 0.5, 0, 255).astype(np.uint8))


def paint(case: str, model: str, vset: str, src: str):
    """model's pixels in the far zone of <src>/ (the renders, in/, or an earlier model's output) -> <model>[_<src>]/"""
    out = out_dir(case, vset)
    names = sorted(p.name for p in (out / "in").glob("*.png"))
    dst = out / (model if src == "in" else f"{model}_{src}")
    dst.mkdir(exist_ok=True)
    times = {}
    if model == "fixer":
        import subprocess
        from ws_fixer import FIXER, IMAGE
        import os
        raw = out / (dst.name + "_raw")
        raw.mkdir(exist_ok=True)
        t = time.time()
        subprocess.run(["docker", "run", "--rm", "--gpus", "all", "--ipc=host", "--user",
                        f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-v", f"{(FIXER / 'src').resolve()}:/work",
                        "-v", f"{(FIXER / 'models').resolve()}:/work/models", "-v", f"{out.resolve()}:/data", IMAGE,
                        "-c", "python /work/src/inference_pretrained_model.py --model "
                              f"/work/models/pretrained/pretrained_fixer.pkl --input /data/{src} --output /data/{raw.name}"
                              f" --timestep {FIXER_TIMESTEP} --resolution 1024"], check=True)
        times["all"] = round(time.time() - t, 1)
        for n in names:
            img = Image.open(out / src / n).convert("RGB")
            zone = np.asarray(Image.open(out / "zone" / n)) > 127
            paste(img, Image.open(raw / n).convert("RGB"), zone).save(dst / n)
    else:
        sys.path.insert(0, str(ws.ROOT / "sim2chip" / "photo2scene" / "steps"))
        if model == "lama":
            from bg_prepare import inpaint
        else:
            import torch
            from erase_qwen import load
            pipe = load(4, lightning=True)
        for n in names:
            img = Image.open(out / "in" / n).convert("RGB")
            zone = np.asarray(Image.open(out / "zone" / n)) > 127
            hole = grow(np.asarray(Image.open(out / "hole" / n)) > 127, GROW)
            t = time.time()
            if vset != "views" and hole.mean() < MIN_HOLE:
                img.save(dst / n)
                continue
            if model == "lama":
                painted = inpaint(img, hole) if hole.any() else img
            else:
                src = np.asarray(img).copy()
                src[hole] = MAGENTA
                painted = pipe(image=[Image.fromarray(src)], prompt=QWEN_PROMPT, negative_prompt=" ",
                               true_cfg_scale=1.0, num_inference_steps=4,
                               generator=torch.Generator("cpu").manual_seed(0)).images[0]
            paste(img, painted, zone).save(dst / n)
            times[n] = round(time.time() - t, 2)
            print(n, times[n], flush=True)
    (out / f"times_{model}.json").write_text(json.dumps(times, indent=1))


def train_views(case: str, vset: str, tag: str, src: str):
    """splat/views_<tag>.npz: views.npz plus the painted pseudo views at the frames' size; only their far zone
    teaches (PSEUDO), and only where the splat or a model left a pixel (sky stays empty)."""
    from ws_fixer import to_frame
    from ws_splat import NONE, PSEUDO
    out = out_dir(case, vset)
    meta = json.loads((out / "meta.json").read_text())
    w2c = np.load(out / "w2c.npy")
    keep = [k for k, s in enumerate(meta["shares"]) if s["zone"] >= MIN_ZONE]
    ims, codes = [], []
    for k in keep:
        n = fname(k, vset)
        ims.append(to_frame(np.asarray(Image.open(out / src / n).convert("RGB"))))
        zone = to_frame((np.asarray(Image.open(out / "zone" / n)) > 127).astype(np.float32)) > 0.5
        codes.append(np.where(zone, PSEUDO, NONE).astype(np.uint8))
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    img, code, depth, w2c0, name = z["img"], z["code"], z["depth"], z["w2c"], z["name"]
    n = len(ims)
    # views.npz holds one depth target per view, or two (proxy, MoGe) for the generated frames: keep its shape
    extra = np.zeros((n,) + depth.shape[1:], np.float16)
    np.savez_compressed(ws.DATA / case / "splat" / f"views_{tag.lower()}.npz",
                        img=np.concatenate([img, np.stack(ims)]), code=np.concatenate([code, np.stack(codes)]),
                        depth=np.concatenate([depth, extra]), w2c=np.concatenate([w2c0, w2c[keep]]),
                        name=np.concatenate([name, [f"farfill_{tag}/{k:03d}" for k in keep]]))
    print(f"views_{tag.lower()}.npz: {len(name)} views + {n} pseudo ({len(meta['views']) - n} with too little far "
          f"zone), {np.mean(np.stack(codes) == PSEUDO):.2f} of their pixels teach")


def sheet(case: str, models: list[str]):
    out = out_dir(case)
    meta = json.loads((out / "meta.json").read_text())
    cw, ch, head = 416, 234, 22
    cols = ["render (G2 splat)", "far zone (red) and holes (magenta)"] + models
    names = sorted(p.name for p in (out / "in").glob("*.png"))
    sh = Image.new("RGB", (cw * len(cols), 30 + len(names) * (ch + head)), (25, 25, 25))
    d = ImageDraw.Draw(sh)
    d.text((6, 8), f"{case}: far zone = beyond {meta['reach_m'] + meta['margin_m']:.1f} m from the centre "
                   f"(walks reach {meta['reach_m']:.1f} m + {meta['margin_m']:.0f} m) or empty and not sky; "
                   f"columns: {' | '.join(cols)}", fill=(255, 255, 255))
    for r, n in enumerate(names):
        y = 30 + r * (ch + head)
        d.text((6, y + 4), f"{meta['views'][r]}  (far zone {meta['shares'][r]['zone']:.0%}, holes "
                           f"{meta['shares'][r]['hole']:.0%})", fill=(255, 255, 0))
        img = Image.open(out / "in" / n).convert("RGB")
        zone = np.asarray(Image.open(out / "zone" / n)) > 127
        hole = np.asarray(Image.open(out / "hole" / n)) > 127
        ov = np.asarray(img, np.float32).copy()
        ov[zone] = ov[zone] * 0.5 + np.array([255, 0, 0]) * 0.5
        ov[hole] = MAGENTA
        tiles = [img, Image.fromarray(ov.astype(np.uint8))]
        tiles += [Image.open(out / m / n).convert("RGB") if (out / m / n).exists() else Image.new("RGB", img.size)
                  for m in models]
        for c, (t, lab) in enumerate(zip(tiles, cols)):
            sh.paste(t.resize((cw, ch)), (c * cw, y + head))
            d.text((c * cw + 4, y + head + 3), lab, fill=(0, 255, 255))
    sh.save(out / "sheet.jpg", quality=88)
    print(out / "sheet.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["renders", "paint", "views", "sheet"])
    ap.add_argument("--set", dest="vset", choices=["views", "pseudo"], default="views")
    ap.add_argument("--src", default="in", help="paint: the images to paint (in, or a model's folder); views: the "
                    "painted folder")
    ap.add_argument("--tag", help="views: the arm the views are for (FF, FX)")
    ap.add_argument("--case", default="photo21_np")
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--model", choices=["lama", "qwen", "fixer"])
    ap.add_argument("--models", nargs="+", default=["lama", "fixer", "qwen"])
    a = ap.parse_args()
    if a.cmd == "renders":
        renders(a.case, a.arm, a.vset)
    elif a.cmd == "paint":
        paint(a.case, a.model, a.vset, a.src)
    elif a.cmd == "views":
        train_views(a.case, a.vset, a.tag, a.src)
    else:
        sheet(a.case, a.models)
