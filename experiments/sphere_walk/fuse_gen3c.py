"""GEN3C's frames painted into our two-layer scene (B_layers), so the site's background is one fixed 3D scene that
Isaac renders from anywhere and GEN3C is not needed again. The geometry stays ours (coherent between views); only
the texels that were never seen (LaMa's paint: the back layer and the front's never-seen ground) take GEN3C's
colour, from every frame that sees them unoccluded (a z-buffer of both layers) and where GEN3C put a surface at
about the same depth (MoGe on the frame, scaled to our scene). Nearer frames weigh more. Run in the
SIM2CHIP_SCENE_PYTHON env (torch, MoGe) after gen3c_batch.py fetch:

    python experiments/sphere_walk/fuse_gen3c.py --dir data/sphere_test/<place>

Writes <dir>/walk/GL_scene/ (scene.npz linked from BL_scene, layers.json, tex_front.png, tex_back.png, info.json),
which render_walk.py --scene takes like BL_scene."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("MOGE_DEPS", "/media/lenena-iker/Crucial/tools/moge_deps"))
sys.path.insert(0, os.environ.get("MOGE_ROOT", "/media/lenena-iker/Crucial/tools/MoGe"))
from layers_ours import occluders  # noqa: E402
from moge.model.v2 import MoGeModel  # noqa: E402
from sim2chip.photo2scene.sphere import capture, render_plan  # noqa: E402

MOGE = os.environ.get("MOGE_CKPT", "/media/lenena-iker/Crucial/tools/moge_ckpt/moge-2-vitl/model.pt")
EVERY = 8          # frames: one in EVERY along each path (the first is our own view)
CELL = 4           # px of the frame per z-buffer cell, so the splatted texels leave no gaps
OCCLUDED = 1.05    # nearer than this times the z-buffer: the texel is what the frame sees
DEPTH_TOL = 0.15   # GEN3C's surface within this share of our depth: it painted the same place
FEATHER = 5        # px of the texture over which the new paint fades in
M3D_PY = "/media/lenena-iker/Crucial/envs/matrix3d/bin/python"  # the env here with a video reader (imageio)


def frames(vid: Path) -> dict:
    """One in EVERY frames of a video (from the second), decoded once into <video>_frames/ by the matrix3d env."""
    fd = vid.with_name(vid.stem + "_frames")
    if not fd.exists():
        fd.mkdir()
        subprocess.run([M3D_PY, "-c", "import sys, imageio.v3 as iio\n"
                        "for k, f in enumerate(iio.imiter(sys.argv[1])):\n"
                        f"    if k and k % {EVERY} == 0: iio.imwrite(f'{{sys.argv[2]}}/f{{k:03d}}.png', f)",
                        str(vid), str(fd)], check=True)
    return {int(p.stem[1:]): np.asarray(Image.open(p).convert("RGB")) for p in sorted(fd.glob("f*.png"))}


def magenta(a):
    return (a[..., 0] > 200) & (a[..., 1] < 90) & (a[..., 2] > 200)


def blur(m, k):
    t = torch.as_tensor(m, dtype=torch.float32, device="cuda")[None, None]
    return F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="replicate"), k, 1)[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--src", help="folder with paths.json and the videos (default <dir>/walk/G_gen3c)")
    ap.add_argument("--out", help="default <dir>/walk/GL_scene")
    args = ap.parse_args()
    d = Path(args.dir)
    w = d / "walk"
    src = Path(args.src) if args.src else w / "G_gen3c"
    out = Path(args.out) if args.out else w / "GL_scene"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    dev = "cuda"
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    pd = np.load(d / "proxy_distance.npz")
    dist, valid, ground = pd["distance"].astype(np.float32), pd["valid"], pd["ground"]
    fg, behind = occluders(dist, valid, ground)

    # Texels of both layers at the back texture's size (2x the distance map), as points in Isaac's world.
    bl = w / "BL_scene"
    back = np.asarray(Image.open(bl / "tex_back.png").convert("RGB"))
    TH, TW = back.shape[:2]
    up = lambda a: np.repeat(np.repeat(a, TH // a.shape[0], 0), TW // a.shape[1], 1)  # noqa: E731
    d_level = capture.pixel_rays(TW, TH) @ R.T  # the texture is in the panorama's frame
    front_full = np.asarray(Image.open(bl / "tex_front.png").convert("RGB"))
    front = np.asarray(Image.fromarray(front_full).resize((TW, TH), Image.Resampling.BILINEAR))
    unseen = magenta(np.asarray(Image.open(w / "tex_proxy.png").convert("RGB").resize((TW, TH), Image.Resampling.NEAREST)))
    layers = {"front": (up(valid), up(dist), front, unseen), "back": (up(fg), up(behind), back, up(fg))}
    pts, info = {}, {}
    for k, (keep, dd, tex, painted) in layers.items():
        idx = np.flatnonzero(keep)
        p = render_plan.to_world(d_level.reshape(-1, 3)[idx] * dd.reshape(-1)[idx, None], h)
        pts[k] = {"idx": torch.as_tensor(idx, device=dev), "p": torch.as_tensor(p, dtype=torch.float32, device=dev),
                  "tex": torch.as_tensor(tex.reshape(-1, 3)[idx], dtype=torch.float32, device=dev),
                  "painted": torch.as_tensor(painted.reshape(-1)[idx], device=dev)}
        pts[k]["acc"] = torch.zeros(len(idx), 3, device=dev)
        pts[k]["wsum"] = torch.zeros(len(idx), device=dev)
        pts[k]["seen"] = torch.zeros(len(idx), dtype=torch.bool, device=dev)  # by some frame, whatever its depth

    moge = MoGeModel.from_pretrained(MOGE).to(dev).eval()
    paths = json.loads((src / "paths.json").read_text())["paths"]
    frames_used, scales = 0, []
    for path in paths:
        vid = src / f"{path['name']}.mp4"
        if not vid.exists():
            print("missing", vid.name)
            continue
        v = path["view"]
        fx = capture.focal(v["size"][0], v["fov_deg"])
        for k, img in frames(vid).items():
            H, W = img.shape[:2]
            pose = torch.as_tensor(np.array(path["poses"][k]), dtype=torch.float32, device=dev)
            proj = {}
            zbuf = torch.full((((H + CELL - 1) // CELL) * ((W + CELL - 1) // CELL),), float("inf"), device=dev)
            for name, L in pts.items():
                c = (L["p"] - pose[:3, 3]) @ pose[:3, :3]  # world to camera (OpenCV axes)
                z = c[:, 2]
                ok = z > 0.3
                u = fx * c[:, 0] / z.clamp(min=1e-3) + W / 2 - 0.5
                vv = fx * c[:, 1] / z.clamp(min=1e-3) + H / 2 - 0.5
                ok &= (u > 0) & (u < W - 1) & (vv > 0) & (vv < H - 1)
                cell = (vv.clamp(0, H - 1) // CELL).long() * ((W + CELL - 1) // CELL) + (u.clamp(0, W - 1) // CELL).long()
                zbuf.scatter_reduce_(0, cell[ok], z[ok], "amin")
                proj[name] = (u, vv, z, ok, cell)
            # The splats leave gaps on the ground and walls: a 3x3 minimum fills them, else far texels show through.
            gw = (W + CELL - 1) // CELL
            zbuf = -F.max_pool2d(-zbuf.reshape(1, 1, -1, gw), 3, 1, 1).reshape(-1)
            x = torch.as_tensor(img, device=dev).permute(2, 0, 1).float() / 255
            with torch.no_grad():
                m = moge.infer(x, fov_x=v["fov_deg"])
            mdep = torch.nan_to_num(m["depth"], nan=0, posinf=0)
            grid = lambda u, vv: torch.stack([u / (W - 1) * 2 - 1, vv / (H - 1) * 2 - 1], -1)[None, None]  # noqa: E731
            look = lambda a, u, vv: F.grid_sample(a[None], grid(u, vv), align_corners=True)[0, :, 0].T  # noqa: E731
            # Scale and colour of the frame against what it sees of the real panorama (the front's seen texels).
            u, vv, z, ok, cell = proj["front"]
            seen = ok & ~pts["front"]["painted"] & (z <= zbuf[cell] * OCCLUDED + 0.05)
            if int(seen.sum()) < 1000:
                continue
            ms = look(mdep[None], u[seen], vv[seen])[:, 0]
            good = ms > 0
            scale = torch.median(z[seen][good] / ms[good])
            col = look(x * 255, u[seen], vv[seen])
            gain = (torch.median(pts["front"]["tex"][seen], 0).values + 1) / (torch.median(col, 0).values + 1)
            gain = gain.clamp(0.7, 1.4)
            scales.append(float(scale))
            for name, L in pts.items():
                u, vv, z, ok, cell = proj[name]
                sel = ok & L["painted"] & (z <= zbuf[cell] * OCCLUDED + 0.05)
                if not bool(sel.any()):
                    continue
                L["seen"] |= sel
                gd = look(mdep[None], u[sel], vv[sel])[:, 0] * scale
                same = (gd - z[sel]).abs() < DEPTH_TOL * z[sel]
                i = torch.nonzero(sel)[:, 0][same]
                wt = 1 / z[i] ** 2
                L["acc"][i] += look(x * 255, u[i], vv[i]) * gain * wt[:, None]
                L["wsum"][i] += wt
            frames_used += 1
        print(path["name"], "done", f"{time.time() - t0:.0f} s", flush=True)

    # The new paint, faded in at its edges, over LaMa's.
    res = {}
    for name, L in pts.items():
        col = torch.zeros(TH * TW, 3, device=dev)
        got = torch.zeros(TH * TW, device=dev)
        has = L["wsum"] > 0
        col[L["idx"][has]] = L["acc"][has] / L["wsum"][has, None]
        got[L["idx"][has]] = 1
        a = (blur(got.reshape(TH, TW).cpu().numpy(), FEATHER) * got.reshape(TH, TW))[..., None]
        old = torch.as_tensor(layers[name][2], dtype=torch.float32, device=dev)
        res[name] = (old * (1 - a) + col.reshape(TH, TW, 3) * a).clamp(0, 255).byte().cpu().numpy(), a[..., 0].cpu().numpy()
        info[f"{name}_painted_texels"] = int(L["painted"].sum())
        if bool(L["painted"].any()):
            info[f"{name}_in_frames"] = round(float(L["seen"][L["painted"]].float().mean()), 3)
            info[f"{name}_from_gen3c"] = round(float(has[L["painted"]].float().mean()), 3)
    Image.fromarray(res["back"][0]).save(out / "tex_back.png")
    # The front keeps the panorama's full resolution: the new paint only where the sphere never saw.
    big = np.asarray(Image.fromarray(unseen.astype(np.uint8) * 255).resize(front_full.shape[1::-1])) > 0
    fr = np.asarray(Image.fromarray(res["front"][0]).resize(front_full.shape[1::-1], Image.Resampling.BICUBIC))
    new_front = front_full.copy()
    new_front[big] = fr[big]
    Image.fromarray(new_front).save(out / "tex_front.png")
    if not (out / "scene.npz").exists():
        os.symlink(os.path.relpath(bl / "scene.npz", out), out / "scene.npz")
    (out / "layers.json").write_text((bl / "layers.json").read_text())
    # The sky for a look around in Isaac's GUI (render_walk.py --save): the erased panorama, levelled.
    from matrix3d_walk import levelled
    sky = levelled(np.asarray(Image.open(d / "background.jpg").convert("RGB")), R, (2048, 1024))
    Image.fromarray(np.clip(sky + 0.5, 0, 255).astype(np.uint8)).save(out / "sky.png")
    info.update(frames=frames_used, depth_scale=[round(float(q), 2) for q in np.percentile(scales, [10, 50, 90])], seconds=round(time.time() - t0, 1))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
