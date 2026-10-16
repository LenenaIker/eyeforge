"""S.3 of the sphere route, erase: the objects instances.py found (targets and movable things) taken out of the
working panorama with Qwen-Image-Edit 2511 and the Object-Remover LoRA (erase_qwen.py's pipeline and its check of
what was left). Each object is erased once, in a view made for it: neighbouring objects are grouped, the view is
centred on the group and just wide enough for it, and the edit goes back to the panorama only inside the group's
masks, grown a little for the contact shadows and the masks' own edges. Overlapping cuts erased apart would fill
the same pixels differently and leave ghosts where they meet (docs/conocimiento_photosphere.md). The groups go one
after another on the panorama as it is being erased, so a later view sees the earlier fills. Run in the erase
Python (SIM2CHIP_ERASE_PYTHON):

    python erase.py --dir data/sphere_test/<place>

Writes <dir>/background.jpg (the erased working panorama), <dir>/erased.png (where it changed),
<dir>/erase/NN_before.jpg and NN_after.jpg (each group's view) and <dir>/erase.json (the groups: objects, view,
passes, what was left, seconds).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # the sphere's capture.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "steps"))  # photo2scene's erase_qwen.py
import capture  # noqa: E402
from erase_plan import group, view_for  # noqa: E402
from erase_qwen import left_behind, load, prompt_for  # noqa: E402

GROW = 0.025           # the paste mask grown by this share of the view's width
TOUCH = 0.004          # objects whose masks come this close (share of the panorama's width) are neighbours
FEATHER = 0.006        # and blurred by this share


def neighbours(masks: dict) -> dict:
    """For each object, the others whose masks touch it on the panorama."""
    ids = list(masks)
    H, W = next(iter(masks.values())).shape
    r = max(1, round(TOUCH * W))
    grown = {i: torch.nn.functional.max_pool2d(torch.from_numpy(masks[i]).float()[None, None], 2 * r + 1, stride=1,
                                               padding=r)[0, 0].numpy() > 0 for i in ids}
    return {i: [j for j in ids if j != i and (grown[i] & masks[j].astype(bool)).any()] for i in ids}


def erase_group(pipe, pano, pano_rays, changed, masks, objs, name, d, args, seed, paste_ids=()):
    """Erase one group in its own view and paste the edit back on the panorama and the changed map (in place).
    The paste also covers paste_ids, the group's neighbours: an earlier erase may have drawn there the hidden part
    of an object of this group (corner_cars 2026-10-16: a car erased in front of a van gave the van its lower
    half, outside the van's mask, so the van's own erase left it). Returns its report."""
    PW = pano.shape[1]
    t = time.time()
    view = view_for(objs, PW / 360)
    rays = capture.view_rays(view)
    before = Image.fromarray(np.clip(capture.sample(pano, rays) + 0.5, 0, 255).astype(np.uint8))
    view_masks = [capture.sample(masks[o["id"]], rays) > 0.5 for o in objs]
    out, todo, passes = before, list(range(len(objs))), []
    for p in range(args.passes):
        out = pipe(image=[out], prompt=prompt_for(objs[i]["phrase"] for i in todo), negative_prompt=" ",
                   true_cfg_scale=1.0, num_inference_steps=4,
                   generator=torch.Generator("cpu").manual_seed(seed + p)).images[0]
        out = out.resize(before.size, Image.Resampling.LANCZOS)
        todo = [todo[i] for i in left_behind(before, out, [view_masks[i] for i in todo], args.left)]
        passes.append([objs[i]["phrase"] for i in todo])
        if not todo:
            break
    w = view["size"][0]
    r = max(1, round(GROW * w))
    paste_masks = view_masks + [capture.sample(masks[i], rays) > 0.5 for i in paste_ids]
    grown = torch.nn.functional.max_pool2d(torch.from_numpy(np.any(paste_masks, 0)).float()[None, None],
                                           2 * r + 1, stride=1, padding=r)[0, 0].numpy()
    paste = Image.fromarray((grown * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(1, FEATHER * w)))
    before.save(os.path.join(d, "erase", f"{name}_before.jpg"), quality=90)
    Image.composite(out, before, paste).save(os.path.join(d, "erase", f"{name}_after.jpg"), quality=90)
    # Back on the panorama: each pixel the view sees, blended by the paste mask at its place in the view.
    x, y, inside = capture.project(view, pano_rays)
    sel = np.flatnonzero(inside)
    xi, yi = np.clip(np.round(x.ravel()[sel]), 0, w - 1).astype(int), \
        np.clip(np.round(y.ravel()[sel]), 0, view["size"][1] - 1).astype(int)
    alpha = np.asarray(paste, np.float32)[yi, xi] / 255
    hit = alpha > 0
    sel, alpha, xi, yi = sel[hit], alpha[hit], xi[hit], yi[hit]
    src = np.asarray(out, np.float32)[yi, xi]
    flat = pano.reshape(-1, 3)
    flat[sel] = np.clip(flat[sel] * (1 - alpha[:, None]) + src * alpha[:, None] + 0.5, 0, 255).astype(np.uint8)
    changed.reshape(-1)[sel] = np.maximum(changed.reshape(-1)[sel], alpha)
    rep = {"name": name, "objects": [o["id"] for o in objs], "phrases": [o["phrase"] for o in objs], "view": view,
           "passes": passes, "left": [objs[i]["id"] for i in todo], "seconds": round(time.time() - t, 1)}
    print(f"{name}: {len(objs)} objects, view {view}, left {rep['left']}, {rep['seconds']} s", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--passes", type=int, default=3, help="Qwen passes per group; later ones name what is left")
    ap.add_argument("--left", type=float, default=12.0, help="mean change (0-255) under which an object is left")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = args.dir
    t0 = time.time()
    pano = np.asarray(Image.open(os.path.join(d, "pano.jpg")).convert("RGB")).copy()
    PH, PW = pano.shape[:2]
    inst = json.load(open(os.path.join(d, "instances.json")))
    z = np.load(os.path.join(d, "instances.npz"))
    MW, MH = (int(v) for v in z["size"])
    masks = {o["id"]: np.unpackbits(z["masks"][i])[:MW * MH].reshape(MH, MW) for i, o in enumerate(inst["objects"])}
    pipe = load(4, lightning=True, gpu=os.environ.get("SIM2CHIP_ERASE_GPU") == "1")
    changed = np.zeros((PH, PW), np.float32)
    pano_rays = capture.pixel_rays(PW, PH)
    os.makedirs(os.path.join(d, "erase"), exist_ok=True)
    report = []
    near = neighbours(masks)
    near_of = lambda objs: sorted({j for o in objs for j in near[o["id"]]} - {o["id"] for o in objs})  # noqa: E731
    for k, objs in enumerate(group(inst["objects"])):
        report.append(erase_group(pipe, pano, pano_rays, changed, masks, objs, f"{k:02d}", d, args, args.seed,
                                  near_of(objs)))
    # What a group left gets a second chance alone, in a view fitted to it, with other seeds: a crowded view of
    # big, touching cars was the eraser's failure (corner_cars 2026-10-16: 4 of 7 left after 3 passes).
    by_id = {o["id"]: o for o in inst["objects"]}
    left = []
    for i in [i for r in list(report) for i in r["left"]]:
        rep = erase_group(pipe, pano, pano_rays, changed, masks, [by_id[i]], f"retry{i:02d}", d, args,
                          args.seed + 100, near_of([by_id[i]]))
        report.append(rep)
        left += rep["left"]
    Image.fromarray(pano).save(os.path.join(d, "background.jpg"), quality=95)
    Image.fromarray((changed * 255).astype(np.uint8)).save(os.path.join(d, "erased.png"))
    json.dump({"groups": report, "left": left, "seconds": round(time.time() - t0, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)},
              open(os.path.join(d, "erase.json"), "w"), indent=1)
    print(f"{len(report)} groups in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
