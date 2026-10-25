"""Walk test arm B, the holes of arm A's views painted in 2D, view by view: LaMa (SIM2CHIP_SAM3_PYTHON env, as
bg_prepare.py) or Qwen-Image-Edit 2511 with the Object-Remover and Lightning LoRAs (SIM2CHIP_ERASE_PYTHON env, as
erase_qwen.py), told to remove the magenta areas. Only the hole (grown by GROW px, feathered) takes the model's
pixels: the rest of the view stays the mesh's.

    python experiments/sphere_walk/fill.py --dir data/sphere_test/<place> --model lama|qwen [--src B_layers]
        [--only h000_f5,...]

Writes <dir>/walk/B_<model>/NAME.png and fill_<model>.json (seconds per view)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
GROW = 3      # px the hole grows before painting: the mesh's torn edges around it go too
FEATHER = 2   # px of blur on the paste edge
QWEN_PROMPT = ('Remove the magenta areas from the image, filling them with the scene that continues behind them, '
               'while preserving the remaining elements, maintaining realism and original details.')


def grow(mask, px):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * px + 1))) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", choices=["lama", "qwen"], required=True)
    ap.add_argument("--only", default="", help="comma-separated view names (default all)")
    ap.add_argument("--src", default="A", help="the arm whose holes are painted (A, or B_layers for what its back "
                    "layer left); the output is B_<model> for A, else <src>_<model>")
    args = ap.parse_args()
    w = Path(args.dir) / "walk"
    out = w / (f"B_{args.model}" if args.src == "A" else f"{args.src}_{args.model}")
    holes_dir = w / ("holes" if args.src == "A" else f"{args.src}_holes")
    out.mkdir(exist_ok=True)
    names = [v["name"] for v in json.loads((w / "views.json").read_text())["views"]]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]

    if args.model == "lama":
        sys.path.insert(0, str(ROOT / "sim2chip" / "photo2scene" / "steps"))
        from bg_prepare import inpaint

        def paint(img, mask):
            return inpaint(img, mask)
    else:
        import torch
        sys.path.insert(0, str(ROOT / "sim2chip" / "photo2scene" / "steps"))
        from erase_qwen import load
        pipe = load(4, lightning=True)

        def paint(img, mask):
            return pipe(image=[img], prompt=QWEN_PROMPT, negative_prompt=" ", true_cfg_scale=1.0,
                        num_inference_steps=4, generator=torch.Generator("cpu").manual_seed(0)).images[0].resize(
                img.size, Image.Resampling.LANCZOS)

    report = {}
    for n in names:
        img = Image.open(w / args.src / f"{n}.png").convert("RGB")
        hole = grow(np.asarray(Image.open(holes_dir / f"{n}.png")) > 127, GROW)
        t = time.time()
        if not hole.any():
            img.save(out / f"{n}.png")
            report[n] = 0.0
            continue
        filled = paint(img, hole)
        alpha = np.asarray(Image.fromarray((hole * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(FEATHER)), np.float32)[..., None] / 255
        alpha = np.maximum(alpha, hole[..., None])  # the hole itself is always the model's
        res = np.asarray(filled, np.float32) * alpha + np.asarray(img, np.float32) * (1 - alpha)
        Image.fromarray(np.clip(res + 0.5, 0, 255).astype(np.uint8)).save(out / f"{n}.png")
        report[n] = round(time.time() - t, 2)
        print(n, report[n], flush=True)
    prev = w / f"fill_{out.name}.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    prev.write_text(json.dumps({**old, **report}, indent=1))


if __name__ == "__main__":
    main()
