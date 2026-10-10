"""Object removal, faster candidates against Qwen-Image-Edit-2511 at 40 steps (erase_qwen.py), on a photo whose
things to erase are already known (bg_prepare.py instances.npz). Every method gets the same photo and the same
names or mask:

- qwen_light: Qwen-Image-Edit-2511 in 4 bits with the Object-Remover LoRA and the Lightning 4-step LoRA
  (lightx2v, Apache 2.0);
- klein: FLUX.2 [klein] 4B, distilled, 4 steps (Black Forest Labs, Apache 2.0), told the names;
- rorem: RORem (SDXL inpainting trained for removal, Apache 2.0), given the erase mask.

Scores per method: seconds, the instances still there (mean change under 12 inside the mask, as erase_qwen.py)
and the change outside the grown mask (should stay near 0: the rest of the photo is the scene's texture).
Run in /media/lenena-iker/Crucial/envs/eyeforge-qwenedit with HF_HOME on Crucial:

    python experiments/scene3d/erase_bench.py --dir data/scene3d_test/flow/user_desk --methods qwen_light,klein,rorem

Writes <dir>/erase_bench/<method>.png, bench.json and sheet.jpg (photo, LaMa, Qwen 40 steps, each method).
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sim2chip', 'photo2scene', 'steps'))  # noqa: E402
from erase_qwen import PROMPT, left_behind, load


def names_prompt(names):
    names = list(dict.fromkeys(names))
    return PROMPT.format(objects=', '.join(names[:-1]) + (' and ' if len(names) > 1 else '') + names[-1])


def qwen_light(photo, names, mask):
    pipe = load(4, lightning=True)
    t = time.time()
    out = pipe(image=[photo], prompt=names_prompt(names), negative_prompt=' ', true_cfg_scale=1.0,
               num_inference_steps=4, generator=torch.Generator('cpu').manual_seed(0)).images[0]
    return out, time.time() - t, pipe


def klein(photo, names, mask):
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    w, h = photo.size
    t = time.time()
    out = pipe(image=photo, prompt=names_prompt(names), height=h // 16 * 16, width=w // 16 * 16,
               num_inference_steps=4, guidance_scale=1.0, generator=torch.Generator('cpu').manual_seed(0)).images[0]
    return out, time.time() - t, pipe


def rorem(photo, names, mask):
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained('LetsThink/RORem', torch_dtype=torch.float16).to('cuda')
    w, h = photo.size
    s = 1024 / max(w, h)  # SDXL's own size; RORem's README runs 512 squares
    size = (round(w * s / 8) * 8, round(h * s / 8) * 8)
    m = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(21))
    t = time.time()
    out = pipe(prompt='', image=photo.resize(size), mask_image=m.resize(size), height=size[1], width=size[0],
               num_inference_steps=20, guidance_scale=1.0, strength=0.99,
               generator=torch.Generator('cuda').manual_seed(0)).images[0]
    return out, time.time() - t, pipe


METHODS = {'qwen_light': qwen_light, 'klein': klein, 'rorem': rorem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--methods', default='qwen_light,klein,rorem')
    ap.add_argument('--left', type=float, default=12.0)
    args = ap.parse_args()
    d = args.dir
    out_dir = os.path.join(d, 'erase_bench')
    os.makedirs(out_dir, exist_ok=True)
    photo = Image.open(os.path.join(d, 'photo.png')).convert('RGB')
    z = np.load(os.path.join(d, 'instances.npz'))
    masks, names = z['masks'].astype(bool), [str(p) for p in z['phrases']]
    union = np.any(masks, 0)
    grown = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(41))) > 0
    path = os.path.join(out_dir, 'bench.json')
    bench = json.load(open(path)) if os.path.exists(path) else {}

    def score(img, seconds):
        img = img.convert('RGB').resize(photo.size, Image.Resampling.LANCZOS)
        a, b = np.asarray(photo, np.float32), np.asarray(img, np.float32)
        left = left_behind(photo, img, masks, args.left)
        return img, {'seconds': round(seconds, 1), 'left': len(left), 'of': len(masks),
                     'left_names': [names[i] for i in left],
                     'change_outside': round(float(np.abs(a[~grown] - b[~grown]).mean()), 2)}

    refs = {'lama': 'background_lama.png', 'qwen40': 'background_qwen_raw.png'}
    for k, f in refs.items():
        if os.path.exists(os.path.join(d, f)):
            img, s = score(Image.open(os.path.join(d, f)), 0)
            img.save(os.path.join(out_dir, f'{k}.png'))
            bench[k] = {**s, 'seconds': {'lama': 5, 'qwen40': 260}[k]}  # from the pipeline's own logs
    for m in [x.strip() for x in args.methods.split(',') if x.strip()]:
        out, seconds, pipe = METHODS[m](photo, names, union)
        img, s = score(out, seconds)
        img.save(os.path.join(out_dir, f'{m}.png'))
        bench[m] = s
        print(m, json.dumps(s), flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    json.dump(bench, open(path, 'w'), indent=1)

    tiles = [('photo', photo)] + [(k, Image.open(os.path.join(out_dir, f'{k}.png'))) for k in bench]
    tw = 640
    th = round(tw * photo.height / photo.width)
    sheet = Image.new('RGB', (tw * 3, (th + 24) * -(-len(tiles) // 3)), 'white')
    dr = ImageDraw.Draw(sheet)
    for i, (k, t) in enumerate(tiles):
        x, y = (i % 3) * tw, (i // 3) * (th + 24)
        sheet.paste(t.resize((tw, th)), (x, y + 24))
        info = bench.get(k)
        label = k if not info else f"{k}: {info['seconds']} s, left {info['left']}/{info['of']}, " \
                                   f"outside {info['change_outside']}"
        dr.text((x + 6, y + 6), label, fill=(0, 0, 0))
    sheet.save(os.path.join(out_dir, 'sheet.jpg'), quality=90)


if __name__ == '__main__':
    main()
