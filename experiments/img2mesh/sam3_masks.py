"""SAM 3 masks for the image-to-3D test (run in the `teacher-sam3` conda env).

For each input photo, SAM 3 segments the object named by its text query and we keep the best instance.
Writes, per photo, into <root>/masks/<stem>/:
    mask.png     full-size binary mask (what SAM 3D takes next to the photo)
    rgba.png     square crop around the mask with the background made transparent (what TRELLIS takes)
    overlay.jpg  the photo with the mask tinted, to check it by eye

Usage:
    python sam3_masks.py --root ~/work/eyeforge/data/img2mesh_test [--only wrench_1,bee_1]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

# Input file prefix -> SAM 3 noun phrase.
QUERIES = {
    'wrench': 'adjustable wrench',
    'bee': 'bee',
    'cam': 'security camera',
}


def square_crop(rgba, mask, margin=0.1):
    """Square crop around the mask, with some margin, clipped to the image."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (1 + 2 * margin) / 2
    box = [max(0, cx - half), max(0, cy - half), min(rgba.width, cx + half), min(rgba.height, cy + half)]
    return rgba.crop([int(v) for v in box])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--only', default=None, help='comma-separated input stems')
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')

    stems = sorted(f[:-4] for f in os.listdir(os.path.join(root, 'inputs')) if f.endswith('.jpg'))
    if args.only:
        stems = [s for s in stems if s in args.only.split(',')]
    report = []
    for stem in stems:
        query = QUERIES[stem.split('_')[0]]
        out = os.path.join(root, 'masks', stem)
        os.makedirs(out, exist_ok=True)
        image = ImageOps.exif_transpose(Image.open(os.path.join(root, 'inputs', stem + '.jpg'))).convert('RGB')

        t0 = time.time()
        inp = proc(images=image, text=query, return_tensors='pt').to('cuda')
        with torch.no_grad():
            res = model(**inp)
        r = proc.post_process_instance_segmentation(
            res, threshold=0.3, mask_threshold=0.5, target_sizes=inp.get('original_sizes').tolist())[0]
        dt = time.time() - t0
        if len(r['scores']) == 0:
            print({'stem': stem, 'found': 0}, flush=True)
            report.append({'stem': stem, 'query': query, 'found': 0})
            continue
        i = int(r['scores'].argmax())
        mask = r['masks'][i].cpu().numpy().astype(bool)

        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out, 'mask.png'))
        rgba = image.copy()
        rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
        square_crop(rgba, mask).save(os.path.join(out, 'rgba.png'))
        tint = np.asarray(image).astype(np.float32)
        tint[mask] = 0.5 * tint[mask] + 0.5 * np.array([255, 0, 0])
        Image.fromarray(tint.astype(np.uint8)).save(os.path.join(out, 'overlay.jpg'), quality=85)

        row = {'stem': stem, 'query': query, 'found': len(r['scores']), 'score': round(float(r['scores'][i]), 3),
               'mask_frac': round(float(mask.mean()), 4), 'seconds': round(dt, 2)}
        print(row, flush=True)
        report.append(row)

    with open(os.path.join(root, 'masks', 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
