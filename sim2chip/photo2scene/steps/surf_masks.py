"""Surface route, step 1: where each surface phrase is in each photo (SAM 3, run in the `teacher-sam3` env):

    python surf_masks.py --photos photos.txt --phrases "grass,concrete,dirt" --out masks/ [--side 336]

photos.txt lists one image path per line. Writes <out>/<stem>.npz per photo: masks (phrases, H, W) bool, the union
of the phrase's instances, at the photo's shape with its long side --side; and <out>/phrases.json. A photo whose
npz exists is skipped.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', required=True)
    ap.add_argument('--phrases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--side', type=int, default=336)
    ap.add_argument('--threshold', type=float, default=0.4)
    args = ap.parse_args()
    phrases = [p.strip() for p in args.phrases.split(',') if p.strip()]
    photos = [line.strip() for line in open(args.photos) if line.strip()]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'phrases.json'), 'w') as f:
        json.dump(phrases, f)
    todo = [p for p in photos if not os.path.exists(os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + '.npz'))]
    if not todo:
        return
    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    for p in todo:
        image = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        image.thumbnail((1008, 1008))
        w, h = image.size
        k = args.side / max(w, h)
        size = (max(1, round(w * k)), max(1, round(h * k)))
        out = np.zeros((len(phrases), size[1], size[0]), bool)
        for i, phrase in enumerate(phrases):
            inp = proc(images=image, text=phrase, return_tensors='pt').to('cuda')
            with torch.no_grad():
                res = model(**inp)
            r = proc.post_process_instance_segmentation(
                res, threshold=args.threshold, mask_threshold=0.5, target_sizes=inp.get('original_sizes').tolist())[0]
            if len(r['masks']):
                union = r['masks'].any(0).cpu().numpy()
                out[i] = np.asarray(Image.fromarray(union).resize(size, Image.NEAREST))
        np.savez_compressed(os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + '.npz'), masks=out)
        print(os.path.basename(p), ' '.join(f'{ph}={m.mean():.2f}' for ph, m in zip(phrases, out)), flush=True)


if __name__ == '__main__':
    main()
