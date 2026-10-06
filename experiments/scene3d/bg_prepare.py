"""Step 1 of our background from one photo: SAM 3 masks for the objects and the support surface, then the objects
erased with LaMa (Apache 2.0) so the photo becomes an empty background. Run in the `teacher-sam3` env:

    python bg_prepare.py --image data/real/count/x.jpg --objects "bottle,bottle cap,moka pot" --support table \
        --out data/scene3d_test/ours/bottles

With --shadows the objects' cast shadows are erased too (sunny scenes).
Writes photo.png (resized), support.png (support plus erased pixels), support_raw.png (SAM 3 only), objects.png (all object masks), instances.npz (one mask per instance and
its phrase) and background.png (objects erased).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

LAMA = '/media/lenena-iker/Crucial/tools/lama/big-lama.pt'


def segment(model, proc, image, phrase, threshold):
    inp = proc(images=image, text=phrase, return_tensors='pt').to('cuda')
    with torch.no_grad():
        res = model(**inp)
    r = proc.post_process_instance_segmentation(
        res, threshold=threshold, mask_threshold=0.5, target_sizes=inp.get('original_sizes').tolist())[0]
    order = r['scores'].argsort(descending=True)
    return [(float(r['scores'][i]), r['masks'][i].cpu().numpy().astype(bool)) for i in order]


def dilate(mask, px):
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return (F.max_pool2d(t, 2 * px + 1, stride=1, padding=px)[0, 0] > 0).numpy()


def reconstruct(seed, mask):
    """Parts of mask connected to seed (morphological reconstruction by repeated 3x3 dilation inside mask)."""
    m = torch.from_numpy(mask).cuda()
    cur = torch.from_numpy(seed & mask).cuda()
    while True:
        nxt = (F.max_pool2d(cur[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0) & m
        if torch.equal(nxt, cur):
            return cur.cpu().numpy()
        cur = nxt


def inpaint(image, mask):
    """LaMa on the whole image; sides padded to a multiple of 8."""
    lama = torch.jit.load(LAMA, map_location='cuda').eval()
    img = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)[None].float().cuda() / 255
    msk = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
    h, w = img.shape[2:]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    img, msk = F.pad(img, (0, pw, 0, ph), mode='reflect'), F.pad(msk, (0, pw, 0, ph))
    with torch.no_grad():
        out = lama(img, msk)[0, :, :h, :w]
    return Image.fromarray((out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--objects', required=True, help='comma-separated phrases to erase')
    ap.add_argument("--support", required=True, help="surface the objects rest on (table, floor, road); erased pixels count as support")
    ap.add_argument('--out', required=True)
    ap.add_argument('--long-side', type=int, default=1344)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--shadows', action='store_true', help='also erase the cast shadows touching the objects')
    ap.add_argument('--shadow-threshold', type=float, default=0.3)
    ap.add_argument('--grow', type=float, default=0.03, help='mask growth before erasing, fraction of long side')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    photo = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    s = args.long_side / max(photo.size)
    photo = photo.resize((round(photo.width * s), round(photo.height * s)), Image.Resampling.LANCZOS)
    photo.save(os.path.join(args.out, 'photo.png'))

    model = Sam3Model.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3Processor.from_pretrained('facebook/sam3')
    masks, phrases = [], []
    for phrase in [p.strip() for p in args.objects.split(',')]:
        for _, m in segment(model, proc, photo, phrase, args.threshold):
            if not any((m & k).sum() / max((m | k).sum(), 1) > 0.7 for k in masks):
                masks.append(m)
                phrases.append(phrase)
    support = np.zeros(photo.size[::-1], bool)
    for _, m in segment(model, proc, photo, args.support, args.threshold):
        support |= m
    objects = np.any(masks, axis=0) if masks else np.zeros_like(support)
    grow = max(2, round(args.grow * args.long_side))
    # Cast shadows of the objects: SAM 3 'shadow' regions on the support that connect to an object. Left in the
    # background they would be shadows without an owner, and they are what the light fit in Isaac compares against.
    shadows = np.zeros_like(support)
    if args.shadows:
        for _, m in segment(model, proc, photo, 'shadow', args.shadow_threshold):
            shadows |= m
        shadows &= dilate(support, grow)
        shadows = reconstruct(dilate(objects, grow) & shadows, shadows)
    del model
    torch.cuda.empty_cache()

    erase = dilate(objects | shadows, grow)
    background = inpaint(photo, erase)
    background.save(os.path.join(args.out, 'background.png'))
    for name, m in [('support', support | erase), ('support_raw', support), ('objects', objects), ('shadows', shadows), ('erased', erase)]:
        Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(args.out, f'{name}.png'))
    np.savez_compressed(os.path.join(args.out, 'instances.npz'), masks=np.array(masks), phrases=np.array(phrases))
    with open(os.path.join(args.out, 'prepare.json'), 'w') as f:
        json.dump({'image': args.image, 'size': photo.size, 'support': args.support,
                   'instances': phrases, 'support_frac': round(float(support.mean()), 3),
                   'shadow_frac': round(float(shadows.mean()), 3)}, f, indent=2)
    print(json.dumps({'instances': phrases, 'support_frac': round(float(support.mean()), 3),
                      'shadow_frac': round(float(shadows.mean()), 3)}))


if __name__ == '__main__':
    main()
