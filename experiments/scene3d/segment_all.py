"""Segment everything in a photo, without naming anything: SAM 3's point prompts (Sam3TrackerModel, the SAM 2
style interactive head) on a grid over the whole picture, three masks per point (part, object, group), the
unsure and unstable ones dropped and the duplicates merged, as SAM 1 and 2's automatic mask generator did.
What each mask is, and which scale is a whole thing, is decided afterwards (scene_split.py). Run in the
`teacher-sam3` env:

    python experiments/scene3d/segment_all.py --dir data/scene3d_test/flow/user_desk

Reads <dir>/photo.png; writes <dir>/all_masks.npz (masks: bool N x H x W, score, stability).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor


def stability(logits, offset=1.0):
    """IoU of the mask thresholded at +offset and -offset: a mask that changes with the threshold is unsure."""
    hi, lo = (logits > offset).sum((-1, -2)), (logits > -offset).sum((-1, -2))
    return hi.float() / lo.clamp(min=1).float()


def nms(masks, score, iou_max, small):
    """Greedy mask NMS on a downscaled copy: the best score keeps its place, near duplicates go."""
    m = F.interpolate(masks[:, None].float(), size=small, mode='area')[:, 0].flatten(1) > 0.5
    order = torch.argsort(score, descending=True)
    keep = []
    for i in order.tolist():
        if not keep:
            keep.append(i)
            continue
        k = m[keep]
        inter = (k & m[i]).sum(1).float()
        union = (k | m[i]).sum(1).float().clamp(min=1)
        if (inter / union).max() < iou_max:
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--grid', type=int, default=32, help='points along the long side')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--min-score', type=float, default=0.75, help="SAM's own IoU guess")
    ap.add_argument('--min-stability', type=float, default=0.85)
    ap.add_argument('--min-area', type=float, default=0.0005, help='share of the photo')
    ap.add_argument('--nms', type=float, default=0.7)
    args = ap.parse_args()
    photo = Image.open(os.path.join(args.dir, 'photo.png')).convert('RGB')
    w, h = photo.size
    model = Sam3TrackerModel.from_pretrained('facebook/sam3').cuda().eval()
    proc = Sam3TrackerProcessor.from_pretrained('facebook/sam3')
    nx = args.grid if w >= h else max(2, round(args.grid * w / h))
    ny = args.grid if h > w else max(2, round(args.grid * h / w))
    xs, ys = (np.arange(nx) + 0.5) * w / nx, (np.arange(ny) + 0.5) * h / ny
    pts = [[float(x), float(y)] for y in ys for x in xs]
    inputs = proc(images=photo, return_tensors='pt').to('cuda')
    with torch.no_grad():
        emb = model.get_image_embeddings(inputs['pixel_values'])
    masks, score, stab = [], [], []
    for i in range(0, len(pts), args.batch):
        chunk = pts[i:i + args.batch]
        p = proc(images=photo, input_points=[[[q] for q in chunk]], input_labels=[[[1] for _ in chunk]],
                 return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model(image_embeddings=emb, input_points=p['input_points'], input_labels=p['input_labels'],
                        multimask_output=True)
        logits = proc.post_process_masks(out.pred_masks.cpu(), p['original_sizes'].cpu(), binarize=False)[0]
        logits = logits.flatten(0, 1)  # points x 3 scales
        sc = out.iou_scores[0].flatten().float().cpu()
        st = stability(logits)
        area = (logits > 0).flatten(1).float().mean(1)
        ok = (sc > args.min_score) & (st > args.min_stability) & (area > args.min_area)
        masks.append(logits[ok] > 0)
        score.append(sc[ok])
        stab.append(st[ok])
    masks, score, stab = torch.cat(masks), torch.cat(score), torch.cat(stab)
    keep = nms(masks, score * stab, args.nms, (h // 4, w // 4))
    masks, score, stab = masks[keep], score[keep], stab[keep]
    np.savez_compressed(os.path.join(args.dir, 'all_masks.npz'), masks=masks.numpy(), score=score.numpy(),
                        stability=stab.numpy())
    print(json.dumps({'points': len(pts), 'masks': len(keep)}))


if __name__ == '__main__':
    main()
