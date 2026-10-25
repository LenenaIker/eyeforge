"""One source per object (R7): each object is taught by the single generation that sees it best, not by the average.

R5 and R6 masked pixel by pixel and only what another leg contradicted, so a car three legs invented slightly
differently still came out as the blur of three cars. Here:

    segment  SAM 3 finds the objects (PHRASES) in every generated frame of a ws_splat views file.
    select   each instance is lifted to 3D with its leg's MapAnything depth (ws_depth.py) and voxelised; instances
             sharing voxels are one object, across frames and legs. Per object, the leg folders (one WorldStereo
             forward each) are ranked by score: sum over their frames of the object's pixels (closer and more
             frames is more pixels) times the square root of their sharpness against the object's median. Each
             voxel of the object is taught only by the best-ranked folder that sees it: a walk that sees a car's
             side keeps the side, the spin that alone saw its nose keeps the nose, and no surface is the average
             of two versions. That overrides R6's rule on the object; outside any object R6's mask stays.

    <sam3 env>/bin/python experiments/worldstereo/ws_objects.py segment --case photo21
    <mapanything env>/bin/python experiments/worldstereo/ws_objects.py select --case photo21

Writes depth_ma/objects_<views>.npz (instance labels per frame) and depth_ma/select_<views>.npz (name, keep), which
ws_splat reads as its "consensus" mask, plus depth_ma/sheet_select_<views>.jpg."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402

SAM3 = "facebook/sam3"
PHRASES = ("car", "van", "truck", "tree")
FLOOR = 0.5            # SAM 3 instance score
VOXEL = 0.25           # metres
SAME_OBJECT = 0.3      # shared voxels over the smaller instance's voxels to call two instances one object
MAX_EXTENT = 15.0      # metres: an instance wider than this (a whole tree line) is left to R6's rule


def voxel_keys(v: np.ndarray) -> np.ndarray:
    """Integer voxel coordinates (N, 3) as one int64 each (21 bits per axis)."""
    v = v + (1 << 20)
    return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]


def dilate(keys: np.ndarray) -> np.ndarray:
    """The voxels and their 26 neighbours."""
    m = (1 << 21) - 1
    v = np.stack([(keys >> 42) & m, (keys >> 21) & m, keys & m], -1) - (1 << 20)
    off = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return np.unique(voxel_keys((v[:, None] + off[None]).reshape(-1, 3)))


def gen_names(views: np.lib.npyio.NpzFile) -> list[tuple[int, str]]:
    return [(i, str(n)) for i, n in enumerate(views["name"]) if not str(n).startswith("pano/")]


def segment(case: str, views: str):
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    t0 = time.time()
    vz = np.load(ws.DATA / case / "splat" / views)
    model = Sam3Model.from_pretrained(SAM3).cuda().eval()
    proc = Sam3Processor.from_pretrained(SAM3)
    imgs = vz["img"]  # read once: every vz["img"] decompresses the whole array
    names, labels, inst = [], [], []
    for i, n in gen_names(vz):
        im = Image.fromarray(imgs[i])
        pix = proc(images=im, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vision = model.get_vision_features(pixel_values=pix["pixel_values"])
        found = []
        for phrase in PHRASES:
            text = proc(text=phrase, return_tensors="pt").to("cuda")
            with torch.no_grad():
                res = model(vision_embeds=vision, input_ids=text["input_ids"], attention_mask=text["attention_mask"])
            r = proc.post_process_instance_segmentation(res, threshold=FLOOR, mask_threshold=0.5,
                                                        target_sizes=pix.get("original_sizes").tolist())[0]
            found += [(float(s), phrase, m.cpu().numpy().astype(bool)) for s, m in zip(r["scores"], r["masks"])]
        lab = np.zeros(imgs.shape[1:3], np.uint8)
        for k, (s, phrase, m) in enumerate(sorted(found, key=lambda x: x[0])[-255:], 1):
            lab[m] = k  # higher scores painted last win the overlaps
            inst.append((n, k, phrase, s))
        names.append(n)
        labels.append(lab)
        print(f"{n}: {len(found)} instances", flush=True)
    out = ws.DATA / case / "depth_ma" / f"objects_{Path(views).stem}.npz"
    np.savez_compressed(out, name=np.array(names), labels=np.stack(labels),
                        inst=np.array([json.dumps(x) for x in inst]))
    print(f"{time.time() - t0:.0f} s -> {out}")


def select(case: str, views: str):
    import cv2
    import ws_depth as wd
    t0 = time.time()
    stem = Path(views).stem
    vz = np.load(ws.DATA / case / "splat" / views)
    oz = np.load(wd.out_dir(case) / f"objects_{stem}.npz")
    cz = np.load(wd.out_dir(case) / f"consensus_{stem}_closest.npz")
    base = {str(n): k for n, k in zip(cz["name"], cz["keep"])}
    names = [str(n) for n in oz["name"]]
    labels = oz["labels"]
    info = {(n, k): (phrase, s) for n, k, phrase, s in map(json.loads, oz["inst"])}
    imgs = vz["img"]  # read once, and copy the frames: a view would keep the whole array alive per frame
    img = {n: imgs[i].copy() for i, n in gen_names(vz)}
    del imgs
    data = {l: wd.load(case, l, 1.0) for l in sorted({n.rsplit("/", 1)[0] for n in names})}

    # every instance: its pixels (MapAnything's resolution) with their voxel keys, pixel count and sharpness
    insts = []  # (frame name, label, folder, voxel keys, pixels, sharpness, ys, xs, pixel keys)
    for fi, n in enumerate(names):
        leg, f = n.rsplit("/", 1)
        z, f = data[leg], int(f)
        h, w = z["depth"][f].shape
        lab = cv2.resize(labels[fi], (w, h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(img[n], cv2.COLOR_RGB2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        K, m = z["K"][f], z["c2w"][f]
        for k in np.unique(labels[fi])[1:]:
            yy, xx = np.nonzero((lab == k) & z["ok"][f])
            if len(yy) < 20:
                continue
            d = z["depth"][f][yy, xx]
            cam = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * d, (yy + 0.5 - K[1, 2]) / K[1, 1] * d, d], -1)
            P = cam @ m[:3, :3].T + m[:3, 3]
            lo, hi = np.percentile(P, [5, 95], axis=0)
            if np.linalg.norm(hi - lo) > MAX_EXTENT:
                continue
            keys = voxel_keys(np.floor(P / VOXEL).astype(np.int64))
            full = labels[fi] == k
            insts.append((n, int(k), leg.split("/")[0], np.unique(keys), int(full.sum()), float(lap[full].var()),
                          yy, xx, keys))

    # instances sharing voxels are one object: shared counts from an instance x voxel incidence matrix, then
    # connected components of the pairs above SAME_OBJECT
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    allkeys = np.concatenate([it[3] for it in insts])
    rows = np.repeat(np.arange(len(insts)), [len(it[3]) for it in insts])
    _, cols = np.unique(allkeys, return_inverse=True)
    M = coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(insts), cols.max() + 1)).tocsr()
    S = (M @ M.T).tocoo()
    size = np.array([len(it[3]) for it in insts])
    link = S.data >= SAME_OBJECT * np.minimum(size[S.row], size[S.col])
    G = coo_matrix((np.ones(link.sum()), (S.row[link], S.col[link])), shape=S.shape)
    _, comp = connected_components(G, directed=False)
    objects = {}
    for a, c in enumerate(comp):
        objects.setdefault(c, []).append(a)

    # per object, generations ranked by score; each voxel is taught by the best-ranked generation that sees it
    # (claims are dilated one voxel so two versions of one surface a few cm apart do not both teach)
    teach, report, detail = {}, [], []  # teach: (frame name, label) -> bool per pixel of that instance
    first = {}  # (frame name, label) -> the instance's generation is the object's best (for pixels without depth)
    for members in objects.values():
        med = np.median([insts[a][5] for a in members]) or 1.0
        score = {}
        for a in members:
            score[insts[a][2]] = score.get(insts[a][2], 0.0) + insts[a][4] * np.sqrt(insts[a][5] / med)
        order = sorted(score, key=score.get, reverse=True)
        claimed, owned = np.zeros(0, np.int64), {}
        for g in order:
            v = np.unique(np.concatenate([insts[a][3] for a in members if insts[a][2] == g]))
            owned[g] = v[~np.isin(v, claimed)]
            claimed = np.union1d(claimed, dilate(v))
        for a in members:
            it = insts[a]
            teach[(it[0], it[1])] = (it[6], it[7], np.isin(it[8], owned[it[2]]))
            first[(it[0], it[1])] = it[2] == order[0]
        phrase = info[(insts[members[0]][0], insts[members[0]][1])][0]
        report.append((phrase, len(members), len(score), order[0], round(score[order[0]] / sum(score.values()), 2)))
        total = sum(len(o) for o in owned.values()) or 1
        detail.append({"phrase": phrase, "order": order, "score": score,
                       "voxel_share": {g: round(len(o) / total, 3) for g, o in owned.items()},
                       "frames": sorted({insts[a][0] for a in members})})

    keep_all, changed = [], []
    for fi, n in enumerate(names):
        leg, f = n.rsplit("/", 1)
        h, w = data[leg]["depth"][int(f)].shape
        keep = base[n].copy()
        before = keep.copy()
        for k in np.unique(labels[fi])[1:]:
            key = (n, int(k))
            if key not in teach:
                continue  # a whole tree line, or too small: R6's rule
            ys, xs, ok = teach[key]
            tmap = np.full((h, w), 2, np.uint8)  # 2: no depth there, the object's best generation decides
            tmap[ys, xs] = ok
            tfull = cv2.resize(tmap, (keep.shape[1], keep.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = labels[fi] == k
            keep[mask] = np.where(tfull[mask] == 2, int(first[key]), tfull[mask])
        keep_all.append(keep)
        changed.append(float((keep != before).mean()))
    (wd.out_dir(case) / f"select_{stem}.json").write_text(json.dumps(detail, indent=1))
    out = wd.out_dir(case) / f"select_{stem}.npz"
    np.savez_compressed(out, name=np.array(names), keep=np.stack(keep_all))

    multi = [r for r in report if r[2] > 1]
    print(f"{len(insts)} instances, {len(objects)} objects, {len(multi)} seen by more than one generation")
    for r in sorted(multi, key=lambda r: -r[1])[:25]:
        print(f"  {r[0]:6s} {r[1]:3d} instances in {r[2]} generations, best {r[3]} ({r[4]:.0%} of the score)")
    for g in sorted({n.split("_")[0] for n in names}):
        v = [c for n, c in zip(names, changed) if n.startswith(g)]
        print(f"{g:12s} share of pixels changed against R6: mean {np.mean(v):.3f}")
    write_sheet(case, stem, names, labels, keep_all, base, img)
    print(f"{time.time() - t0:.0f} s -> {out}")


def write_sheet(case, stem, names, labels, keep_all, base, img):
    """Frames with the most object pixels: green: object pixels that teach, red: no longer teaches."""
    import cv2
    order = np.argsort([-(l > 0).sum() for l in labels])
    pick = [order[k] for k in range(0, min(len(order), 60), 10)]
    rows = []
    for fi in pick:
        n = names[fi]
        im = img[n].astype(np.float32)
        over = im.copy()
        obj = labels[fi] > 0
        over[obj & (keep_all[fi] == 1)] = over[obj & (keep_all[fi] == 1)] * 0.5 + np.array([0, 255, 0]) * 0.5
        over[keep_all[fi] == 0] = over[keep_all[fi] == 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        r6 = im.copy()
        r6[base[n] == 0] = r6[base[n] == 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        row = np.concatenate([im, r6, over], 1).astype(np.uint8)
        cv2.putText(row, f"{n}   frame | R6 mask | R7 mask (green: object teaches, red: dropped)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows.append(row)
    sheet = np.concatenate(rows, 0)
    sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
    out = ws.DATA / case / "depth_ma" / f"sheet_select_{stem}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"sheet -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["segment", "select"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--views", default="views_s3.npz")
    a = ap.parse_args()
    segment(a.case, a.views) if a.cmd == "segment" else select(a.case, a.views)
