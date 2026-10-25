"""Splat of an arm rendered from the training cameras of a few legs: flow and sharpness against the frames.
    <gsplat env>/bin/python experiments/worldstereo/ws_diag.py S3 R1 R2  (R2: from its refined cameras, poses.npz)"""
import sys
import numpy as np, torch, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "experiments/worldstereo")
import ws_nbv as nb, ws_test as ws
dev = torch.device("cuda")
D = ws.DATA / "photo21"
z = np.load(D / "splat/views_s3.npz")
names = [str(n) for n in z["name"]]
K = np.array(ws.intrinsic(), np.float32)
LEGS = ("spg_0_2/traj0", "reconstruct_0/traj0", "spg_1_1/traj0")
FR = (0, 3, 10, 14, 20)
font = ImageFont.load_default(size=16)
arms = sys.argv[1:]
renders, stats = {}, {}
for arm in arms:
    sp = nb.load_splat(D / f"splat/arms/{arm}/splat.ply", dev)
    pf = D / f"splat/arms/{arm}/poses.npz"
    fixed = {}
    if pf.exists():
        q = np.load(pf)
        fixed = {str(n): m for n, m in zip(q["name"], q["w2c"])}
    for leg in LEGS:
        for f in FR:
            key = f"{leg}/{f:02d}"
            i = names.index(key)
            gt, code = z["img"][i], z["code"][i]
            m = fixed.get(key, z["w2c"][i])
            r = nb.render(sp, m[None], K, ws.W, ws.H)[0][0].cpu().numpy()
            g1, g2 = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 4, 21, 3, 5, 1.1, 0)
            mag = np.linalg.norm(flow, axis=-1)[code != 0]
            lap = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)[code != 0].var()
            stats[(arm, key)] = (np.median(mag), np.percentile(mag, 90), lap(gt), lap(r))
            renders[(arm, key)] = r
    del sp; torch.cuda.empty_cache()
print("frame | " + " | ".join(f"{a} flow med/p90, sharp" for a in arms))
for leg in LEGS:
    for f in FR:
        key = f"{leg}/{f:02d}"
        tag = "held" if f % 7 == 3 else "train"
        print(f"{key} {tag} frame sharp {stats[(arms[0], key)][2]:.0f} | " + " | ".join(
            f"{stats[(a, key)][0]:.1f}/{stats[(a, key)][1]:.1f} px, {stats[(a, key)][3]:.0f}" for a in arms))
for a in arms:
    tr = [v for (aa, k), v in stats.items() if aa == a and int(k[-2:]) % 7 != 3]
    print(f"{a} trained frames: flow median {np.median([t[0] for t in tr]):.2f} px, p90 {np.median([t[1] for t in tr]):.1f} px, "
          f"sharpness kept {np.median([t[3] / t[2] for t in tr]):.2f}")
cw, ch = 448, 258
keys = [f"{l}/{f:02d}" for l in LEGS for f in (0, 14, 20)]
sh = Image.new("RGB", (cw * (1 + len(arms)), len(keys) * (ch + 22)))
d = ImageDraw.Draw(sh)
for k, key in enumerate(keys):
    y = k * (ch + 22)
    d.text((6, y + 2), f"{key}: WorldStereo frame | " + " | ".join(arms) + " from the same camera", fill=(255, 255, 0), font=font)
    sh.paste(Image.fromarray(z["img"][names.index(key)]).resize((cw, ch)), (0, y + 22))
    for j, a in enumerate(arms):
        sh.paste(Image.fromarray(renders[(a, key)]).resize((cw, ch)), (cw * (j + 1), y + 22))
out = D / f"splat/arms/sheet_diag_{'_'.join(arms)}.jpg"
sh.save(out, quality=88); print(out)
