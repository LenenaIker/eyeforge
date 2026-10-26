"""Clips for the gsplat and Fixer cards: video frames tiling into the 3D street, then blotches in red and a wipe to Fixer."""
import subprocess, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image

FF = "/home/lenena-iker/miniforge3/envs/triaje-vr/bin/ffmpeg"
CASE = Path("/home/lenena-iker/work/eyeforge/data/worldstereo_test/photo21_np")
S = Path(sys.argv[1]); SPOT = S / "spot"; OUT = S / "shots"
W, H, FPS = 1280, 720, 24
BG = np.array([13, 15, 18], np.float32)


def write(name, frames):
    p = subprocess.Popen([FF, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{name}.mp4")], stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes()); n += 1
    p.stdin.close(); p.wait()
    print(name, n / FPS, "s")


ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))
load = lambda p: np.asarray(Image.open(p).convert("RGB").resize((W, H)), np.float32)

# 1. video frames -> 3D: the generated frames tile the screen leg by leg, then the splat fades in over them
z = np.load(CASE / "splat/views.npz")
imgs, names = z["img"], z["name"]
del z
keep = [i for i, n in enumerate(names) if not str(n).startswith("pano")]
print("video frames", len(keep))
cols, rows = 10, 10
pick = np.linspace(0, len(keep) - 1, cols * rows).round().astype(int)
tw, th = W // cols, H // rows
thumbs = [cv2.resize(np.asarray(imgs[keep[k]]), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) for k in pick]
splat = load(SPOT / "global_fwd_G2.jpg")


def vid2splat():
    fill, fade, hold = 2.2, 1.0, 1.0
    for f in range(int((fill + fade + hold) * FPS)):
        t = f / FPS
        im = np.tile(BG, (H, W, 1))
        for k, th_ in enumerate(thumbs):
            a = ease((t - k / len(thumbs) * fill) / 0.25)
            if a <= 0:
                continue
            r, c = divmod(k, cols)
            y, x = r * th, c * tw
            cell = im[y:y + th, x:x + tw]
            cell[:] = cell * (1 - a) + th_[:cell.shape[0], :cell.shape[1]] * a
        a = ease((t - fill) / fade)
        yield im * (1 - a) + splat * a


if "--only-fixer" not in sys.argv:
    write("vid2splat", vid2splat())

# 2. Fixer: blotches in red on the example view, then a wipe to the repaired render
a, b = load(SPOT / "spot_h180_G2.jpg"), load(SPOT / "spot_h180_F3c.jpg")


def detail(x):
    g = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2)), (0, 0), 12)


d = detail(a) - detail(b)
m = (d > np.percentile(d, 85)).astype(np.uint8)
el = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, el(25)), cv2.MORPH_CLOSE, el(61))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
m = np.isin(lab, [k for k in range(1, n) if st[k, 4] > W * H * 0.01]).astype(np.uint8)
soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6)[..., None]
cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
line = np.zeros((H, W), np.uint8)
cv2.drawContours(line, cs, -1, 1, 4, lineType=cv2.LINE_AA)
line = cv2.GaussianBlur(line.astype(np.float32), (0, 0), 1)[..., None]
RED = np.array([255, 40, 40], np.float32)
red = a * (1 - 0.45 * soft) + RED * 0.45 * soft
red = red * (1 - line) + RED * line
print("marked", round(float(m.mean()), 3))


def blotches():
    for f in range(int(4.2 * FPS)):
        k = ease((f / FPS - 2.0) / 0.8)
        yield a * (1 - k) + red * k


def wipe():
    xs = np.arange(W)[None, :, None]
    LIME = np.array([224, 255, 79], np.float32)
    for f in range(int(3.0 * FPS)):
        x = ease((f / FPS - 0.3) / 1.4) * (W + 8) - 4
        im = np.where(xs < x, b, red)
        im[:, max(0, int(x) - 3):max(0, int(x) + 3)] = LIME
        yield im


write("fixer_red", blotches())
write("fixer_wipe", wipe())
Image.fromarray(splat.astype(np.uint8)).save(OUT / "splat_global.jpg", quality=88)
