"""The Isaac Replicator clip: walkscene's composites (bins over the splat) plus, drawn on top, the student camera
taking its shots (a frustum that flashes) and a ring where each bin spawns. Writes replicator.mp4."""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FF = "/home/lenena-iker/miniforge3/envs/triaje-vr/bin/ffmpeg"
D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
plan = json.loads((D / "plan.json").read_text())
anim = json.loads((D / "anim.json").read_text())
W, H = plan["size"]
FPS = anim["fps"]
f = W / 2 / math.tan(math.radians(plan["fov_deg"]) / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
ground = -1.6
LIME, WHITE = (224, 255, 79), (255, 255, 255)
SS = 2  # drawing supersample, for smooth lines


def project(m, pts):
    p = (np.asarray(m)[:3, :3] @ np.asarray(pts, float).T).T + np.asarray(m)[:3, 3]
    ok = p[:, 2] > 0.1
    uv = (K @ p.T).T
    return uv[:, :2] / uv[:, 2:3], ok


def frustum(pos, yaw, pitch, length=1.6, hfov=60.0, aspect=0.75):
    y, p = math.radians(yaw), math.radians(pitch)
    fwd = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    up = np.cross(right, fwd)
    hw = length * math.tan(math.radians(hfov / 2))
    hh = hw * aspect
    c = np.asarray(pos) + fwd * length
    corners = [c + right * sx * hw + up * sy * hh for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    body = [np.asarray(pos) - fwd * 0.25 + right * sx * 0.14 + up * sy * 0.1 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.asarray(pos), corners, body


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def frames():
    shots = anim["shots"]
    for v in plan["views"]:
        k = int(v["name"])
        t = k / FPS
        im = cv2.imread(str(D / f"render_{v['name']}.png"))[..., ::-1].astype(np.float32)
        big = cv2.resize(im, (W * SS, H * SS), interpolation=cv2.INTER_LINEAR)
        layer = np.zeros_like(big)
        alpha = np.zeros(big.shape[:2], np.float32)
        m = v["w2c"]

        def poly(pts, colour, a, fill=False, width=2):
            uv, ok = project(m, pts)
            if not ok.all():
                return
            q = (uv * SS).round().astype(np.int32)
            tmp = np.zeros(big.shape[:2], np.uint8)
            if fill:
                cv2.fillPoly(tmp, [q], 255, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(tmp, [q], True, 255, width * SS, lineType=cv2.LINE_AA)
            aa = tmp.astype(np.float32) / 255 * a
            layer[:] = layer * (1 - aa[..., None]) + np.array(colour, np.float32) * aa[..., None]
            alpha[:] = np.maximum(alpha, aa)

        # rings where bins spawn, for a third of a second
        for b in anim["bins"]:
            age = t - b["start"]
            if 0 <= age < 0.35:
                r = b["radius"] * (1.0 + 1.2 * age / 0.35)
                ring = [[b["xy"][0] + r * math.cos(a), b["xy"][1] + r * math.sin(a), ground + 0.02]
                        for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
                poly(ring, LIME, 1.0 - age / 0.35, width=3)
        # the student camera: at the last shot taken, its frustum flashing white at the shot
        past = [s for s in shots if s["t"] - 0.15 <= t]
        if past:
            s = past[-1]
            since = t - s["t"]
            apex, corners, body = frustum(s["pos"], s["yaw"], s["pitch"])
            show = ease((since + 0.15) / 0.15)
            flash = max(0.0, 1.0 - since / 0.3) if since >= 0 else 0.0
            for a_, b_ in zip(corners, corners[1:] + corners[:1]):
                poly([apex, a_, b_], WHITE, 0.55 * flash, fill=True)
            poly(body, (30, 32, 36), 0.95 * show, fill=True)
            poly(body, LIME, show, width=2)
            for c in corners:
                poly([apex, c], LIME, show, width=2)
            poly(corners, LIME, show, width=2)
            if flash > 0:
                uv, ok = project(m, [apex])
                if ok.all():
                    c = (uv[0] * SS).round().astype(int)
                    tmp = np.zeros(big.shape[:2], np.float32)
                    cv2.circle(tmp, tuple(c), int(40 * SS * (0.6 + 0.4 * flash)), 1.0, -1, lineType=cv2.LINE_AA)
                    tmp = cv2.GaussianBlur(tmp, (0, 0), 14 * SS) * flash
                    big[:] = big * (1 - tmp[..., None]) + 255 * tmp[..., None]
        big = big * (1 - alpha[..., None]) + layer * alpha[..., None]
        yield cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


p = subprocess.Popen([FF, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                      "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "22", "-preset", "slow",
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
n = 0
for fr in frames():
    p.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
    n += 1
p.stdin.close()
p.wait()
print(OUT, n / FPS, "s")
