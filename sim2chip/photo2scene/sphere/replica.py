"""S.5 of the sphere route, the replica as a check (docs/roadmap_photosphere.md): every target the eraser took out
goes back where it stood as a 3D asset of its class, rendered (steps/render.py) in a view centred on it over the
erased panorama, and the same view of the original panorama is the reference. What it measures, per sphere:

    where    the asset's silhouette against the real mask (IoU): ground plane, scale and placement together
    light    the darkness of the contact shadow against the real one (the brightness of the objects is only
             reported: the asset's paint is not the real car's)
    camera   sharpness and noise of the objects against the real objects', which the renders then take
             (match_objects): the background already comes from the user's camera, only the objects need it

Each target's pose comes from its mask and the ground it met (single view metrology, as geometry.py's contacts):
a box of its class's size stands on the proxy's ground under the mask, and its place, heading and size are searched
so its silhouette covers the mask best, pixels of other objects left out. The ground fixes the scale (a bigger box
farther away would not stand on it). A box cannot tell an object's front from its back, so the heading is known up to 180 degrees.
Numpy and PIL, run in the main Python.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import capture, render_plan

FIT_SIZE = (256, 192)      # the view the silhouettes are compared in
FIT_MARGIN_DEG = 4.0       # room around the mask in that view
VIEW_MARGIN_DEG = 10.0     # and in the replica's view
VIEW_FOV_DEG = (30.0, 90.0)
VIEW_SIZE = (1024, 768)
CONTACT_MIN_DOWN = 0.02    # a contact ray must point at least this far below the horizon (sine)
SIZE_RANGE = (0.8, 1.25)   # each side of the box within this share of its class's typical size: free sizes made
                           # near-square boxes whose long side turned 90 degrees (corner_cars, 2026-10-17)
ORIENT_MIN_CONFIDENCE = 0.5  # Orient Anything's peak under this: its heading is not used
ORIENT_SPAN_DEG = 15       # the box's heading is searched this far around the model's
CUTOUT_SIZE = (512, 512)   # the cutouts the model sees
MIN_CONTACTS = 3           # panorama columns where the object is seen meeting the ground
MIN_IOU = 0.7              # a fit under this is not trusted: the object stays out of the replica (the
                           # whole cars of the 2026-10-16 spheres fit at 0.78-0.94; under it, cars half hidden)
RING_M = 0.6               # contact shadow: ground this far around the footprint,
FAR_RING_M = (1.5, 3.0)    # against the ground this far from it, out of the shadow
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
BLUR_STEPS = np.round(np.arange(0.0, 2.01, 0.25), 2)  # object blurs tried, Gaussian sigma in output pixels


def to_level(d_pano: np.ndarray, R: np.ndarray) -> np.ndarray:
    return d_pano @ np.asarray(R).T


def box_corners(centre_xz, yaw_deg: float, size, height: float) -> np.ndarray:
    """The 8 corners (levelled frame, y down) of a box standing on level ground at y = height: centre of its base
    at centre_xz, its length along yaw_deg (counterclockwise from +x seen from above, render_plan's world yaw)."""
    L, W, H = size
    a = math.radians(yaw_deg)
    u, v = np.array([math.cos(a), 0, math.sin(a)]), np.array([-math.sin(a), 0, math.cos(a)])
    c = np.array([centre_xz[0], height, centre_xz[1]])
    return np.array([c + su * L / 2 * u + sv * W / 2 * v + np.array([0, -top * H, 0])
                     for su in (-1, 1) for sv in (-1, 1) for top in (0, 1)])


def hull(p: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, counterclockwise (monotone chain)."""
    p = sorted(map(tuple, p))
    if len(p) < 3:
        return np.array(p)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lo, hi = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    return np.array(lo[:-1] + hi[:-1])


def silhouette(view: dict, corners: np.ndarray) -> np.ndarray:
    """The box's silhouette in a view of the levelled frame (the hull of its projected corners)."""
    x, y, _ = capture.project(view, corners / np.linalg.norm(corners, axis=1, keepdims=True))
    img = Image.new("L", tuple(view["size"]), 0)
    ImageDraw.Draw(img).polygon([tuple(q) for q in hull(np.stack([x, y], 1))], fill=1)
    return np.asarray(img, bool)


def centred_view(rays_level: np.ndarray, margin_deg: float, size, fov_range=(10.0, 120.0)) -> dict:
    """A view of the levelled frame centred on these rays, wide enough for all of them plus the margin."""
    c = rays_level.mean(0)
    c /= np.linalg.norm(c)
    yaw, pitch = math.degrees(math.atan2(c[0], c[2])), math.degrees(math.asin(np.clip(c[1], -1, 1)))
    cam = rays_level @ capture.rotation(yaw, pitch)
    half_w = np.degrees(np.arctan(np.abs(cam[:, 0] / cam[:, 2]))).max()
    half_h = np.degrees(np.arctan(np.abs(cam[:, 1] / cam[:, 2]))).max()
    aspect = size[1] / size[0]
    need = max(half_w, math.degrees(math.atan(math.tan(math.radians(half_h + margin_deg)) / aspect)) - margin_deg)
    fov = float(np.clip(2 * (need + margin_deg), *fov_range))
    return {"yaw_deg": round(yaw % 360, 3), "pitch_deg": round(pitch, 3), "fov_deg": round(fov, 2),
            "size": list(size)}


def contact_points(mask: np.ndarray, others: np.ndarray, rays_level: np.ndarray, proxy: dict) -> np.ndarray:
    """Where the object met the ground (levelled x, y, z): below the lowest mask pixel of each panorama column, when
    nothing else stands right under it, the proxy's ground (proxy_distance.npz: the erased panorama's, flat where
    paved and near, MoGe's elsewhere), so the object stands on the same ground the render shows. A plane would put
    far objects behind the proxy where MoGe's ground rises or comes closer."""
    H, W = mask.shape
    dist, ground = proxy["distance"], proxy["ground"] & proxy["valid"]
    ph, pw = dist.shape
    cols = np.flatnonzero(mask.any(0))
    out = []
    for c in cols:
        r = H - 1 - int(np.argmax(mask[::-1, c]))
        if others[min(r + 1, H - 1):min(r + 3, H), c].any() or rays_level[r, c, 1] < CONTACT_MIN_DOWN:
            continue
        pr, pc = min((r + 2) * ph // H, ph - 1), c * pw // W
        if not ground[pr, pc]:
            continue
        out.append(rays_level[min(r + 2, H - 1), c] * dist[pr, pc])
    return np.array(out).reshape(-1, 3)


def front_yaw(azimuth_deg: float, view_yaw_deg: float) -> float:
    """An object's front as a world yaw (box_corners' convention) from Orient Anything V2's azimuth on a cutout
    whose view looks along view_yaw_deg (levelled heading). The model's azimuth is the front's heading relative to
    the line of sight, 0 facing the camera; fitted on 14 renders of known heading, a median 5 degrees off (corner_cars
    2026-10-17)."""
    front_heading = azimuth_deg + view_yaw_deg - 180  # degrees to the right of the panorama's centre
    return (90 - front_heading) % 360


def cutout(img: np.ndarray, alpha: np.ndarray) -> Image.Image:
    return Image.fromarray(np.dstack([img, alpha.astype(np.uint8) * 255]).astype(np.uint8), "RGBA")


def cutouts(d: Path, out: Path) -> dict:
    """RGBA cutouts of every whole target of the original panorama (transparent outside its own mask and over other
    objects), for steps/orient.py: <out>/oNN.png, and the heading each was looked at along."""
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    geo = json.loads((d / "geometry.json").read_text())
    R = np.array(geo["levelling"])
    objects, masks = load_masks(d)
    pano = np.asarray(Image.open(d / "pano.jpg").convert("RGB"))
    Hm, Wm = next(iter(masks.values())).shape
    rays = to_level(capture.pixel_rays(Wm, Hm), R)
    looked = {}
    for o in objects:
        if o["role"] != "target" or o.get("truncated"):
            continue
        v = centred_view(rays[masks[o["id"]]], FIT_MARGIN_DEG, CUTOUT_SIZE)
        pr = render_plan.pano_rays(capture.view_rays(v), R)
        rgb = np.clip(capture.sample(pano, pr) + 0.5, 0, 255).astype(np.uint8)
        cutout(rgb, capture.sample(masks[o["id"]].astype(np.uint8), pr) > 0.5).save(out / f"o{o['id']:02d}.png")
        looked[f"o{o['id']:02d}"] = v["yaw_deg"]
    (out / "looked.json").write_text(json.dumps(looked))
    return looked


def headings(orient: dict, looked: dict) -> dict:
    """Per object name: its front's world yaw from the model, and whether the model saw a front (one front, peak
    over ORIENT_MIN_CONFIDENCE) or only an axis (two fronts: a heading known up to 180 degrees)."""
    out = {}
    for name, r in orient.items():
        if name in looked and r["confidence"] >= ORIENT_MIN_CONFIDENCE and r["symmetry"] in (1, 2):
            out[name] = {"yaw_deg": round(front_yaw(r["azimuth_deg"], looked[name]), 1), "front": r["symmetry"] == 1}
    return out


def fit_box(mask: np.ndarray, others: np.ndarray, R: np.ndarray, proxy: dict, size0,
            yaw_prior: float | None = None) -> dict | None:
    """Pose and size of a box standing on the ground where the object met it (contact_points) whose silhouette
    covers the object's mask best (IoU, other objects' pixels left out). Masks are the panorama's (pano frame);
    size0 its class's typical (L, W, H). A car in a shop window meets the ground at the glass, where no box of its
    class's size covers it: its IoU stays low. With yaw_prior (the model's heading) the box turns only
    ORIENT_SPAN_DEG around it, and the yaw returned is the one of the two ends nearer the prior; without it the
    heading is known up to 180 degrees (a box seen at three quarters fits nearly as well side-on)."""
    Hm, Wm = mask.shape
    rays = to_level(capture.pixel_rays(Wm, Hm), R)
    contacts = contact_points(mask, others, rays, proxy)
    if len(contacts) < MIN_CONTACTS:
        return None
    height = float(np.median(contacts[:, 1]))  # the ground under the object, levelled y (down)
    view = centred_view(rays[mask], FIT_MARGIN_DEG, FIT_SIZE)
    view_pano = render_plan.pano_rays(capture.view_rays(view), R)
    target = capture.sample(mask.astype(np.uint8), view_pano) > 0.5
    ignore = capture.sample(others.astype(np.uint8), view_pano) > 0.5
    target &= ~ignore

    def iou(c, yaw, size):
        sil = silhouette(view, box_corners(c, yaw, size, height)) & ~ignore
        return float((sil & target).sum() / max((sil | target).sum(), 1))

    near = np.median(contacts[:, [0, 2]], 0)
    away = near / max(np.linalg.norm(near), 1e-6)
    size0 = np.asarray(size0, float)
    c0 = near + away * size0[1] / 2
    best = (-1.0, c0, 0.0, size0)

    lo, hi = size0 * SIZE_RANGE[0], size0 * SIZE_RANGE[1]

    def consider(c, yaw, size):
        nonlocal best
        if yaw_prior is not None and abs((yaw - yaw_prior + 90) % 180 - 90) > ORIENT_SPAN_DEG:
            return
        size = np.clip(size, lo, hi)
        v = iou(c, yaw, size)
        if v > best[0]:
            best = (v, np.asarray(c, float), float(yaw), np.asarray(size, float))

    yaws = np.arange(0, 180, 5) if yaw_prior is None else yaw_prior % 180 + np.arange(-ORIENT_SPAN_DEG,
                                                                                   ORIENT_SPAN_DEG + 1, 3)
    for yaw in yaws:
        for k in (0.8, 1.0, 1.2):
            consider(c0, yaw, size0 * [k, 1, 1])
    across = np.array([-away[1], away[0]])
    for step, n in ((0.25, 4), (0.08, 3)):
        for _ in range(2):
            _, c, yaw, size = best
            for a in range(-n, n + 1):
                for b in range(-n, n + 1):
                    consider(c + (a * away + b * across) * step, yaw, size)
            _, c, yaw, size = best
            for dy in (-6, -4, -2, 2, 4, 6):
                consider(c, yaw + dy, size)
            for kl in (0.9, 1.0, 1.1):
                for kw in (0.9, 1.0, 1.1):
                    for kh in (0.9, 1.0, 1.1):
                        consider(c, yaw, size * [kl, kw, kh])
    v, c, yaw, size = best
    if size[1] > size[0] and yaw_prior is None:  # the long side is the length
        size, yaw = size[[1, 0, 2]], yaw + 90
    yaw %= 180
    if yaw_prior is not None and abs((yaw - yaw_prior + 180) % 360 - 180) > 90:
        yaw += 180
    return {"iou": round(v, 3), "centre_xz": np.round(c, 3).tolist(), "yaw_deg": round(yaw, 1),
            "size_m": np.round(size, 3).tolist(), "contacts": len(contacts),
            "ground_y": round(height, 3),
            "distance_m": round(float(np.linalg.norm(c)), 2)}


def pick_asset(pool: list[dict], size) -> dict:
    """The asset of the class whose (length, width, height) is closest to the fitted size, in log ratio."""
    s = np.log(np.asarray(size, float))
    return min(pool, key=lambda a: float(np.abs(np.log(np.asarray(a["size_m"], float)) - s).sum()))


def load_masks(d: Path) -> tuple[list[dict], dict]:
    inst = json.loads((d / "instances.json").read_text())
    z = np.load(d / "instances.npz")
    W, H = (int(v) for v in z["size"])
    masks = {o["id"]: np.unpackbits(z["masks"][i])[:W * H].reshape(H, W).astype(bool)
             for i, o in enumerate(inst["objects"])}
    return inst["objects"], masks


def plan(d: Path, pools: dict[str, list[dict]], heads: dict | None = None, flips: dict | None = None,
         sky: str = "uniform") -> dict:
    """Fit every whole target (not cut by the capture's edge) and write the replica's render folder
    (<d>/replica, as render_plan.write): one view per fitted target, every fitted target in each.
    pools: the assets per target phrase; heads: headings() of the real objects; flips: degrees to add to an
    asset's yaw (by uid) so its front is where the box's is (asset_flips). Writes <d>/replica/fits.json too."""
    heads, flips = heads or {}, flips or {}
    d = Path(d)
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    objects, masks = load_masks(d)
    proxy = dict(np.load(d / "proxy_distance.npz"))
    fits = []
    for o in objects:
        if o["role"] != "target" or o.get("truncated") or not pools.get(o["phrase"]):
            continue
        others = np.any([m for i, m in masks.items() if i != o["id"]], 0) if len(masks) > 1 \
            else np.zeros_like(masks[o["id"]])
        size0 = np.median([a["size_m"] for a in pools[o["phrase"]]], 0)
        head = heads.get(f"o{o['id']:02d}")
        f = fit_box(masks[o["id"]], others, R, proxy, size0, head["yaw_deg"] if head else None)
        if f is None:
            continue
        f.update(id=o["id"], phrase=o["phrase"], kept=f["iou"] >= MIN_IOU, front=bool(head and head["front"]))
        if f["kept"]:
            a = pick_asset(pools[o["phrase"]], f["size_m"])
            f.update(usd=a["usd"], uid=a["uid"])
        fits.append(f)
    kept = [f for f in fits if f["kept"]]
    objs = [{"usd": f["usd"], "length_m": f["size_m"][0], "label": f["phrase"], "uid": f["uid"], "id": f["id"],
             "key": f"o{f['id']}",
             "pos": render_plan.to_world(np.array([f["centre_xz"][0], f["ground_y"], f["centre_xz"][1]]),
                                         h).tolist(),
             "yaw_deg": (f["yaw_deg"] + flips.get(f["uid"], 0.0)) % 360} for f in kept]
    Hm, Wm = next(iter(masks.values())).shape
    rays = to_level(capture.pixel_rays(Wm, Hm), R)
    views = [{"name": f"r{f['id']:02d}", "target": f["id"], "objects": objs,
              **centred_view(rays[masks[f["id"]]], VIEW_MARGIN_DEG, VIEW_SIZE, VIEW_FOV_DEG)} for f in kept]
    out = d / "replica"
    out.mkdir(exist_ok=True)
    (out / "fits.json").write_text(json.dumps(fits, indent=1))
    return render_plan.write(d, views, sun=None, sub="replica", sky=sky) if views else {"views": []}


def rendered_cutouts(d: Path, out: Path) -> dict:
    """Cutouts of each view's own object as Isaac drew it (its visible pixels inside its grown box), for the same
    model: an asset's front is wherever its modeller put it."""
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan_ = json.loads((d / "replica" / "plan.json").read_text())
    fits = {f["id"]: f for f in json.loads((d / "replica" / "fits.json").read_text()) if f["kept"]}
    looked = {}
    for v in plan_["views"]:
        f = fits[v["target"]]
        box = grow(silhouette(v, box_corners(f["centre_xz"], f["yaw_deg"], f["size_m"], f["ground_y"])), 6)
        obj = (np.asarray(Image.open(d / "replica" / f"obj_{v['name']}.png")) > 127) & box
        img = np.asarray(Image.open(d / "replica" / f"render_{v['name']}.png").convert("RGB"))
        if obj.sum() > 100:
            cutout(img, obj).save(out / f"o{f['id']:02d}.png")
            looked[f"o{f['id']:02d}"] = v["yaw_deg"]
    (out / "looked.json").write_text(json.dumps(looked))
    return looked


def asset_flips(d: Path, heads_rendered: dict, flips: dict | None = None) -> dict:
    """Degrees to add to each asset's yaw (0 or 180) so its front faces where the box's does: the model on the
    rendered object against the yaw it was given, by majority over the objects drawn with that asset. Only
    objects whose real front is known (fit "front") and whose render shows one count."""
    d = Path(d)
    flips = dict(flips or {})
    votes = {}
    for f in json.loads((d / "replica" / "fits.json").read_text()):
        h = heads_rendered.get(f"o{f['id']:02d}")
        if not f.get("kept") or not f.get("front") or not h or not h["front"]:
            continue
        given = (f["yaw_deg"] + flips.get(f["uid"], 0.0)) % 360
        votes.setdefault(f["uid"], []).append(abs((h["yaw_deg"] - given + 180) % 360 - 180) > 90)
    for uid, v in votes.items():
        if sum(v) * 2 > len(v):
            flips[uid] = (flips.get(uid, 0.0) + 180) % 360
    return flips


# Measures on a rendered replica.

def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an (H, W[, C]) float image, edges repeated."""
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (r, r)) + ((0, 0),) * (x.ndim - 2), mode="edge")
    rows = sum(k[i] * pad[i:i + x.shape[0]] for i in range(2 * r + 1))
    return sum(k[i] * rows[:, i:i + x.shape[1]] for i in range(2 * r + 1))


def grow(m: np.ndarray, r: int) -> np.ndarray:
    return gaussian(m.astype(np.float32), r / 2) > 0.05 if r > 0 else m


def shrink(m: np.ndarray, r: int) -> np.ndarray:
    return ~grow(~m, r)


def luminance(img: np.ndarray) -> np.ndarray:
    return render_plan.srgb_to_linear(img) @ LUM


def detail(lum: np.ndarray, where: np.ndarray) -> float:
    """Fine detail over coarse detail on these pixels: mean |I - blur(I, 1)| / mean |I - blur(I, 3)|. A softer image
    has less of its detail at the finest scale; the ratio does not depend on the contrast."""
    fine, coarse = np.abs(lum - gaussian(lum, 1.0))[where], np.abs(lum - gaussian(lum, 3.0))[where]
    return float(fine.mean() / max(coarse.mean(), 1e-6)) if where.sum() > 50 else float("nan")


def noise_sigma(gray: np.ndarray, where: np.ndarray, block: int = 8, quantile: float = 0.1) -> float:
    """Camera noise, 0-255 (Immerkaer 1996: the 3x3 mask that cancels smooth images, read on the flattest blocks:
    texture adds to the estimate, so its low quantile is the noise)."""
    g = gray.astype(np.float32)
    k = g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:] \
        + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:]
    e = np.abs(k) * math.sqrt(math.pi / 2) / 6
    w = where[1:-1, 1:-1]
    hh, ww = e.shape[0] // block * block, e.shape[1] // block * block
    eb = e[:hh, :ww].reshape(hh // block, block, ww // block, block).mean((1, 3))
    full = w[:hh, :ww].reshape(hh // block, block, ww // block, block).all((1, 3))
    return float(np.quantile(eb[full], quantile)) if full.sum() >= 5 else float("nan")


def match_objects(img: np.ndarray, obj: np.ndarray, blur: float, noise: float, rng) -> np.ndarray:
    """The objects of a composite given the camera's sharpness and noise (the background already has them): the
    objects and a thin band around them blurred, then noise added on the objects."""
    if blur <= 0 and noise <= 0:
        return img
    x = img.astype(np.float32)
    w = gaussian(grow(obj, 2).astype(np.float32), 1.0)[..., None]
    x = x * (1 - w) + gaussian(x, blur) * w
    x = x + rng.normal(0, 1, x.shape).astype(np.float32) * noise * (obj[..., None] > 0)
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def ground_points(view: dict, R: np.ndarray, height: float) -> np.ndarray:
    """Each view pixel's ray down to the ground plane (levelled x, z), NaN above the horizon."""
    r = capture.view_rays(view)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(r[..., 1] > CONTACT_MIN_DOWN, height / r[..., 1], np.nan)
    return np.stack([r[..., 0] * t, r[..., 2] * t], -1)


def in_box_xz(p: np.ndarray, fit: dict, grow_m: float = 0.0) -> np.ndarray:
    a = math.radians(fit["yaw_deg"])
    q = p - np.asarray(fit["centre_xz"])
    u = q[..., 0] * math.cos(a) + q[..., 1] * math.sin(a)
    v = -q[..., 0] * math.sin(a) + q[..., 1] * math.cos(a)
    L, W = fit["size_m"][:2]
    return (np.abs(u) <= L / 2 + grow_m) & (np.abs(v) <= W / 2 + grow_m)


def measure(d: Path, rng=None) -> dict:
    """Compare each replica view (replica/render_NAME.png) with the original panorama along the same rays, fit the
    objects' blur and noise to the real objects', and write replica/matched_NAME.png, replica/real_NAME.png and
    replica/measure.json."""
    d = Path(d)
    rng = rng or np.random.default_rng(0)
    out = d / "replica"
    plan_ = json.loads((out / "plan.json").read_text())
    fits = {f["id"]: f for f in json.loads((out / "fits.json").read_text()) if f["kept"]}
    geo = json.loads((d / "geometry.json").read_text())
    R, h = np.array(geo["levelling"]), geo["camera_height_m"]
    pano = np.asarray(Image.open(d / "pano.jpg").convert("RGB"))
    ground = np.load(d / "stuff.npz")["ground"].astype(np.float32)
    objects, masks = load_masks(d)
    replicated = np.any([masks[i] for i in fits], 0)
    rest = np.any([m for i, m in masks.items() if i not in fits], 0) if len(masks) > len(fits) \
        else np.zeros_like(replicated)
    views = []
    for v in plan_["views"]:
        name, f = v["name"], fits[v["target"]]
        rays = render_plan.pano_rays(capture.view_rays(v), R)
        real = np.clip(capture.sample(pano, rays) + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(real).save(out / f"real_{name}.png")
        rend = np.asarray(Image.open(out / f"render_{name}.png").convert("RGB"))
        obj = np.asarray(Image.open(out / f"obj_{name}.png")) > 127
        tgt = capture.sample(replicated.astype(np.uint8), rays) > 0.5
        own = capture.sample(masks[f["id"]].astype(np.uint8), rays) > 0.5
        other = capture.sample(rest.astype(np.uint8), rays) > 0.5
        ok = ~other
        iou = float((obj & tgt & ok).sum() / max(((obj | tgt) & ok).sum(), 1))
        lr, lo = luminance(real), luminance(rend)
        inside = shrink(own, 3) & shrink(obj, 3)
        bright = float(np.median(lo[inside]) / max(np.median(lr[inside]), 1e-4)) if inside.sum() > 50 else None
        # Contact shadow: the ground next to the footprint over the ground a little farther, outside every object,
        # on the real cut and on the replica alike (the erased panorama's fill is no reference: the eraser decides
        # its brightness).
        gp = ground_points(v, R, f["ground_y"])
        free = (capture.sample(ground, rays) > 0.5) & ~grow(tgt | obj | other, 2)
        with np.errstate(invalid="ignore"):
            near = in_box_xz(gp, f, RING_M) & ~in_box_xz(gp, f) & free
            far = in_box_xz(gp, f, FAR_RING_M[1]) & ~in_box_xz(gp, f, FAR_RING_M[0]) & free
        shadow = None
        if near.sum() > 50 and far.sum() > 50:
            rel = lambda lum: float(np.median(lum[near]) / max(np.median(lum[far]), 1e-4))  # noqa: E731
            shadow = {"real": round(rel(lr), 3), "replica": round(rel(lo), 3), "pixels": int(near.sum())}
        band_real, band_obj = grow(own, 2) & ~shrink(own, 2) & ok, grow(obj, 2) & ~shrink(obj, 2) & ok
        views.append({"name": name, "id": f["id"], "iou": round(iou, 3),
                      "brightness": None if bright is None else round(bright, 3), "shadow": shadow,
                      "detail_real": detail(lr, band_real), "lum": (lr, lo, band_obj),
                      "noise_real": noise_sigma(real.astype(np.float32) @ LUM, ~(tgt | other | obj)),
                      "noise_obj": noise_sigma(rend.astype(np.float32) @ LUM, shrink(obj, 2)),
                      "obj": obj, "rend": rend})
    # One blur and one noise per sphere (one camera): the blur whose object edges have the real edges' detail, the
    # noise the camera's over what the render already has.
    blur = 0.0
    usable = [v for v in views if not math.isnan(v["detail_real"])]
    if usable:
        err = [sum(abs(detail(gaussian(v["lum"][1], s), v["lum"][2]) - v["detail_real"]) for v in usable)
               for s in BLUR_STEPS]
        blur = float(BLUR_STEPS[int(np.argmin(err))])
    nr = np.nanmedian([v["noise_real"] for v in views]) if views else float("nan")
    no = np.nanmedian([v["noise_obj"] for v in views]) if views else float("nan")
    noise = float(math.sqrt(max(nr ** 2 - no ** 2, 0.0))) if not (math.isnan(nr) or math.isnan(no)) else 0.0
    for v in views:
        matched = match_objects(v["rend"], v["obj"], blur, noise, rng)
        Image.fromarray(matched).save(out / f"matched_{v['name']}.png")
        v["detail_replica"] = detail(v["lum"][1], v["lum"][2])
        v["detail_matched"] = detail(luminance(matched), v["lum"][2])
        for k in ("lum", "obj", "rend"):
            v.pop(k)
        for k in ("detail_real", "detail_replica", "detail_matched", "noise_real", "noise_obj"):
            v[k] = None if math.isnan(v[k]) else round(v[k], 3)
    report = {"views": views, "camera": {"blur_px": blur, "noise": round(noise, 2),
                                         "noise_real": None if math.isnan(nr) else round(float(nr), 2),
                                         "noise_render": None if math.isnan(no) else round(float(no), 2)},
              "summary": summary(views)}
    (out / "measure.json").write_text(json.dumps(report, indent=1))
    return report


def summary(views: list[dict]) -> dict:
    med = lambda xs: round(float(np.median(xs)), 3) if xs else None  # noqa: E731
    sh = [v["shadow"] for v in views if v["shadow"]]
    return {"views": len(views), "iou": med([v["iou"] for v in views]),
            "brightness": med([v["brightness"] for v in views if v["brightness"]]),
            "shadow_real": med([s["real"] for s in sh]), "shadow_replica": med([s["replica"] for s in sh])}
