"""The sphere route of photo2scene (docs/roadmap_photosphere.md): a place for the synthetic frames from one
photosphere taken where the user's camera will stand. The panorama stays the background in every direction, a proxy
from its 360 depth catches shadows and hides what stands behind it, and the targets are 3D objects composited on top
by differential rendering.

Steps:
    capture.py         S.1 the photosphere read into the working panorama and cut into pinhole views
    steps/depth.py     S.2 MoGe-2 360 distance map (mesh Python)
    steps/stuff.py     S.2 SAM 3 ground kinds and sky voted onto the panorama (sam3 Python)
    geometry.py        S.2 ground plane, vertical, scale and proxy mesh
    steps/instances.py S.3 SAM 3 targets and movable things, one object per thing across the cuts (sam3 Python)
    erase_plan.py      S.3 the eraser's groups and views
    steps/erase.py     S.3 the objects erased from the panorama, once each (erase Python)
    render_plan.py     S.4 the views, their backgrounds, the objects' places and the light
    steps/render.py    S.4 Isaac's two passes and the differential composite (Isaac Python)
    camera_fx.py       S.4 the target hardware's camera on the frame
    replica.py         S.5 the erased targets put back as assets and compared with the real panorama
    steps/orient.py    S.5 which way each object faces, Orient Anything V2 (sam3 Python)
    extrapolate.py     S.6 new views with targets placed where the sphere can hold them, and their labels
    context.py         S.6 where the targets stand in the user's photos, for a sphere holding none of them
    steps/photo_depth.py, steps/photo_stuff.py  S.6 MoGe-2 and SAM 3 on the user's photos, for context.py
    support.py         S.6 what holds each target up in the user's photos (local VLM, checked by the base)
    steps/photo_holders.py  S.6 SAM 3's instances of those holders, measured into rigs by context.py

Frame of the sphere, shared by every step: the capture point at the origin, OpenCV axes for heading 0 (the
panorama's centre column): +x right, +y down, +z forward. Headings (yaw) grow to the right, pitches downwards.
"""
from __future__ import annotations

from pathlib import Path

from .. import Flow
from . import capture, geometry

STEPS = Path(__file__).resolve().parent / "steps"
# Phrases for SAM 3. The kinds of ground a camera of ours looks at, outdoors and in: where S.6 may put targets
# (on the kinds the sphere showed them on); together they are the ground of geometry.py. Sky is left out of the
# proxy (MoGe's own mask already drops most of it).
GROUND_KINDS = {"road": ("road", "street"), "parking": ("parking lot", "parking space"),
                "sidewalk": ("sidewalk", "pavement"), "grass": ("grass", "lawn"), "floor": ("ground", "floor")}
SKY_PHRASES = ("sky",)
# Built surfaces: only these are straightened as walls in the proxy (geometry.py), never trees or hedges.
STRUCTURE_PHRASES = ("building", "wall", "fence")


def stuff_args(d: Path) -> list:
    groups = [a for kind, phrases in GROUND_KINDS.items() for a in ("--group", f"{kind}={','.join(phrases)}")]
    return ["--dir", d, *groups, "--group", "sky=" + ",".join(SKY_PHRASES), "--group",
            "structure=" + ",".join(STRUCTURE_PHRASES), "--union", "ground=" + ",".join(GROUND_KINDS)]


def build_geometry(flow: Flow, photosphere: Path, capture_height_m: float | None = None) -> dict:
    """S.1 and S.2 for one place in flow.d: cuts, 360 depth, ground votes, ground plane, scale and proxy. The scale is
    MoGe-2's own unless capture_height_m differs from it by more than 10 % (geometry.HEIGHT_OVERRIDE)."""
    info = capture.prepare(photosphere, flow.d)
    flow.run("sphere_depth", "mesh", str(STEPS / "depth.py"), ["--dir", flow.d], outputs=["depth.npz"])
    flow.run("sphere_stuff", "sam3", str(STEPS / "stuff.py"), stuff_args(flow.d), outputs=["stuff.npz"])
    return {**info, "geometry": geometry.solve(flow.d, capture_height_m)}


def build_background(flow: Flow, targets: list[str], movables: list[str],
                     capture_height_m: float | None = None) -> dict:
    """S.3 for a place whose S.2 is done: the targets (the spec's classes) and the movable things (the
    observation's phrases) found and erased; the erased panorama is the background, and the proxy is built again
    from its depth."""
    args = ["--dir", flow.d, *[a for t in targets for a in ("--target", t)],
            *[a for m in movables for a in ("--movable", m)]]
    flow.run("sphere_instances", "sam3", str(STEPS / "instances.py"), args, outputs=["instances.json"])
    flow.run("sphere_erase", "erase", str(STEPS / "erase.py"), ["--dir", flow.d],
             outputs=["background.jpg", "erase.json"])
    flow.run("sphere_depth_bg", "mesh", str(STEPS / "depth.py"),
             ["--dir", flow.d, "--image", "background.jpg", "--out", "depth_bg.npz"], outputs=["depth_bg.npz"])
    flow.run("sphere_stuff_bg", "sam3", str(STEPS / "stuff.py"),
             [*stuff_args(flow.d), "--pano", "background.jpg", "--out", "stuff_bg.npz"], outputs=["stuff_bg.npz"])
    return {"instances": flow.json("instances.json"), "erase": flow.json("erase.json"),
            "geometry": geometry.solve(flow.d, capture_height_m, proxy_depth="depth_bg.npz",
                                       proxy_stuff="stuff_bg.npz")}
