"""photo2scene's street and room routes (roadmap C.1): a whole place rebuilt from one photo, so new cameras can
walk it. Three phases, as in the experiment they come from (experiments/scene3d/photo2scene.py):

1. Extraction: the VLM's observation and the route by rules (1.1, 1.2), the movable things erased (1.3a) and
   kept as 3D models (1.4b), depth and camera (1.4), the surroundings segmented and checked (1.3b, 1.3c) and
   meshed by SAM 3D (1.5).
2. Imitation: ground and facade or wall pieces, the surroundings meshes on them, geometric gates against the
   photo (2.1, 2.2), its light and colours by measurement (2.3, 2.4a), and a side-by-side check (2.4).
3. Extrapolation: the route's template extends the place past what the photo shows (a crossroads, a straight
   street, a room) and new views of it are checked (3.2, 3.4).

build() runs the three phases into a scene folder the library keeps; compose() places the erased things again in
every view (3.5) with each job's seed, renders and labels them. Steps and Pythons as in __init__.py; the steps the
experiment ran in its own `trellis2` env run in the mesh env (roadmap C.2).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .. import config
from . import LONG_SIDE, Flow, Stop

TEMPLATES = {"crossroads": "crossroad.py", "straight_street": "street_template.py", "room": "room_template.py"}
EXTEND_VIEWS = 16
MAX_DE = 35  # mean colour difference (delta E) to the photo above which the place stops
# What compose() writes into a scene folder: one job's things, not part of the place the library keeps.
COMPOSED = ("populated",)


def hdri_dir(group: str) -> Path:
    """The skies of the route's group: urban ones for a city street, the general ones otherwise."""
    root = Path(config.env("SIM2CHIP_HDRI_DIR") or Path(__file__).resolve().parents[2] / "data" / "hdri")
    return root / "urban" if group == "urban" else root


def _phrases(entries) -> str:
    return ",".join(e["phrase"] if isinstance(e, dict) else e for e in entries)


def photo_png(d: Path, photo: Path) -> Path:
    """The photo upright and at the size the steps work at, once."""
    if not (d / "photo.png").exists():
        from PIL import Image, ImageOps
        with Image.open(photo) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
            im.save(d / "photo.png")
    return d / "photo.png"


def observe(flow: Flow, photo: Path) -> dict:
    """1.1 and 1.2: what the photo shows (three VLM samples voted) and the route by rules. Returns route.json
    with the observation under "observe"; raises Stop when the photo is out of scope or the answers disagree."""
    d = flow.d
    image = photo_png(d, photo)
    flow.run("1.1_observe", "sam3", "flow_observe.py", ["--image", image, "--out", d], ["observe.json"])
    code = flow.run("1.2_route", "sam3", "flow_route.py", ["--dir", d], ["route.json"], ok=(0, 3))
    r = flow.json("route.json")
    if code == 3 or r["route"] == "out_of_scope":
        raise Stop("the photo is out of scope: " + ("; ".join(r["why"]) or "no route fits it"))
    return {**r, "observe": flow.json("observe.json")["final"]}


def build(flow: Flow, photo: Path, targets: list[str]) -> dict:
    """The place of one street or room photo into flow.d (observe() first). targets: the phrases of the user's
    classes, always erased and kept as targets. Returns its summary; raises Stop at the first failed check."""
    d = flow.d
    r = observe(flow, photo)
    if r["route"] == "support":
        raise Stop("the photo is a working surface: the support route builds it")
    if not r.get("template_built"):
        raise Stop(f"no template builds a {r['template']} yet")
    image = d / "photo.png"
    fin = r["observe"]
    room = r["route"] == "indoor_floor"
    ground = _phrases(fin["ground"])
    t = ",".join(targets)
    # 1.3a Movable things out of the photo. In a room every thing is a whole object measured on the photo's depth,
    # the big ones scene and the small ones and the user's classes erased (scene_split.py); a street erases the
    # observation's movable objects.
    if room:
        flow.run("1.3a_things", "sam3", "flow_things.py", ["--image", image, "--dir", d, "--targets", t],
                 ["things.json", "things.npz"])
        flow.run("1.3a_depth_photo", "mesh", "bg_depth.py", ["--dir", d, "--only", "photo"], ["depth_photo.npz"])
        flow.run("1.3a_split", "sam3", "scene_split.py", ["--dir", d], ["split.json", "objects.npz"])
        flow.run("1.3a_prepare", "sam3", "bg_prepare.py", ["--image", image, "--object-masks", d / "objects.npz",
                                                          "--support", ground, "--out", d],
                 ["background.png", "ground.npz"])
    else:
        seen = [p for p in _phrases(fin["objects"]).split(",") if p]
        objects = ",".join(dict.fromkeys([x.lower() for x in targets] + seen)) or "car"
        flow.run("1.3a_prepare", "sam3", "bg_prepare.py",
                 ["--image", image, "--objects", objects, "--support", ground,
                  "--keep-inside", _phrases(fin["surroundings"]), "--out", d]
                 + (["--shadows"] if fin["weather"] == "sunny" else []), ["background.png", "ground.npz"])
    flow.run("1.3a_erase", "erase", "erase_qwen.py", ["--dir", d, "--replace"], ["background_qwen.png"])
    prep = flow.json("prepare.json")
    if prep["support_frac"] <= 0.03:
        raise Stop(f"the segmenter found almost no ground ({prep['support_frac']:.0%} of the photo)")
    # 1.4 Depth of the photo and of the erased photo; 1.4b the erased things as 3D models; the camera.
    flow.run("1.4_depth", "mesh", "bg_depth.py", ["--dir", d], ["depth_photo.npz", "depth_background.npz"])
    flow.run("1.4b_assets", "sam3", "flow_assets.py", ["--dir", d, "--targets", t], ["assets.json"])
    mesh = ["--dir", d, "--masks", "assets.npz", "--image", "photo", "--out", "assets"]
    flow.run("1.4b_meshes", "mesh", "flow_sam3d.py", mesh, ["assets/scene_graph.json"])
    assets = flow.json("assets.json")["assets"]
    if not all((d / a["mesh"]).exists() for a in assets):
        flow.run("1.4b_meshes_retry", "mesh", "flow_sam3d.py", mesh)
    if flow.run("1.4_camera", "mesh", "flow_camera.py", ["--dir", d], ["camera.json"], ok=(0, 3)) == 3:
        raise Stop("the camera does not fit the route: " + "; ".join(flow.json("camera.json")["why"]))
    # 1.3b, 1.3c The surroundings: SAM 3 candidates on the erased photo, gates in code, the VLM per candidate.
    flow.run("1.3b_segment", "sam3", "flow_segment.py", ["--dir", d] + (["--from-split"] if room else []),
             ["env_candidates.npz"])
    flow.run("1.3c_check_masks", "sam3", "flow_check_masks.py", ["--dir", d], ["env_masks.json"])
    if not any(c["keep"] for c in flow.json("env_masks.json")["candidates"]):
        raise Stop("no house, tree or furniture of the photo passed the mask check")
    # 1.5 Meshes of the surroundings. A mask that fails twice is too big for the GPU (a whole row of houses): it
    # leaves the kept masks and the meshes are made again without it (their indices follow the masks).
    for attempt in range(2):
        flow.run(f"1.5a_sam3d_{attempt}", "mesh", "flow_sam3d.py", ["--dir", d, "--fresh", "--voxels", "sam3d"],
                 ["meshes/scene_graph.json"])
        n = len(np.load(d / "env_masks.npz")["phrases"])
        have = lambda: {int(p.stem) for p in (d / "meshes/sam3d").glob("[0-9][0-9][0-9].glb")}  # noqa: E731
        if len(have()) < n:
            flow.run(f"1.5a_sam3d_retry_{attempt}", "mesh", "flow_sam3d.py", ["--dir", d, "--voxels", "sam3d"])
        missing = sorted(set(range(n)) - have())
        if not missing:
            break
        z = dict(np.load(d / "env_masks.npz"))
        keep = [i for i in range(n) if i not in missing]
        np.savez_compressed(d / "env_masks.npz", **{k: v[keep] for k, v in z.items()})
        flow.log(f"  no mesh for {', '.join(str(z['phrases'][i]) for i in missing)}: left out")
        shutil.rmtree(d / "meshes", ignore_errors=True)
    else:
        raise Stop("SAM 3D made no mesh for some surroundings even after leaving the others out")
    flow.run("1.5b_refine", "mesh", "flow_refine.py", ["--dir", d], ["refine.json"])
    flow.run("1.5c_check_meshes", "mesh", "flow_check_meshes.py", ["--dir", d], ["meshes.json"])
    keep = [x["index"] for x in flow.json("meshes.json")["meshes"] if x["keep"]]
    if not keep:
        raise Stop("no mesh of the surroundings passed the check")
    imitate(flow, r, keep)
    extend(flow, r)
    return {"route": r["route"], "template": r["template"], "hdri_group": r["hdri_group"],
            "objects": [a for a in flow.json("assets.json")["assets"] if (d / a["mesh"]).exists()]}


def imitate(flow: Flow, r: dict, keep: list[int]):
    """Phase 2: the place from the photo's camera, without its movable things, checked against the photo."""
    d = flow.d
    flow.run("2.1a_surfaces", "sam3", "piece_surfaces.py", ["--dir", d], ["surfaces.npz"])
    kind = "--outdoor" if r["route"] == "village_street_day" else "--indoor"
    flow.run("2.1b_pieces", "mesh", "piece_build.py", ["--dir", d, kind, "--views", "0", "--depth", "background"],
             ["pieces.json", "isaac/scene.json"])
    # 2.2 The surroundings meshes into the place, then the geometric gates; a mesh that fails is dropped once.
    for attempt in range(2):
        flow.run(f"2.2a_env_{attempt}", "mesh", "env_add.py",
                 ["--env", d / "meshes", "--scene", d, "--out", d / "place", "--fill-ground", "--photo-texture",
                  "--keep", ",".join(map(str, keep))] + (["--drop-covered-walls"] if kind == "--outdoor" else []),
                 ["place/isaac/scene.json"])
        flow.run(f"2.2b_gates_{attempt}", "mesh", "flow_gates.py", ["--dir", d], ["gates.json"])
        g = flow.json("gates.json")
        drop = [i for i in g["drop"] if i in keep]
        if not drop or attempt == 1:
            break
        keep = [i for i in keep if i not in drop]
        flow.log(f"  the gates dropped meshes {drop}")
        for k in ("place/isaac/scene.json", "gates.json"):
            (d / k).unlink()
    if not g["ok"]:
        raise Stop(f"the rebuilt ground is off the photo (depth error {g['ground_depth_err']}, cover "
                   f"{g.get('ground_cover')})")
    if not keep or drop:
        raise Stop(f"meshes still fail the gates: {drop}")
    # 2.3 Light and sky: the route's HDRI group, then the photo's light by measurement (view 0 under a fixed set of
    # lights, the closest kept).
    hdri = hdri_dir(r["hdri_group"])
    render = lambda step, *extra: flow.run(step, "isaac", "piece_isaac.py",  # noqa: E731
                                           ["--dir", d / "place", "--hdri-dir", hdri, *extra])
    render("2.3_render")
    render("2.3b_sweep", "--sweep")
    flow.run("2.3b_light", "mesh", "flow_light.py", ["--dir", d], ["light.json"])
    render("2.3c_render")
    # 2.4a Photometry against the photo, two rounds; a round is kept only if the colour difference goes down.
    if (d / "photometry.json").exists() and not (d / "place/isaac/orig_scene.json").exists():
        (d / "photometry.json").unlink()
    first = None
    for rnd in range(2):
        flow.run(f"2.4a_photometry_{rnd}", "mesh", "flow_photometry.py", ["--dir", d, "--apply"])
        ph = flow.json("photometry.json")
        first = first or ph
        if ph["ok"] or not ph.get("applied"):
            break
        render(f"2.4a_render_{rnd}")
    flow.run("2.4a_photometry_final", "mesh", "flow_photometry.py", ["--dir", d, "--out", "photometry_final"])
    fin = flow.json("photometry_final.json")
    src = d / "place/isaac"
    if fin["mean_de"] > first["mean_de"] and (src / "orig_scene.json").exists():
        for name in ("scene.json", "meshes.npz"):  # the gains made it worse: back to the original colours
            (src / name).write_bytes((src / f"orig_{name}").read_bytes())
        render("2.4a_render_revert")
        fin = first
    if not fin["checks"]["exposure within 0.8-1.25"]:
        # The views are exposed by this much more or less, view 0 too (it keeps a fixed exposure from its first
        # render, 4.5 times darker than the photo under a light that shaded the facade, terrace_row 2026-10-07).
        sc = json.loads((src / "scene.json").read_text())
        gain0 = sc.get("exposure_0") or json.loads((src / "exposure_0.json").read_text())["gain"]
        for name in ("scene.json", "orig_scene.json"):
            p = src / name
            if p.exists():
                sc = json.loads(p.read_text())
                sc["exposure_scale"] = round(min(1.5, max(0.5, fin["exposure"])), 3)
                sc["exposure_0"] = float(gain0 * min(4.0, max(0.25, fin["exposure"])))
                p.write_text(json.dumps(sc))
        render("2.4a_render_exposure")
        flow.run("2.4a_photometry_exposed", "mesh", "flow_photometry.py", ["--dir", d, "--out", "photometry_final"])
        fin = flow.json("photometry_final.json")
    flow.log(f"  colours: mean delta E {first['mean_de']} -> {fin['mean_de']}, exposure {fin['exposure']}")
    if fin["mean_de"] > MAX_DE:
        raise Stop(f"the place's colours are far from the photo (mean delta E {fin['mean_de']})")
    # 2.4 Side by side with the photo, closed questions in both orders.
    flow.run("2.4_compare", "sam3", "flow_compare.py", ["--dir", d], ["compare.json"])
    c = flow.json("compare.json")
    if not c["ok"]:
        raise Stop("the place does not look like the photo: " + ", ".join(c["failed"]))


def extend(flow: Flow, r: dict):
    """Phase 3: the route's template extends the imitated place; its views are checked by the VLM."""
    d = flow.d
    t = r["template"]
    args = ["--scene", d / "place", "--out", d / "extended", "--views", EXTEND_VIEWS]
    if t == "crossroads":
        args += ["--photo-scene", d, "--no-objects"]
    elif t == "straight_street":
        args += ["--photo-scene", d]
    flow.run("3.2_extend", "mesh", TEMPLATES[t], args, ["extended/isaac/scene.json"])
    flow.run("3.4a_render", "isaac", "piece_isaac.py", ["--dir", d / "extended", "--hdri-dir", hdri_dir(r["hdri_group"])],
             ["extended/sheet_isaac.jpg"])
    flow.run("3.4b_views", "sam3", "flow_check_views.py", ["--dir", d / "extended"] + (["--indoor"] if t == "room" else []),
             ["extended/views.json"])
    v = flow.json("extended/views.json")
    flow.log(f"  {len(v['kept'])} of the template's views pass ({v['share']:.0%})")
    if not v["ok"]:
        raise Stop(f"only {v['share']:.0%} of the extended place's views pass the check")


def compose(flow: Flow, seed: int) -> list[dict]:
    """3.5 The erased things placed again at random in every kept view of a built place (a new draw per view and
    per seed), rendered and labelled. Returns [{"file": populated/isaac/render_NN.png, "boxes": [...]}]."""
    d = flow.d
    for out in COMPOSED:  # every job its own things; removed, never written over (links to the library's files)
        shutil.rmtree(d / out, ignore_errors=True)
    r = flow.json("route.json")
    flow.run("3.5_populate", "mesh", "flow_populate.py", ["--dir", d, "--seed", seed], ["populated/isaac/scene.json"])
    p = flow.json("populated/populate.json")
    if not sum(p["placed_per_view"]):
        flow.log("  no erased thing found a place in any view")
        return []
    flow.run("3.5_render", "isaac", "piece_isaac.py", ["--dir", d / "populated", "--hdri-dir", hdri_dir(r["hdri_group"]),
                                                       "--seed", seed], ["populated/sheet_isaac.jpg"])
    flow.run("3.5_labels", "mesh", "view_labels.py", ["--dir", d / "populated"], ["populated/isaac/labels.json"])
    kept = set(flow.json("extended/views.json")["kept"])
    labels = flow.json("populated/isaac/labels.json")
    return [{"file": f"populated/isaac/{name}", "boxes": boxes} for name, boxes in sorted(labels.items())
            if int(name.split("_")[1].split(".")[0]) in kept and (d / "populated/isaac" / name).exists()]
