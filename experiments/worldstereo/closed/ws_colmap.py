"""A mesh from WorldStereo's frames by multi-view stereo (test M3): COLMAP (official Docker image, CUDA) on every view
of ws_splat.py's views.npz, starting from the cameras the walks asked for. SIFT features (sky masked), exhaustive
matching, points triangulated at the known cameras, then a bundle adjustment that moves each camera to where
WorldStereo really put its frame (2 to 15 px off, ws_splat.POSE_LR), PatchMatch depth from several views at once and
two meshers (Poisson, Delaunay). The bundle adjustment may drift as a whole, so the meshes are brought back onto the
walks' frame by the similarity that maps the adjusted camera centres onto the asked ones. M2 (ws_mesh.py) fused one
monocular depth per frame instead.

    <panogen venv> experiments/worldstereo/ws_colmap.py run --case e2e_photo27     # everything, skips what is done
    <panogen venv> experiments/worldstereo/ws_colmap.py sheet --case e2e_photo27   # G2 splat, M2, M3 side by side
    <edge-sdg> experiments/worldstereo/ws_colmap.py usd --case e2e_photo27

Writes data/worldstereo_test/<case>/mesh_M3/: colmap/ (workspace), poisson.ply, delaunay.ply, info.json, sheet.jpg."""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws_test as ws  # noqa: E402
from ws_mesh import cameras, render  # noqa: E402

IMAGE = "colmap/colmap:latest"   # COLMAP 4.2.1 with CUDA (pulled 2026-10-24)
SKY = 0                          # ws_splat.py's pixel code for sky
POISSON_DEPTH = 11
POISSON_TRIM = 7


def out_dir(case: str) -> Path:
    return ws.DATA / case / "mesh_M3"


def colmap(work: Path, *args: str):
    """One COLMAP command in its Docker image, the workspace mounted at /w, files owned by the user."""
    real = work.resolve()
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{real}:/w", "-w", "/w", IMAGE, "colmap", *args]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (real / "logs").mkdir(exist_ok=True)
    (real / "logs" / f"{args[0]}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        sys.exit(f"colmap {args[0]} failed, see {real / 'logs' / (args[0] + '.log')}")
    print(f"colmap {args[0]}: {time.time() - t:.0f} s", flush=True)


def export(case: str, work: Path) -> list[str]:
    """Every view as a PNG, its sky as a COLMAP mask (0 = no features), names in views.npz order."""
    z = np.load(ws.DATA / case / "splat" / "views.npz")
    (work / "images").mkdir(parents=True, exist_ok=True)
    (work / "masks").mkdir(exist_ok=True)
    names = []
    for k, (img, code, name) in enumerate(zip(z["img"], z["code"], z["name"])):
        fn = f"{k:04d}_{str(name).replace('/', '_')}.png"
        Image.fromarray(img).save(work / "images" / fn)
        Image.fromarray(((code != SKY) * 255).astype(np.uint8)).save(work / "masks" / f"{fn}.png")
        names.append(fn)
    np.save(work / "w2c.npy", z["w2c"])
    return names


def qvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to COLMAP's (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation
    x, y, zz, w = Rotation.from_matrix(r).as_quat()
    return np.array([w, x, y, zz])


def known_model(work: Path, names: list[str]):
    """The asked cameras as a COLMAP text model whose ids match the database's (no points)."""
    w2c = np.load(work / "w2c.npy")
    db = sqlite3.connect(work / "db.db")
    ids = {n: (i, c) for i, n, c in db.execute("SELECT image_id, name, camera_id FROM images")}
    cams = sorted({c for _, c in ids.values()})
    db.close()
    m = work / "known"
    m.mkdir(exist_ok=True)
    (m / "cameras.txt").write_text("".join(f"{c} PINHOLE {ws.W} {ws.H} {ws.F} {ws.F} {ws.W / 2} {ws.H / 2}\n"
                                           for c in cams))
    lines = []
    for k, n in enumerate(names):
        i, c = ids[n]
        q, t = qvec(w2c[k][:3, :3]), w2c[k][:3, 3]
        lines.append(f"{i} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.9f}' for v in t)} {c} {n}\n\n")
    (m / "images.txt").write_text("".join(lines))
    (m / "points3D.txt").write_text("")


def read_centres(model: Path) -> dict[str, np.ndarray]:
    """Camera centres of a COLMAP text model, by image name."""
    from scipy.spatial.transform import Rotation
    out = {}
    rows = [ln for ln in (model / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    for ln in rows[::2]:
        p = ln.split()
        q, t = np.array(p[1:5], float), np.array(p[5:8], float)
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        out[p[9]] = -r.T @ t
    return out


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) with dst ~ s R src + t."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    sgn = np.eye(3)
    sgn[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ sgn @ vt
    s = np.trace(np.diag(d) @ sgn) / a.var(0).sum()
    return s, r, md - s * r @ ms


def run(case: str):
    t0 = time.time()
    out = out_dir(case)
    work = out / "colmap"
    work.mkdir(parents=True, exist_ok=True)
    names = export(case, work) if not (work / "w2c.npy").exists() else \
        sorted(p.name for p in (work / "images").glob("*.png"))
    if not (work / "db.db").exists():
        colmap(work, "feature_extractor", "--database_path", "db.db", "--image_path", "images",
               "--ImageReader.mask_path", "masks", "--ImageReader.camera_model", "PINHOLE",
               "--ImageReader.single_camera", "1",
               "--ImageReader.camera_params", f"{ws.F},{ws.F},{ws.W / 2},{ws.H / 2}")
        colmap(work, "exhaustive_matcher", "--database_path", "db.db")
    steps = [
        ("tri", lambda: (known_model(work, names),
                         colmap(work, "point_triangulator", "--database_path", "db.db", "--image_path", "images",
                                "--input_path", "known", "--output_path", "tri", "--refine_intrinsics", "0"))),
        ("ba", lambda: colmap(work, "bundle_adjuster", "--input_path", "tri", "--output_path", "ba",
                              "--BundleAdjustment.refine_focal_length", "0",
                              "--BundleAdjustment.refine_principal_point", "0",
                              "--BundleAdjustment.refine_extra_params", "0")),
        ("dense", lambda: colmap(work, "image_undistorter", "--image_path", "images", "--input_path", "ba",
                                 "--output_path", "dense", "--output_type", "COLMAP")),
        ("dense/stereo/depth_maps/done", lambda: (
            colmap(work, "patch_match_stereo", "--workspace_path", "dense",
                   "--PatchMatchStereo.max_image_size", str(ws.W), "--PatchMatchStereo.geom_consistency", "1"),
            (work / "dense/stereo/depth_maps/done").touch())),
        ("dense/fused.ply", lambda: colmap(work, "stereo_fusion", "--workspace_path", "dense",
                                           "--output_path", "dense/fused.ply")),
        ("dense/poisson.ply", lambda: colmap(work, "poisson_mesher", "--input_path", "dense/fused.ply",
                                             "--output_path", "dense/poisson.ply",
                                             "--PoissonMeshing.depth", str(POISSON_DEPTH),
                                             "--PoissonMeshing.trim", str(POISSON_TRIM))),
        ("dense/delaunay.ply", lambda: colmap(work, "delaunay_mesher", "--input_path", "dense",
                                              "--output_path", "dense/delaunay.ply")),
    ]
    for done, fn in steps:
        if not (work / done).exists():
            if done in ("tri", "ba"):  # COLMAP wants its output folder made
                (work / done).mkdir()
            fn()
    for m in ("tri", "ba"):
        if not (work / f"{m}_txt").exists():
            (work / f"{m}_txt").mkdir()
            colmap(work, "model_converter", "--input_path", m, "--output_path", f"{m}_txt", "--output_type", "TXT")
    align(case, work, names, time.time() - t0)


def align(case: str, work: Path, names: list[str], seconds: float):
    """Meshes back onto the walks' frame; how far the adjustment moved the cameras."""
    import open3d as o3d
    w2c = np.load(work / "w2c.npy")
    asked = {n: -w2c[k][:3, :3].T @ w2c[k][:3, 3] for k, n in enumerate(names)}
    ba = read_centres(work / "ba_txt")
    common = [n for n in names if n in ba]
    src, dst = np.stack([ba[n] for n in common]), np.stack([asked[n] for n in common])
    s, r, t = umeyama(src, dst)
    moved = np.linalg.norm((s * src @ r.T + t) - dst, axis=1)
    pts = sum(1 for ln in (work / "tri_txt" / "points3D.txt").read_text().splitlines() if ln and ln[0] != "#")
    info = {"views": len(names), "registered": len(common), "sparse_points": pts, "similarity_scale": round(s, 4),
            "camera_shift_m_median": round(float(np.median(moved)), 3),
            "camera_shift_m_p90": round(float(np.percentile(moved, 90)), 3), "seconds": round(seconds)}
    tf = np.eye(4)
    tf[:3, :3], tf[:3, 3] = s * r, t
    pc = o3d.io.read_point_cloud(str(work / "dense" / "fused.ply"))
    tree = o3d.geometry.KDTreeFlann(pc)
    for name in ("poisson", "delaunay"):
        m = o3d.io.read_triangle_mesh(str(work / "dense" / f"{name}.ply"))
        if not m.has_vertex_colors():  # the Delaunay mesher writes none: the nearest fused point's colour
            near = [tree.search_knn_vector_3d(v, 1)[1][0] for v in np.asarray(m.vertices)]
            m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pc.colors)[near])
        m.transform(tf)
        o3d.io.write_triangle_mesh(str(out_dir(case) / f"{name}.ply"), m)
        info[f"{name}_triangles"] = len(m.triangles)
    info["fused_points"] = len(pc.points)
    (out_dir(case) / "info.json").write_text(json.dumps(info, indent=1))
    print(info)


def usd(case: str):
    from ws_ground import ply_to_usd
    for name in ("poisson", "delaunay"):
        ply_to_usd(out_dir(case) / f"{name}.ply", out_dir(case) / f"{name}.usd")
        print(out_dir(case) / f"{name}.usd")


def sheet(case: str):
    """Per view: the G2 splat, M2 (TSDF of the frames) and M3's two meshes."""
    cams = cameras()
    nov = np.load(ws.DATA / case / "splat" / "arms" / "G2" / "novel.npz")
    by_label = {str(lab): im for lab, im in zip(nov["label"], nov["img"])}
    rows = [("G2 splat", [by_label[lab] for lab, _ in cams]),
            ("M2 TSDF", render(ws.DATA / case / "mesh_M2" / "mesh.ply", cams)),
            ("M3 COLMAP Poisson", render(out_dir(case) / "poisson.ply", cams)),
            ("M3 COLMAP Delaunay", render(out_dir(case) / "delaunay.ply", cams))]
    info = json.loads((out_dir(case) / "info.json").read_text())
    cw, ch, pad = 400, 231, 34
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    sh = Image.new("RGB", (cw * len(cams), pad + len(rows) * ch), (0, 0, 0))
    ImageDraw.Draw(sh).text((8, 4), f"{case}: rows G2 splat, M2 TSDF, M3 COLMAP Poisson, M3 COLMAP Delaunay. COLMAP "
                                     f"moved cameras {info['camera_shift_m_median']} m (median), "
                                     f"{info['fused_points'] / 1e6:.1f}M dense points",
                            fill=(255, 255, 255), font=font)
    for r, (name, ims) in enumerate(rows):
        for k, im in enumerate(ims):
            t = Image.fromarray(im).resize((cw, ch))
            ImageDraw.Draw(t).text((6, 4), f"{name}: {cams[k][0]}", fill=(255, 255, 0), font=small)
            sh.paste(t, (k * cw, pad + r * ch))
    sh.save(out_dir(case) / "sheet.jpg", quality=85)
    print(out_dir(case) / "sheet.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "usd", "sheet"])
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    {"run": run, "usd": usd, "sheet": sheet}[a.cmd](a.case)
