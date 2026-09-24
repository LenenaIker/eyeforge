"""Convierte los GLB de Objaverse a USD con Isaac Sim y normaliza escala y orientación.

Uso: <isaac_sim>/python.sh spike/convert_assets.py
Cada modelo se escala para que su dimensión horizontal mayor coincida con la longitud
típica de su clase, se apoya en el suelo (z=0) y se centra en el origen.
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.asset_converter as converter  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

# Medida de referencia (m) por clase y sobre qué dimensión se aplica.
# Coches, camiones y buses se escalan por el ANCHO (muy normalizado: ~2.5 m en pesados, 1.7-1.9 en turismos);
# el largo varía demasiado (un bus articulado mide 18 m). Motos y bicis por el LARGO (el ancho es el manillar).
LENGTH = {"small_vehicle": ("width", 1.65, 1.95), "large_vehicle": ("width", 2.40, 2.55),
          "motorcycle": ("length", 1.8, 2.3), "bicycle": ("length", 1.6, 1.9)}


async def convert(src: str, dst: str) -> bool:
    ctx = converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.embed_textures = True
    task = converter.get_instance().create_converter_task(src, dst, None, ctx)
    ok = await task.wait_until_finished()
    if not ok:
        print("ERROR", src, task.get_error_message())
    return ok


def world_points(stage, max_pts: int = 60000) -> np.ndarray:
    """Vértices de todas las mallas en coordenadas de mundo (submuestreados)."""
    chunks = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        p = np.hstack([np.array(pts, dtype=np.float64), np.ones((len(pts), 1))]) @ m  # convención fila de USD
        chunks.append(p[:, :3])
    if not chunks:
        return np.zeros((0, 3))
    # Fuera planos grandes sin grosor (suelos o recogedores de sombra invisibles): inflan la medida
    # y el vehículo acaba encogido. Una matrícula o un cristal son planos pero pequeños y se quedan.
    total = np.ptp(np.vstack(chunks), axis=0).max()
    solid = [c for c in chunks if not (len(c) <= 64 and np.sort(np.ptp(c, axis=0))[0] < 0.01 * np.ptp(c, axis=0).max()
                                        and np.ptp(c, axis=0).max() > 0.5 * total)]
    allp = np.vstack(solid or chunks)
    if len(allp) > max_pts:
        allp = allp[np.random.default_rng(0).choice(len(allp), max_pts, replace=False)]
    return allp


def min_area_rect(xy: np.ndarray):
    """Rectángulo de área mínima (calibres rotatorios sobre la envolvente convexa).

    Devuelve (ángulo del lado largo en radianes, largo, ancho, centro).
    """
    pts = np.unique(xy.round(6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = np.arctan2(e[1], e[0])
        c, s_ = np.cos(a), np.sin(a)
        r = hull @ np.array([[c, -s_], [s_, c]])  # coordenadas en el marco del lado
        lo, hi = r.min(0), r.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = ((lo + hi) / 2) @ np.array([[c, s_], [-s_, c]])
            best = (area, a, hi - lo, ctr)
    _, a, (d0, d1), ctr = best
    if d1 > d0:  # que el ángulo sea el del lado largo
        a, d0, d1 = a + np.pi / 2, d1, d0
    return a, d0, d1, ctr


def normalize(raw_usd: Path, out_usd: Path, cls: str) -> dict | None:
    """USD envoltorio que endereza, centra, apoya en el suelo y escala el modelo.

    La longitud se mide sobre el rectángulo mínimo en planta: con el bbox alineado a ejes, un
    modelo girado dentro del archivo (p. ej. 45 grados) parece más largo y quedaba hasta un 30% pequeño.
    """
    src = Usd.Stage.Open(str(raw_usd))
    up_i = 1 if UsdGeom.GetStageUpAxis(src) == UsdGeom.Tokens.y else 2
    # El conversor mete su propia rotación R en el prim raíz. El marco "de pie" validado en la revisión
    # visual es p_local * to_z * R (to_z = +90 en X si el stage es Y-up); ojo, to_z y R no conmutan.
    # En el envoltorio se anulan los xformOps de la raíz y se aplica una sola matriz a p_local.
    R = UsdGeom.Xformable(src.GetDefaultPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    to_z = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)) if up_i == 1 else Gf.Matrix4d(1)
    U = to_z * R
    pw = world_points(src)
    if len(pw) < 4:
        return None
    homog = lambda a: np.hstack([a, np.ones((len(a), 1))])
    p_local = (homog(pw) @ np.array(R.GetInverse()))[:, :3]
    pz = (homog(p_local) @ np.array(U))[:, :3]
    ang, length, width, ctr = min_area_rect(pz[:, :2])
    if length <= 0:
        return None
    dim, lo, hi = LENGTH[cls]
    s = random.uniform(lo, hi) / (width if dim == "width" else length)
    zmin = pz[:, 2].min()
    m_up = (Gf.Matrix4d().SetTranslate(Gf.Vec3d(-ctr[0], -ctr[1], -zmin))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -np.degrees(ang)))
            * Gf.Matrix4d().SetScale(s))
    m_local = U * m_up
    fin = (homog(p_local) @ np.array(m_local))[:, :3]
    size = fin.max(0) - fin.min(0)

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    geo = UsdGeom.Xform.Define(stage, "/Asset/Geo")
    geo.GetPrim().GetReferences().AddReference(str(raw_usd.resolve()))
    xf = UsdGeom.Xformable(geo)
    xf.ClearXformOpOrder()  # ignora los xformOps que trae el prim referenciado
    xf.AddTransformOp().Set(m_local)
    stage.GetRootLayer().Save()
    # size_m en el marco final: [largo (X), ancho (Y), alto (Z)]
    return {"scale": s, "size_m": [float(v) for v in size], "yaw_fix_deg": float(np.degrees(ang))}


async def run(args) -> None:
    manifest = json.loads(args.manifest.read_text())
    # Correcciones de clase tras la revisión humana (LVIS mete SUVs y pickups en "truck")
    ov_path = args.out / "class_override.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    for m in manifest:
        if m["uid"] in overrides:
            m["class"] = overrides[m["uid"]]["class"]
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for i, m in enumerate(manifest):
        raw = raw_dir / f"{m['uid']}.usd"
        if not raw.exists() and not await convert(m["path"], str(raw)):
            continue
        out = args.out / f"{m['class']}__{m['uid']}.usd"
        info = normalize(raw, out, m["class"])
        if info is None:
            print("SKIP bbox vacío", m["uid"])
            continue
        done.append({**m, "usd": str(out), **info})
        print(f"[{i + 1}/{len(manifest)}] {m['class']:14s} {info['size_m']}")
    (args.out / "assets.json").write_text(json.dumps(done, indent=2))
    print(f"{len(done)} assets listos en {args.out}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=root / "data" / "objaverse" / "manifest.json")
    p.add_argument("--out", type=Path, default=root / "data" / "usd")
    args, _ = p.parse_known_args()
    random.seed(0)
    asyncio.get_event_loop().run_until_complete(run(args))
    app.close()


if __name__ == "__main__":
    main()
