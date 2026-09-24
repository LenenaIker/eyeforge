"""Generador de datos sintéticos para la prueba de viabilidad (cámara de poste en un cruce).

Modos:
  preview  renderiza cada asset una vez con cámara fija, para revisión visual
  dataset  genera N frames aleatorios (una clase por frame, o vacío) con bbox 2D

Uso: <isaac_sim>/python.sh spike/generate.py --mode dataset --frames 2000 --out data/sdg/run_01
"""
import argparse
import json
import math
import random
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["preview", "review", "dataset"], default="dataset")
p.add_argument("--include-excluded", action="store_true", help="no aplicar exclude.txt (para revisar todos)")
p.add_argument("--frames", type=int, default=2000)
p.add_argument("--out", type=Path, default=ROOT / "data" / "sdg" / "run_01")
p.add_argument("--assets", type=Path, default=ROOT / "data" / "usd" / "assets.json")
p.add_argument("--exclude", type=Path, default=ROOT / "data" / "usd" / "exclude.txt")
p.add_argument("--width", type=int, default=320)
p.add_argument("--height", type=int, default=240)
p.add_argument("--empty-ratio", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
args, _ = p.parse_known_args()
args.out = args.out.resolve()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Semantics, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets"
SKIES = sorted(str(f) for f in (ROOT / "data" / "hdri").glob("*.hdr"))
TREES = [f"{S3}/Vegetation/Trees/{n}.usd" for n in ["American_Beech", "Black_Oak", "Norway_Spruce", "Red_Maple", "White_Pine"]]
PEOPLE = [f"{S3}/Isaac/5.1/Isaac/People/Characters/{n}/{n}.usd" for n in [
    "male_adult_construction_01_new", "female_adult_police_01_new", "F_Business_02", "M_Medical_01",
]]
# Pinturas reales de vehículo (RGB lineal aproximado); se añade ruido para variar
PAINTS = [(0.85, 0.85, 0.85), (0.03, 0.03, 0.03), (0.55, 0.56, 0.58), (0.25, 0.26, 0.28), (0.55, 0.04, 0.03),
          (0.04, 0.12, 0.45), (0.05, 0.22, 0.10), (0.75, 0.55, 0.05), (0.55, 0.47, 0.35), (0.35, 0.08, 0.08)]
SIN_COLOR = set(json.loads((ROOT / "data" / "review" / "sin_color.json").read_text())) \
    if (ROOT / "data" / "review" / "sin_color.json").exists() else set()
CLASSES = ["empty", "small_vehicle", "large_vehicle", "motorcycle", "bicycle"]
# Distancia cámara-objetivo (m) por clase, para que el tamaño aparente sea realista
DIST = {"small_vehicle": (7, 30), "large_vehicle": (10, 40), "motorcycle": (5, 22), "bicycle": (5, 20)}

stage = None


def add_label(prim, label: str) -> None:
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(label)


def add_ref(path: str, url: str, visible: bool = False):
    # La referencia va en un hijo: algunos assets traen sus propios xformOps (double3) y chocarían con los nuestros
    prim = stage.DefinePrim(path, "Xform")
    stage.DefinePrim(f"{path}/Ref", "Xform").GetReferences().AddReference(url)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp()
    xf.AddRotateZOp()
    xf.AddScaleOp()
    set_visible(prim, visible)
    return prim


def set_visible(prim, visible: bool) -> None:
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def set_pose(prim, pos, yaw_deg: float = 0.0, scale: float = 1.0) -> None:
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(*pos))
    ops[1].Set(float(yaw_deg))
    scale *= BASE_SCALE.get(str(prim.GetPath()), 1.0)
    ops[2].Set(Gf.Vec3f(scale, scale, scale))


def look_at(cam_prim, eye, target) -> None:
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def make_quad(path: str, sx: float, sy: float, z: float):
    """Plano con material de color plano; devuelve el input de color y el prim (con escala Y y giro Z editables)."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-sx, -sy, z), (sx, -sy, z), (sx, sy, z), (-sx, sy, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mat = UsdShade.Material.Define(stage, f"{path}Mat")
    shader = UsdShade.Shader.Define(stage, f"{path}Mat/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    xf = UsdGeom.Xformable(mesh)
    xf.AddRotateZOp()
    xf.AddScaleOp()
    return color, mesh.GetPrim()


def build_scene():
    global stage
    settings = carb.settings.get_settings()
    settings.set("/rtx-transient/resourcemanager/texturestreaming/async", False)
    settings.set("/omni/replicator/asyncRendering", False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    ground_color, _ = make_quad("/World/Ground", 400, 400, 0.0)
    road_color, road = make_quad("/World/Road", 400, 1, 0.02)

    # Material de "pintura" para aleatorizar el color de los objetivos
    paint = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    psh = UsdShade.Shader.Define(stage, "/World/Looks/Paint/Shader")
    psh.CreateIdAttr("UsdPreviewSurface")
    paint_inputs = (psh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f),
                    psh.CreateInput("roughness", Sdf.ValueTypeNames.Float),
                    psh.CreateInput("metallic", Sdf.ValueTypeNames.Float))
    paint.CreateSurfaceOutput().ConnectToSource(psh.ConnectableAPI(), "surface")
    global PAINT
    PAINT = (paint, paint_inputs)

    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    UsdGeom.Xformable(dome).AddRotateZOp()
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    UsdGeom.Xformable(sun).AddRotateXYZOp()

    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000))
    # OV2640 con lente de serie: ~60º de FOV horizontal
    cam.CreateFocalLengthAttr(20.955 / 2 / math.tan(math.radians(30)))

    assets = json.loads(args.assets.read_text())
    excluded = set(args.exclude.read_text().split()) if args.exclude.exists() and not args.include_excluded else set()
    assets = [a for a in assets if a["uid"] not in excluded]
    targets = {}
    for i, a in enumerate(assets):
        prim = add_ref(f"/World/Targets/t{i:03d}", a["usd"])
        add_label(prim, a["class"])
        targets.setdefault(a["class"], []).append((a, prim))
    trees = [add_ref(f"/World/Distractors/tree{i}", u) for i, u in enumerate(TREES)]
    people = [add_ref(f"/World/Distractors/person{i}", u) for i, u in enumerate(PEOPLE)]
    for _ in range(10):
        app.update()
    recenter([prim for cls in targets.values() for _, prim in cls])
    trees = fit_height(trees, (6, 16))
    people = fit_height(people, (1.6, 1.9))
    return (ground_color, road_color, road), dome, sun, cam, targets, trees, people


def recenter(prims) -> None:
    """Centra cada target en el origen y lo apoya en z=0 midiendo el bbox ya compuesto.

    El offset precalculado en convert_assets.py no siempre sobrevive a la composición
    (modelos con transformaciones raras en la raíz), así que se corrige aquí sobre el resultado real.
    """
    bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for prim in prims:
        ref = stage.GetPrimAtPath(f"{prim.GetPath()}/Ref")
        r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        c = (r.GetMin() + r.GetMax()) / 2
        UsdGeom.Xformable(ref).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -r.GetMin()[2]))


BASE_SCALE: dict[str, float] = {}


def fit_height(prims, h_range) -> list:
    """Guarda una escala base por distractor para que mida h_range metros (los assets vienen en cm o m).

    Los assets remotos pueden no haber terminado de cargar: se espera un poco y, si siguen
    sin geometría, se descartan (con escala 1 un árbol en cm mediría kilómetros).
    """
    ok = []
    for prim in prims:
        ref = stage.GetPrimAtPath(f"{prim.GetPath()}/Ref")
        h = 0.0
        for _ in range(60):
            bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            r = bc.ComputeLocalBound(ref).ComputeAlignedRange()
            h = r.GetMax()[2] - r.GetMin()[2] if not r.IsEmpty() else 0.0
            if h > 0:
                break
            app.update()
        if h <= 0:
            print(f"[fit] descartado {prim.GetPath()}: sin geometría", flush=True)
            continue
        BASE_SCALE[str(prim.GetPath())] = random.uniform(*h_range) / h
        ok.append(prim)
    return ok


# Colores de arcén: hierba, tierra, grava
VERGES = [(0.18, 0.28, 0.10), (0.30, 0.25, 0.16), (0.35, 0.33, 0.30), (0.22, 0.30, 0.14)]


PAINT = None


def repaint(prim, uid: str, p_colored: float = 0.35) -> bool:
    """Pinta el objetivo entero de un color de la paleta (siempre si el modelo no trae color).

    Se liga con strongerThanDescendants para imponerse a los materiales propios del modelo.
    Limitación asumida: ruedas y cristales toman el mismo color; a 96x96 px pesa poco.
    """
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if uid not in SIN_COLOR and random.random() > p_colored:
        api.UnbindDirectBinding()
        return False
    mat, (color, rough, metal) = PAINT
    base = random.choice(PAINTS)
    color.Set(Gf.Vec3f(*(min(1.0, max(0.0, c * random.uniform(0.8, 1.2))) for c in base)))
    rough.Set(random.uniform(0.2, 0.55))
    metal.Set(random.uniform(0.2, 0.8))
    api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    return True


def randomize_env(ground, dome, sun, change_sky: bool) -> tuple[float, float]:
    """Randomiza suelo, carretera, cielo y sol. Devuelve (yaw de la carretera, semiancho)."""
    ground_color, road_color, road = ground
    v = random.choice(VERGES)
    k = random.uniform(0.6, 1.4)
    ground_color.Set(Gf.Vec3f(*(c * k for c in v)))
    g = random.uniform(0.04, 0.18)
    road_color.Set(Gf.Vec3f(g, g, g * random.uniform(0.95, 1.1)))
    road_yaw = random.uniform(0, 180)
    half_w = random.uniform(2.5, 4.5)
    ops = UsdGeom.Xformable(road).GetOrderedXformOps()
    ops[0].Set(road_yaw)
    ops[1].Set(Gf.Vec3f(1, half_w, 1))
    if change_sky:
        dome.CreateTextureFileAttr().Set(random.choice(SKIES))
    dome.CreateIntensityAttr().Set(random.uniform(400, 1500))
    UsdGeom.Xformable(dome).GetOrderedXformOps()[0].Set(random.uniform(0, 360))
    sun.CreateIntensityAttr().Set(random.uniform(0, 3000))
    UsdGeom.Xformable(sun).GetOrderedXformOps()[0].Set(Gf.Vec3f(random.uniform(-70, 70), random.uniform(-70, 70), 0))
    return road_yaw, half_w


def main() -> None:
    random.seed(args.seed)
    ground, dome, sun, cam, targets, trees, people = build_scene()
    for _ in range(30):
        app.update()

    rep.orchestrator.set_capture_on_play(False)
    rp = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    args.out.mkdir(parents=True, exist_ok=True)
    writer.initialize(output_dir=str(args.out), rgb=True, bounding_box_2d_tight=True)
    writer.attach([rp])

    labels = []
    visible = []

    def show(prims):
        for pr in visible:
            set_visible(pr, False)
        visible.clear()
        for pr in prims:
            set_visible(pr, True)
            visible.append(pr)

    if args.mode == "preview":
        ground[0].Set(Gf.Vec3f(0.3, 0.3, 0.3))
        dome.CreateTextureFileAttr().Set(SKIES[0])
        dome.CreateIntensityAttr().Set(1000)
        sun.CreateIntensityAttr().Set(1500)
        items = [x for cls in targets.values() for x in cls]
        for a, prim in items:
            show([prim])
            set_pose(prim, (0, 0, 0), 0)
            d = {"small_vehicle": 9, "large_vehicle": 18, "motorcycle": 4.5, "bicycle": 4}[a["class"]]
            look_at(cam.GetPrim(), (d * 0.8, d * 0.6, d * 0.45), (0, 0, 0.6))
            rep.orchestrator.step(rt_subframes=8)
            labels.append({"frame": len(labels), "class": a["class"], "uid": a["uid"], "name": a["name"]})
    elif args.mode == "review":
        # 4 vistas por asset encuadradas por su tamaño real; la última, picada, delata peanas y bases
        ground[0].Set(Gf.Vec3f(0.32, 0.32, 0.30))
        dome.CreateTextureFileAttr().Set(SKIES[0])
        dome.CreateIntensityAttr().Set(1000)
        sun.CreateIntensityAttr().Set(1500)
        views = [(35, 0.35), (145, 0.35), (270, 0.12), (215, 1.1)]  # (azimut, elevación relativa)
        items = [x for cls in targets.values() for x in cls]
        for a, prim in items:
            show([prim])
            set_pose(prim, (0, 0, 0), 0)
            L = max(a["size_m"]) * 1.9
            for v, (az, el) in enumerate(views):
                t = math.radians(az)
                look_at(cam.GetPrim(), (L * math.cos(t), L * math.sin(t), L * el + 0.5), (0, 0, min(a["size_m"]) * 0.4))
                rep.orchestrator.step(rt_subframes=8)
                labels.append({"frame": len(labels), "uid": a["uid"], "view": v})
    else:
        for f in range(args.frames):
            road_yaw, half_w = randomize_env(ground, dome, sun, change_sky=f % 20 == 0)
            ry = math.radians(road_yaw)
            along, across = (math.cos(ry), math.sin(ry)), (-math.sin(ry), math.cos(ry))

            def on_road(t: float, lat: float):
                """Punto a t metros a lo largo de la carretera y lat metros del eje."""
                return (t * along[0] + lat * across[0], t * along[1] + lat * across[1], 0)

            cls = "empty" if random.random() < args.empty_ratio else random.choice(CLASSES[1:])
            shown = []
            if cls != "empty":
                a, prim = random.choice(targets[cls])
                # Circula por su carril, en uno u otro sentido
                lane = random.choice([-1, 1]) * half_w * random.uniform(0.2, 0.6)
                heading = road_yaw + (0 if lane < 0 else 180) + random.uniform(-10, 10)
                set_pose(prim, on_road(0, lane), heading)
                painted = repaint(prim, a["uid"])
                shown.append(prim)
                dist = random.uniform(*DIST[cls])
            else:
                a, painted = None, False
                dist = random.uniform(5, 35)
            # Cámara en poste junto a la vía: altura 2-6 m, mirando hacia la zona del objetivo
            az = math.radians(random.uniform(0, 360))
            h = random.uniform(2, 6)
            eye = (dist * math.cos(az), dist * math.sin(az), h)
            jitter = dist * 0.12
            look_at(cam.GetPrim(), eye, (random.uniform(-jitter, jitter), random.uniform(-jitter, jitter), random.uniform(0.3, 1.2)))

            def clear_spot(pos, margin: float) -> bool:
                """Lejos de la cámara, del objetivo y fuera del asfalto."""
                lat = abs(pos[0] * across[0] + pos[1] * across[1])
                return (math.dist(pos[:2], eye[:2]) > margin and math.hypot(*pos[:2]) > margin
                        and lat > half_w + 1)

            # Distractores: árboles en los márgenes, personas en el arcén (espectadores)
            for pr in random.sample(trees, random.randint(0, 3)):
                for _ in range(20):
                    pos = on_road(random.uniform(-40, 40), random.choice([-1, 1]) * random.uniform(half_w + 3, 30))
                    if clear_spot(pos, 7):
                        set_pose(pr, pos, random.uniform(0, 360), random.uniform(0.8, 1.2))
                        shown.append(pr)
                        break
            for pr in random.sample(people, random.randint(0, 3)):
                for _ in range(20):
                    pos = on_road(random.uniform(-15, 15), random.choice([-1, 1]) * (half_w + random.uniform(0.5, 5)))
                    if clear_spot(pos, 2):
                        set_pose(pr, pos, random.uniform(0, 360))
                        shown.append(pr)
                        break
            show(shown)
            rep.orchestrator.step(rt_subframes=4)
            labels.append({"frame": f, "class": cls, "uid": a["uid"] if a else None,
                           "cam_dist": round(dist, 2), "cam_h": round(h, 2), "painted": painted})
            if f % 100 == 0:
                print(f"[{f}/{args.frames}] {cls}", flush=True)

    rep.orchestrator.wait_until_complete()
    (args.out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"{len(labels)} frames en {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # app.close() termina el proceso sin propagar la excepción: hay que imprimirla antes
        traceback.print_exc()
    finally:
        app.close()
