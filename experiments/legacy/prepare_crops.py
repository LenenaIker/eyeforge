"""Prepara recortes 96x96 para clasificar la zona de interés (ROI) de una cámara fija.

- Sintético: recorte alrededor del bbox del objetivo con margen aleatorio; frames vacíos -> recortes al azar.
- Real (MIO-TCD, solo evaluación y modo híbrido): se agrupan sus clases en las nuestras.

Uso: python experiments/legacy/prepare_crops.py
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SIZE = 96
MIO_MAP = {
    "small_vehicle": ["car", "pickup_truck", "work_van"],
    "large_vehicle": ["articulated_truck", "bus", "single_unit_truck"],
    "motorcycle": ["motorcycle"],
    "bicycle": ["bicycle"],
    "empty": ["background"],
}
MIN_SIDE = 20  # px; por debajo el objeto no aporta nada útil


def crop_resize(img: Image.Image, box) -> Image.Image:
    return img.crop(tuple(int(round(v)) for v in box)).resize((SIZE, SIZE), Image.BILINEAR)


def synth(run_dir: Path, out: Path) -> dict:
    labels = json.loads((run_dir / "labels.json").read_text())
    counts: dict[str, int] = {}
    for lab in labels:
        f, cls = lab["frame"], lab["class"]
        img = Image.open(run_dir / f"rgb_{f:04d}.png").convert("RGB")
        W, H = img.size
        if cls == "empty":
            # Recortes de tamaño y proporción parecidos a los de vehículos, en la mitad inferior (calzada)
            for _ in range(2):
                w = random.uniform(40, 260)
                h = w / random.uniform(0.7, 1.8)
                x = random.uniform(0, W - w)
                y = random.uniform(H * 0.35, max(H * 0.35, H - h))
                box = (x, y, x + w, min(H, y + h))
                d = out / "empty"
                d.mkdir(parents=True, exist_ok=True)
                crop_resize(img, box).save(d / f"{run_dir.name}_{f:04d}_{_}.png")
                counts["empty"] = counts.get("empty", 0) + 1
            continue
        bb = np.load(run_dir / f"bounding_box_2d_tight_{f:04d}.npy")
        names = json.loads((run_dir / f"bounding_box_2d_tight_labels_{f:04d}.json").read_text())
        ids = [int(k) for k, v in names.items() if v.get("class") == cls]
        rows = [r for r in bb if int(r["semanticId"]) in ids]
        if not rows:
            continue  # objetivo fuera de cuadro
        r = rows[0]
        x0, y0, x1, y1 = float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) < MIN_SIDE:
            continue
        # Objetivo muy cortado por el borde: se descarta
        if x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2:
            if random.random() < 0.7:
                continue
        m = [random.uniform(0.03, 0.25) for _ in range(4)]
        box = (max(0, x0 - m[0] * bw), max(0, y0 - m[1] * bh), min(W, x1 + m[2] * bw), min(H, y1 + m[3] * bh))
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        crop_resize(img, box).save(d / f"{run_dir.name}_{f:04d}.png")
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def real(mio_dir: Path, out: Path, n_test: int, n_few: int) -> dict:
    """Separa por clase: n_test para evaluación y n_few para el modo híbrido (pocas fotos reales)."""
    counts = {}
    for cls, srcs in MIO_MAP.items():
        files = [f for s in srcs for f in (mio_dir / s).glob("*.jpg")]
        files = [f for f in files if min(Image.open(f).size) >= 32]  # MIO-TCD tiene recortes minúsculos
        random.shuffle(files)
        for split, sel in (("test", files[:n_test]), ("few", files[n_test:n_test + n_few])):
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in sel:
                Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(d / f"{f.stem}.png")
        counts[cls] = len(files[:n_test])
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", type=Path, default=[ROOT / "data" / "sdg" / "run_01"])
    p.add_argument("--mio", type=Path, default=ROOT / "data" / "real" / "train")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "crops")
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--n-few", type=int, default=50)
    p.add_argument("--skip-real", action="store_true")
    args = p.parse_args()
    random.seed(0)
    for run in args.runs:
        print("sintético", run.name, synth(run, args.out / "synth"))
    if not args.skip_real:
        print("real", real(args.mio, args.out / "real", args.n_test, args.n_few))


if __name__ == "__main__":
    main()
