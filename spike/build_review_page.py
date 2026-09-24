"""Genera la página de revisión humana de assets (4 vistas por modelo + veredicto de Claude).

Las decisiones del usuario se guardan en el db del artifact (colección `reviews`, id = uid de Objaverse)
y se vuelcan a data/usd/exclude.txt con spike/apply_review.py.

Uso: python spike/build_review_page.py  ->  data/review/criba.html
"""
import base64
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASS_ES = {"small_vehicle": "Vehículo pequeño", "large_vehicle": "Vehículo grande",
            "motorcycle": "Moto", "bicycle": "Bici"}


def thumb(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    # Recorte central al 80%: el modelo está centrado y así se gana tamaño sin reescalar
    im = im.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=72, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    review = DATA / "sdg" / "review"
    labels = json.loads((review / "labels.json").read_text())
    assets = {a["uid"]: a for a in json.loads((DATA / "usd" / "assets.json").read_text())}
    excluded = set((DATA / "usd" / "exclude.txt").read_text().split())
    views: dict[str, list] = {}
    for lab in labels:
        views.setdefault(lab["uid"], [None] * 4)[lab["view"]] = review / f"rgb_{lab['frame']:04d}.png"
    items = []
    for uid, paths in views.items():
        a = assets[uid]
        dims = sorted(a["size_m"], reverse=True)
        items.append({
            "uid": uid, "cls": a["class"], "clsEs": CLASS_ES[a["class"]], "lvis": a["lvis"],
            "name": (a.get("name") or "").strip() or "(sin nombre)", "author": a.get("author") or "",
            "license": (a.get("license") or "").upper(), "source": a.get("source") or "",
            "dims": [round(d, 1) for d in dims], "claude": "discard" if uid in excluded else "keep",
            "img": [thumb(pth) for pth in paths],
        })
    template = (Path(__file__).parent / "review_template.html").read_text()
    html = template.replace("__ASSETS_JSON__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    out = DATA / "review"
    out.mkdir(exist_ok=True)
    (out / "criba.html").write_text(html)
    print(f"{len(items)} assets, {len(html) / 1e6:.1f} MB -> {out / 'criba.html'}")


if __name__ == "__main__":
    main()
