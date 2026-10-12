"""Builds the comparison sheet for the user (docs/propuesta_etiquetas_y_avisos.md): today's screens, three
proposals for showing classes to non-experts (static mocks with crops of real photos) and the data warnings
that checks.py fires on every dataset, with what the model sees at each device's input size.

Usage: python experiments/data_warnings/sheet.py --shots <dir with dataset.png, labeling.png> [--out <html>]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

import checks

JOBS = checks.P2S / "jobs"
OUT = checks.P2S / "compare/propuestas_etiquetas_avisos.html"


def b64(im: Image.Image, q=80) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def thumb(path: Path, side=160) -> str:
    im = ImageOps.exif_transpose(Image.open(path))
    return b64(ImageOps.fit(im, (side, side)))


def crops(job: str, cls: str, n=4, side=120) -> list[str]:
    """The teacher's best scored boxes of a class, cut out with a margin: the class's look at a glance."""
    v = json.loads((JOBS / job / "label/verdicts.json").read_text())
    found = sorted(((o["score"], f, o["box"]) for f, x in v.items() for o in x.get("objects") or []
                    if o["class"] == cls and o.get("box")), reverse=True)
    out, seen = [], set()
    for _, f, b in found:
        if f in seen:
            continue
        seen.add(f)
        im = Image.open(JOBS / job / "label/images" / f)
        W, H = im.size
        mx, my = (b[2] - b[0]) * 0.15, (b[3] - b[1]) * 0.15
        box = (max(0, (b[0] - mx) * W), max(0, (b[1] - my) * H), min(W, (b[2] + mx) * W), min(H, (b[3] + my) * H))
        out.append(b64(ImageOps.pad(im.crop(box), (side, side), color=(240, 240, 243))))
        if len(out) == n:
            break
    return out


def class_photos(job: str, cls: str, n=4) -> list[str]:
    files = sorted((JOBS / job / "label" / cls).iterdir())[:n]
    return [thumb(f, 120) for f in files]


def seen_at(job: str, name: str, sizes=(None, 320, 96), width=300) -> list[tuple[str, str]]:
    """One photo with its marks, as the user took it and as the model sees it at each input size (squashed to a
    square, then shown enlarged with its pixels kept)."""
    v = json.loads((JOBS / job / "label/verdicts.json").read_text())[name]
    im = Image.open(JOBS / job / "label/images" / name).convert("RGB")
    out = []
    for s in sizes:
        small = im if s is None else im.resize((s, s), Image.BILINEAR)
        big = small.resize((width, round(width * small.height / small.width)), Image.NEAREST if s else Image.BILINEAR)
        d = ImageDraw.Draw(big)
        for o in v["objects"]:
            x0, y0, x1, y1 = o["box"]
            d.rectangle([x0 * big.width, y0 * big.height, x1 * big.width, y1 * big.height], outline=(255, 196, 0), width=2)
        out.append((b64(big, 85), "your photo" if s is None else f"what the model sees at {s} x {s} px"))
    return out


def shot(path: Path, width=900) -> str:
    im = Image.open(path)
    return b64(im.resize((width, round(width * im.height / im.width))), 78)


CSS = """
:root{--brand:#614efa;--brand50:#efedff;--brand200:#c9c2fe;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;
--ember50:#fffbeb;--ember100:#fef3c7;--ember400:#fbbf24;--ember700:#b45309;--bg:#f3f4f6}
*{box-sizing:border-box}body{font-family:Inter,system-ui,sans-serif;max-width:1350px;margin:auto;padding:16px;
line-height:1.45;color:var(--ink)}h1{margin-bottom:4px}h2{margin-top:40px;border-bottom:2px solid var(--line);
padding-bottom:6px}h3{margin:24px 0 8px}.muted{color:var(--muted)}small{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d1d5db;padding:6px 9px;
text-align:left;vertical-align:top}th{background:#f9fafb}
.note{background:#fff7e6;padding:8px 12px;border-radius:6px}.row{display:flex;gap:16px;flex-wrap:wrap}
.shot{max-width:100%;border:1px solid var(--line);border-radius:8px}
.mock{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:18px;margin:10px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:0 1px 2px #0000000d}
.chip{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:2px 10px;font-size:13px;
background:#fff;border:1px solid var(--line)}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.bar{height:6px;border-radius:9px;background:#e5e7eb;overflow:hidden;display:flex}.bar i{display:block}
.btn{display:inline-block;border-radius:10px;padding:5px 12px;font-size:13px;border:1px solid var(--line);
background:#fff}.btn.p{background:var(--brand);color:#fff;border-color:var(--brand)}
.warn{background:var(--ember50);border:1px solid var(--ember100);color:var(--ember700);border-radius:12px;
padding:10px 14px;font-size:14px;display:flex;gap:10px}.warn b{color:#92400e}
.ex img{width:64px;height:64px;border-radius:8px;object-fit:cover;border:3px solid transparent}
.bubble{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px 14px;max-width:560px;font-size:14px}
.me{margin-left:auto;background:var(--brand50);border-color:var(--brand200)}
.col{flex:1;min-width:220px;background:#fff;border-radius:12px;border-top:5px solid;padding:10px}
.col .g{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.col .g img{width:100%;border-radius:6px}
.tag{font-size:11px;border-radius:6px;padding:1px 6px;background:#fff;border:1px solid var(--line)}
@media (max-width:700px){.row{flex-direction:column}}
"""

COLORS = ["#2f6db5", "#d9480f", "#2b8a3e", "#ae3ec9"]  # one color per class, the same on cards and boxes


def mock_a(bottle, sun, water) -> str:
    def card(name, desc, color, imgs, found, photos, checked, extra=""):
        ex = "".join(f'<img src="{s}" style="border-color:{color}">' for s in imgs)
        pc = checked / photos if photos else 0
        return f"""<div class="card" style="flex:1;min-width:280px;border-top:5px solid {color}">
<div style="display:flex;justify-content:space-between;align-items:baseline"><b style="font-size:17px">{name}</b>
<span class="btn">Edit</span></div>
<div class="muted" style="font-size:13px;margin:2px 0 8px">{desc}</div>
<div class="ex" style="display:flex;gap:6px">{ex}</div>
<div style="font-size:13px;margin:10px 0 4px">{found} &middot; in {photos} photos &middot; <b>{checked}</b> checked by you</div>
<div class="bar"><i style="width:{pc * 100:.0f}%;background:{color}"></i><i style="width:{(1 - pc) * 100:.0f}%;background:{color}33"></i></div>
{extra}</div>"""
    return f"""<div class="mock"><div style="font-size:12px;font-weight:600;color:var(--muted);letter-spacing:.05em">
WHAT TO FIND</div><div class="row" style="margin-top:8px">
{card("Bottle", "A drinking bottle, standing or lying, cap on or off.", COLORS[0], bottle, "103 found", 36, 0)}
{card("Sun cream", "Tubes and bottles of sun cream.", COLORS[1], sun, "sorted", 25, 0)}
{card("Water bottle", "Sports and water bottles.", COLORS[2], water, "sorted", 25, 3,
      '<div class="warn" style="margin-top:10px;font-size:12px;padding:6px 10px">25 photos: enough to start, 30 or more make it steadier.</div>')}
</div><div style="margin-top:12px;font-size:13px" class="muted">Each class: its own color (the same as its boxes),
a few photos the AI picked as clear examples, a sentence in your words and how many you checked. The examples
are the best scored marks of the teacher, cut out of your photos.</div></div>"""


def mock_b(photo_with_boxes) -> str:
    return f"""<div class="mock"><div style="display:flex;gap:16px;flex-wrap:wrap">
<div style="width:240px"><div style="font-size:12px;font-weight:600;color:var(--muted)">THINGS TO FIND</div>
<div class="card" style="margin-top:6px;padding:10px"><span class="dot" style="background:{COLORS[0]}"></span>
<b> Bottle</b><div class="muted" style="font-size:12px">103 found in 36 photos<br>0 checked by you</div></div>
<div style="font-size:12px;margin-top:12px" class="muted"><b>How to read the photos</b><br>
<span style="border:2px dashed {COLORS[0]};padding:0 6px">dashed</span> the AI found it, not checked yet<br>
<span style="border:2px solid {COLORS[0]};padding:0 6px">solid</span> you checked it</div></div>
<div style="flex:1;min-width:260px"><div style="display:flex;gap:6px;flex-wrap:wrap">
<span class="chip" style="background:var(--brand50);border-color:var(--brand)">All 50</span>
<span class="chip">To check 50</span><span class="chip">Checked 0</span><span class="chip">Nothing found 14</span></div>
<div style="display:flex;gap:8px;margin-top:8px"><div style="position:relative;width:220px">
<img src="{photo_with_boxes}" style="width:100%;border-radius:10px">
<span class="tag" style="position:absolute;left:6px;bottom:8px">AI suggestion</span></div>
<div class="card" style="font-size:13px;flex:1"><b>Words that change</b><table style="margin-top:6px;font-size:12px">
<tr><th>Today</th><th>Proposed</th></tr><tr><td>Labeling</td><td>What to find / How to sort</td></tr>
<tr><td>marks</td><td>found (count) / sorted (classify)</td></tr><tr><td>teacher</td><td>AI</td></tr>
<tr><td>reviewed</td><td>checked by you</td></tr><tr><td>No marks</td><td>Nothing found</td></tr>
<tr><td>sun_cream</td><td>Sun cream</td></tr></table></div></div></div></div></div>"""


def mock_c(grass, border, nograss) -> str:
    def col(name, color, imgs, n, warn=""):
        g = "".join(f'<img src="{s}">' for s in imgs)
        return f"""<div class="col" style="border-color:{color}"><div style="display:flex;justify-content:space-between">
<b>{name}</b><span class="muted" style="font-size:13px">{n} photos</span></div><div class="muted" style="font-size:12px;
margin-bottom:6px">drag a photo here to move it</div><div class="g">{g}</div>{warn}</div>"""
    return f"""<div class="mock"><div style="font-size:12px;font-weight:600;color:var(--muted)">HOW TO SORT: lawn camera
</div><div class="row" style="margin-top:8px">
{col("Grass", COLORS[0], grass, 41)}
{col("Border", COLORS[1], border, 12, '<div class="warn" style="margin-top:8px;font-size:12px;padding:6px 10px">Only 12: add photos of the border, or the model will lean towards grass.</div>')}
{col("No grass", COLORS[2], nograss, 16)}
</div><div class="muted" style="font-size:13px;margin-top:10px">One column per class, like folders. Moving a photo is
checking it. For counting, the same board groups the cut-out objects by class.</div></div>"""


def chat_mock(warn_texts: list[str]) -> str:
    warns = "".join(f'<div class="warn"><span>&#9888;</span><div>{html.escape(t)}</div></div>' for t in warn_texts)
    return f"""<div class="mock" style="display:flex;flex-direction:column;gap:10px">
<div class="bubble me">Count the cars on my street, it is for an ESP32-CAM on the balcony.</div>
<div class="bubble">The AI marked the cars in 8 of your photos. Check them, fix what is wrong, then train.</div>
{warns}
<div style="display:flex;gap:8px"><span class="btn">Check the photos</span><span class="btn">Show me which</span>
<span class="btn p">Train anyway</span></div></div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    results = []
    for d in checks.datasets():
        for hw in dict.fromkeys(([d.hardware] if d.hardware else ["generic"]) + ["esp32cam"]):
            results.append({"dataset": d.name, "hardware": hw, "own": hw == (d.hardware or "generic"),
                            "stats": checks.stats(d, hw), "warnings": checks.check(d, hw)})

    bottle = crops("bottles_few", "bottle")
    sun, water = class_photos("bottles_classify", "sun_cream", 4), class_photos("bottles_classify", "water_bottle", 4)
    grass = [class_photos("grass", c, 6) for c in ("grass", "border", "no_grass")]
    v = json.loads((JOBS / "bottles_few/label/verdicts.json").read_text())
    best = max(v, key=lambda f: len(v[f].get("objects") or []))
    im = Image.open(JOBS / "bottles_few/label/images" / best).convert("RGB")
    im = im.resize((440, round(440 * im.height / im.width)))
    dr = ImageDraw.Draw(im)
    for o in v[best]["objects"]:
        x0, y0, x1, y1 = o["box"]
        for k in range(0, 2):
            dr.rectangle([x0 * im.width + k, y0 * im.height + k, x1 * im.width - k, y1 * im.height - k], outline=COLORS[0])
    bottle_photo = b64(im)

    cars = seen_at("roads_cars", "PXL_20261002_165335487.jpg")
    bott = seen_at("bottles_few", best, sizes=(None, 96))

    rows = []
    for r in results:
        ws = r["warnings"]
        s = r["stats"]
        stat = (f"{s['photos']} fotos, {s.get('objects', '')} objetos, mediana {s.get('median_px')} px a {s['input']} px"
                if s["task"] == "count" else f"{s['photos']} fotos: " + ", ".join(f"{k} {n}" for k, n in s["per_class"].items()))
        items = "".join(
            f"<li><b>{w['check']}</b>{' <small>(opcional)</small>' if w.get('optional') else ''}"
            f"{' <small>(ya existe)</small>' if w.get('existing') else ''}: {html.escape(w['text'])}</li>" for w in ws)
        rows.append(f"<tr><td><b>{html.escape(r['dataset'])}</b><br><small>{stat}; revisadas {s['reviewed']}</small></td>"
                    f"<td>{checks.HW_NAMES.get(r['hardware'], r['hardware'])}{'' if r['own'] else '<br><small>(y si...)</small>'}</td>"
                    f"<td>{'<ul style=margin:0;padding-left:18px>' + items + '</ul>' if ws else '<span class=muted>ninguno</span>'}</td></tr>")

    cars_texts = [w["text"] for r in results if r["dataset"] == "job roads_cars" and r["hardware"] == "esp32cam"
                  for w in r["warnings"] if not w.get("optional")]

    page = f"""<!doctype html><html lang=es><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Etiquetas y avisos</title><style>{CSS}</style>
<h1>Clases comprensibles y avisos sobre los datos</h1>
<p class=muted>Tareas 5 y 9 del 2026-10-12. Propuesta: nada aplicado en el producto. Detalle y fuentes en
<code>docs/propuesta_etiquetas_y_avisos.md</code>; avisos calculados con <code>experiments/data_warnings/checks.py</code>.</p>

<h2>1. Cómo se ve hoy</h2>
<div class=row><div style="flex:1;min-width:300px"><img class=shot src="{shot(args.shots / 'labeling.png')}">
<small>Página del etiquetado (botellas, servidor local web_demo).</small></div>
<div style="flex:1;min-width:300px"><img class=shot src="{shot(args.shots / 'dataset.png')}">
<small>Página del dataset. El perfil dice "sunglasses" en un dataset de botellas (hecho con 1 foto).</small></div></div>
<ol>
<li>La clase es un punto de color y su clave en inglés tal cual (<code>sun_cream</code>, <code>no_grass</code>); "No description".</li>
<li>Ninguna foto de ejemplo por clase: para saber qué es "border" hay que abrir fotos.</li>
<li>Jerga: "labeling", "marks", "teacher", "reviewed", "No marks". Sin leyenda de discontinuo (profesor) frente a continuo (revisada).</li>
<li>El color sale del orden (<code>classHue</code>), no del <code>color</code> que guarda la tabla <code>class</code> (botella: rojo en pantalla, azul #2f6db5 en la base).</li>
<li>Progreso solo por etiquetado; por clase solo "103 marks", sin fotos por clase ni revisadas por clase.</li>
</ol>

<h2>2. Propuestas para las clases</h2>
<h3>A. Tarjetas de clase con fotos de ejemplo <small>(1 a 1,5 días)</small></h3>
<p>Como Teachable Machine y Create ML: cada clase es una tarjeta con su color, 3 o 4 recortes de ejemplo, una frase
y su progreso. Recortes: las mejores cajas del profesor (contar) o las fotos con más confianza (clasificar). El mock junta clases
de dos etiquetados (botellas para contar, crema y botella de agua para clasificar) para enseñar los dos casos; cifras
de revisadas inventadas.</p>
{mock_a(bottle, sun, water)}
<h3>B. Palabras llanas, leyenda y color fijo <small>(medio día, solo web)</small></h3>
<p>Sin tocar la estructura: nombres de clase legibles, vocabulario sin jerga, leyenda de los trazos y el color de la base.</p>
{mock_b(bottle_photo)}
<h3>C. Tablero por clase (carpetas) <small>(2 días)</small></h3>
<p>Como Lobe o las carpetas de Create ML: una columna por clase con sus fotos; mover una foto es revisarla.</p>
{mock_c(*grass)}

<h2>3. Lo que ve el modelo</h2>
<p>La foto se aplasta a un cuadrado del tamaño de entrada del alumno: 320 px (PicoDet, móvil o PC) o 96 px (ESP32-CAM).
Cajas del profesor en amarillo.</p>
<div class=row>{''.join(f'<div><img src="{s}" style="width:300px;border-radius:8px"><br><small>coches: {c}</small></div>' for s, c in cars)}</div>
<div class=row style="margin-top:10px">{''.join(f'<div><img src="{s}" style="width:300px;border-radius:8px"><br><small>botellas: {c}</small></div>' for s, c in bott)}</div>

<h2>4. Avisos que saltarían hoy</h2>
<p>Umbrales propuestos (deciden ustedes): objeto &lt; {checks.SMALL_PX} px en al menos el {checks.SMALL_SHARE:.0%} de una clase;
objetos que el modelo fundiría en uno en al menos el {checks.MERGE_SHARE:.0%}; &lt; {checks.FEW_STRONG} fotos por clase (fuerte) o
&lt; {checks.FEW_MILD} (suave); desequilibrio {checks.IMBALANCE:.0f} a 1 o más; test por clase &lt; {checks.TEST_MIN} fotos.
"Opcional" son comprobaciones extra; "ya existe" es el aviso actual de <code>counting.crowded</code>.
"Y si..." es la misma comprobación con el perfil ESP32-CAM.</p>
<table><tr><th style="width:24%">Dataset</th><th style="width:11%">Hardware</th><th>Avisos (texto exacto)</th></tr>{''.join(rows)}</table>

<h2>5. Cómo se verían en el chat</h2>
<p>Mensaje amarillo del sistema (el mismo <code>Notice</code> ámbar que ya existe), tras marcar el profesor la muestra y antes de
entrenar. Coches en ESP32-CAM:</p>
{chat_mock(cars_texts)}
"""
    args.out.write_text(page)
    print(args.out, round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
