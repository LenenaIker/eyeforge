"""Scores every runs/teacher_bench/*.json against the user's marks and writes a review page to look at them.

    python experiments/teacher_bench/report.py     -> runs/teacher_bench/review/index.html

A true pot is found when a box contains the user's point; each box finds at most one pot (the nearest).
Unmatched boxes are false marks. Env: any with Pillow.
"""
from __future__ import annotations

import html
import json

import common


def match(boxes: list[dict], pts: list[tuple[float, float]]) -> tuple[list[int | None], list[bool]]:
    """For each true point the index of the box that found it (or None); for each box whether it found one."""
    pairs = []
    for i, (x, y) in enumerate(pts):
        for j, b in enumerate(boxes):
            x0, y0, x1, y1 = b["box"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                pairs.append((((x0 + x1) / 2 - x) ** 2 + ((y0 + y1) / 2 - y) ** 2, i, j))
    found, used = [None] * len(pts), [False] * len(boxes)
    for _, i, j in sorted(pairs):
        if found[i] is None and not used[j]:
            found[i], used[j] = j, True
    return found, used


def main():
    truth = common.truth()
    runs = {}
    for p in sorted(common.OUT.glob("*.json")):
        d = json.loads(p.read_text())
        if "preds" in d:
            for boxes in d["preds"].values():
                for b in boxes:  # some models give the corners in the other order
                    x0, y0, x1, y1 = b["box"]
                    b["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            runs[p.stem] = d
    rows, per_photo = [], {}
    for name, d in runs.items():
        tp = fp = fn = 0
        err = []
        for photo, pts in truth.items():
            boxes = d["preds"].get(photo, [])
            found, used = match(boxes, pts)
            a, b = sum(f is not None for f in found), used.count(False)
            tp, fp, fn = tp + a, fp + b, fn + len(pts) - a
            err.append(abs(len(boxes) - len(pts)))
            rejected = d["meta"].get("rejected", {}).get(photo, [])  # what a verifier threw away
            per_photo.setdefault(photo, {})[name] = (boxes, found, used, rejected)
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        meta = d["meta"]
        speed = meta.get("sec_per_photo") or meta.get("sec_per_photo_parallel") or meta.get("sec_per_box")
        rows.append({"model": name, "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
                     "recall": round(rec, 3), "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3),
                     "count_mae": round(sum(err) / len(err), 2), "speed": speed})
    rows.sort(key=lambda r: -r["f1"])
    print(f"{'model':<18}{'F1':>6}{'prec':>7}{'recall':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'MAE':>6}  speed")
    for r in rows:
        print(f"{r['model']:<18}{r['f1']:>6}{r['precision']:>7}{r['recall']:>8}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
              f"{r['count_mae']:>6}  {r['speed']}")
    (common.OUT / "scores.json").write_text(json.dumps(rows, indent=1))
    page(rows, per_photo, truth)


def page(rows, per_photo, truth):
    out = common.OUT / "review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    for photo in truth:
        dst = out / "img" / photo
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            common.load(photo).save(dst, quality=85)
    models = [r["model"] for r in rows]
    table = "".join(f"<tr><td>{html.escape(r['model'])}</td><td><b>{r['f1']}</b></td><td>{r['precision']}</td>"
                    f"<td>{r['recall']}</td><td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td>"
                    f"<td>{r['count_mae']}</td><td>{r['speed']}</td></tr>" for r in rows)
    cards = []
    for photo, pts in truth.items():
        cells = []
        for m in models:
            boxes, found, used, rejected = per_photo[photo][m]
            svg = []
            for b, cls in [(b, "hit" if ok else "miss") for b, ok in zip(boxes, used)] + [(b, "rej") for b in rejected]:
                (x0, x1), (y0, y1) = sorted(b["box"][::2]), sorted(b["box"][1::2])
                tip = f'<title>{html.escape(b["named"])}</title>' if b.get("named") else ""
                svg.append(f'<rect class="{cls}" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}">{tip}</rect>')
            for (x, y), f in zip(pts, found):
                svg.append(f'<circle class="{"ok" if f is not None else "lost"}" cx="{x}" cy="{y}" r="0.009"/>')
            n_ok = sum(f is not None for f in found)
            cells.append(f'<figure data-model="{html.escape(m)}"><div class="pic"><img src="img/{photo}" loading="lazy">'
                         f'<svg viewBox="0 0 1 1" preserveAspectRatio="none">{"".join(svg)}</svg></div>'
                         f'<figcaption><b>{html.escape(m)}</b> {n_ok}/{len(pts)} found, '
                         f'{used.count(False)} false' + (f', {len(rejected)} rejected' if rejected else '') + '</figcaption></figure>')
        cards.append(f'<section><h3>{photo} <small>{len(pts)} pots</small></h3><div class="row">{"".join(cells)}'
                     f'</div></section>')
    toggles = "".join(f'<label><input type="checkbox" checked value="{html.escape(m)}"> {html.escape(m)}</label>'
                      for m in models)
    (out / "index.html").write_text(TEMPLATE.replace("{table}", table).replace("{toggles}", toggles)
                                    .replace("{cards}", "".join(cards)))
    print(f"review page: {out / 'index.html'}")


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Teacher benchmark</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
table{border-collapse:collapse;margin-bottom:12px}td,th{padding:4px 10px;border-bottom:1px solid #333;text-align:right}
td:first-child,th:first-child{text-align:left}
.legend span{margin-right:16px}.row{display:flex;gap:8px;overflow-x:auto}
figure{margin:0;flex:0 0 360px}.pic{position:relative}.pic img{width:100%;display:block}
.pic svg{position:absolute;inset:0;width:100%;height:100%}
rect{fill:none;stroke-width:.004}rect.hit{stroke:#3af}rect.miss{stroke:#f33}rect.rej{stroke:#999;stroke-dasharray:.01 .006}rect{pointer-events:all}
circle.ok{fill:#2e2}circle.lost{fill:#fa0;stroke:#000;stroke-width:.003}
.big figure{flex-basis:720px}figcaption{padding:4px 0}section{margin:18px 0}
#toggles label{margin-right:12px}
</style></head><body>
<h2>Teacher benchmark: pots in maceta_counter_8 (your 31 corrected photos)</h2>
<table><tr><th>model</th><th>F1</th><th>precision</th><th>recall</th><th>found</th><th>false</th><th>missed</th>
<th>count error</th><th>speed (s)</th></tr>{table}</table>
<p class="legend"><span style="color:#2e2">&#9679; your pot, found</span><span style="color:#fa0">&#9679; your pot, missed</span>
<span style="color:#3af">&#9633; box that found a pot</span><span style="color:#f33">&#9633; false box</span><span style="color:#999">&#9633; dashed: rejected by the verifier (hover: its name)</span></p>
<p id="toggles">{toggles} <label><input type="checkbox" id="big"> large</label></p>
{cards}
<script>
document.querySelectorAll('#toggles input[value]').forEach(c=>c.onchange=()=>document.querySelectorAll(
 `figure[data-model="${c.value}"]`).forEach(f=>f.style.display=c.checked?'':'none'));
document.getElementById('big').onchange=e=>document.body.classList.toggle('big',e.target.checked);
</script></body></html>"""


if __name__ == "__main__":
    main()
