"""Comparison sheet of interview arms (default the model arms S, U, K, Q; --arms picks others) from the runs on disk.

    python experiments/interview_bench/sheet.py --out <file.html> [--notes notes.html] [--excerpts traffic,cat:1]

Per arm: the bench's measures (bench.row) as the mean over all runs, with the mean of each repetition below it
to show the noise; then side by side dialogues of the same persona and repetition; then the notes (HTML) as is.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench import OUT, VARIANTS, row  # noqa: E402
from sim2chip.llm import Client  # noqa: E402

ARMS = ["S", "U", "K", "Q"]
COLS = [  # key, header, format, lower is better
    ("done", "terminadas", "{:.0%}", False),
    ("ok", "spec correcto (terminada y sin fallos)", "{:.0%}", False),
    ("mismatch", "fallos por spec terminado", "{:.2f}", True),
    ("rejected", "specs rechazados por el código", "{:.2f}", True),
    ("format", "respuestas sin formato", "{:.2f}", True),
    ("turns", "turnos del usuario", "{:.2f}", True),
    ("forbidden", "preguntas prohibidas", "{:.2f}", True),
    ("repeat", "preguntas repetidas", "{:.2f}", True),
    ("ask_chars", "caracteres por pregunta", "{:.0f}", True),
    ("llm_s", "s de modelo por entrevista", "{:.0f}", True),
    ("usd", "$ por entrevista", "{:.4f}", True),
]


def load(personas: set[str] | None, arms: list[str]) -> dict[str, list[dict]]:
    out = {}
    for arm in arms:
        recs = [json.loads(f.read_text()) for f in sorted((OUT / arm).glob("*.json"))]
        out[arm] = [r for r in recs if personas is None or r["persona"] in personas]
    return out


def cell(rows: list[dict], key: str) -> float:
    if key == "done":
        return sum(r["outcome"] == "done" for r in rows) / max(1, len(rows))
    if key == "ok":
        return sum(r["outcome"] == "done" and not r["mismatch"] for r in rows) / max(1, len(rows))
    if key == "mismatch":  # among finished interviews: a failed one has no spec to compare
        rows = [r for r in rows if r["outcome"] == "done"]
    return sum(r[key] for r in rows) / max(1, len(rows))


def dialogue_html(rec: dict) -> str:
    lines = []
    for m in rec["dialogue"]:
        who = "EyeForge" if m["role"] == "assistant" else "Usuario"
        lines.append(f"<p class={m['role']}><b>{who}:</b> {html.escape(m['content'])}</p>")
    spec = rec.get("spec") or {}
    tail = (f"<p class=meta>resultado {rec['outcome']}; tarea {spec.get('task')}, hardware {spec.get('hardware')}, "
            f"clases {[c.get('id') for c in spec.get('classes', [])]}; fallos {rec['mismatches'] or 'ninguno'}</p>")
    return "".join(lines) + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--personas", default=None)
    ap.add_argument("--excerpts", default="traffic,cat")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--title", default="Entrevista: qué modelo conversa y quién escribe el spec")
    a = ap.parse_args()
    arms = a.arms.split(",")
    prices = Client().prices()
    runs = load(set(a.personas.split(",")) if a.personas else None, arms)
    rows = {arm: [row(r, prices) | {"rep": r["rep"], "persona": r["persona"]} for r in recs]
            for arm, recs in runs.items()}
    spent = sum(r["usd"] + r["bench_usd"] for rs in rows.values() for r in rs)

    h = ["<!doctype html><meta charset=utf-8><title>Entrevista: modelos</title><style>"
         "body{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;line-height:1.4}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:center}"
         "td.l,th.l{text-align:left}.best{background:#e6f4ea}.note{background:#fff7e6;padding:8px 12px;"
         "border-radius:6px}.cols{display:grid;grid-template-columns:repeat(" + str(len(arms)) + ",1fr);gap:10px}"
         ".cols>div{border:1px solid #ddd;border-radius:6px;padding:6px 10px;font-size:13px}"
         "p.user{color:#225}p.assistant{color:#333}p.meta{color:#777;font-size:12px}</style>",
         f"<h1>{html.escape(a.title)}</h1>",
         "<p>Banco <code>experiments/interview_bench/</code>: la Session real contra usuarios simulados "
         "(Qwen3-235B), análisis de fotos simulado; juez Qwen3-235B para preguntas prohibidas o repetidas; el spec "
         "final se compara con los hechos del usuario. Personas: "
         + ", ".join(sorted({r["persona"] for rs in rows.values() for r in rs}))
         + f". Cada número es la media de todas las entrevistas; debajo, la media de cada repetición (ruido).</p>",
         "<table><tr><th class=l>Brazo</th><th>n</th>"
         + "".join(f"<th>{html.escape(c[1])}</th>" for c in COLS) + "</tr>"]
    for arm in arms:
        rs = rows[arm]
        if not rs:
            continue
        tds = []
        for key, _, fmt, low in COLS:
            v = cell(rs, key)
            vals = [cell(rows[x], key) for x in arms if rows[x]]
            best = (min(vals) if low else max(vals)) == v
            reps = sorted({r["rep"] for r in rs})
            per = " / ".join(fmt.format(cell([r for r in rs if r["rep"] == k], key)) for k in reps)
            tds.append(f"<td class={'best' if best else ''}><b>{fmt.format(v)}</b><br><small>{per}</small></td>")
        h.append(f"<tr><td class=l><b>{arm}</b> {html.escape(VARIANTS[arm]['desc'])}</td><td>{len(rs)}</td>"
                 + "".join(tds) + "</tr>")
    h.append("</table>")
    h.append(f"<p>Gasto del banco en Token Factory (entrevistas, usuario simulado y juez): {spent:.2f} $.</p>")

    h.append("<h2>Fallos de spec por persona (todas las repeticiones)</h2><table><tr><th class=l>Persona</th>"
             + "".join(f"<th>{x}</th>" for x in arms) + "</tr>")
    for p in sorted({r["persona"] for rs in runs.values() for r in rs}):
        tds = []
        for arm in arms:
            got = [", ".join(r["mismatches"]) or ("ok" if r["outcome"] == "done" else r["outcome"])
                   for r in sorted(runs[arm], key=lambda r: r["rep"]) if r["persona"] == p]
            tds.append("<td class=l><small>" + "<br>".join(map(html.escape, got)) + "</small></td>")
        h.append(f"<tr><td class=l>{p}</td>" + "".join(tds) + "</tr>")
    h.append("</table>")

    for item in a.excerpts.split(","):  # persona or persona:rep
        p, _, k = item.partition(":")
        k = int(k or 0)
        h.append(f"<h2>Conversación lado a lado: {p}, repetición {k}</h2><div class=cols>")
        for arm in arms:
            rec = next((r for r in runs[arm] if r["persona"] == p and r["rep"] == k), None)
            h.append(f"<div><h3>{arm}</h3>{dialogue_html(rec) if rec else 'sin datos'}</div>")
        h.append("</div>")
    if a.notes:
        h.append(Path(a.notes).read_text())
    Path(a.out).write_text("\n".join(h))
    print(a.out, f"{spent:.2f} $")


if __name__ == "__main__":
    main()
