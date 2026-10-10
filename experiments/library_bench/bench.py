"""Library bench: the real interview (Nemotron, as the web app builds it) against a copy of a user's library.

It measures how the chat uses datasets the user already has: whether it reuses their labels (the very English
names on the photos), asks before labeling them for new classes, never asks for photos again, and how long and
how many retries that takes. A second model plays the user from each scenario's facts.

    python experiments/library_bench/bench.py --tag before --reps 2
    python experiments/library_bench/bench.py --report

The library copy (library/ and the jobs' small JSON files of the demo server) lives in HOME, on the data disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim2chip import server, teacher  # noqa: E402
from sim2chip.library import Library  # noqa: E402
from sim2chip.llm import Client  # noqa: E402

HOME = ROOT / "data" / "library_bench" / "home"
OUT = ROOT / "runs" / "library_bench"
USER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MAX_USER_TURNS = 8
BINS = {"blue bin", "yellow bin", "brown bin", "gray bin", "green bin"}

SCENARIOS = [
    {"id": "bins", "first": "Quiero contar los contenedores de basura por color (azul, amarillo, marrón, gris y verde) "
                            "en las fotos de la calle del dataset 'Chat photos 09 Oct'. Lo usaré en el móvil.",
     "facts": "Phone. Only the five colored bins, nothing else. Use the photos of that dataset, you have no others. "
              "Balanced errors. If asked about reusing labels the dataset already has, say yes, reuse them.",
     "dataset": "a7d9ef2d4b43c27b", "names": BINS, "ask": False},
    {"id": "bins_new", "first": "Con el dataset 'Chat photos 09 Oct' quiero contar los contenedores por color "
                                "(azul, amarillo, marrón, gris, verde) y también las papeleras. En el móvil.",
     "facts": "Phone. The five colored bins plus small street litter bins (papeleras). If told the dataset is already "
              "labeled for some classes and asked whether to add the new ones, say yes, add the litter bins too.",
     "dataset": "a7d9ef2d4b43c27b", "names": BINS, "ask": True},
    {"id": "table", "first": "Quiero contar los objetos que hay encima de la mesa con mi dataset 'contador mesa', "
                             "para el móvil.",
     "facts": "Phone. Any object lying on the table counts, one class is enough. If asked about the labels the "
              "dataset already has, keep them.",
     "dataset": "8c1b568d88d029d2", "names": {"item", "object"}, "ask": False},
]


class Player:
    """The simulated user: answers the chat from the scenario's facts, briefly, in Spanish."""

    def __init__(self, client: Client, sc: dict):
        self.client, self.sc = client, sc

    def reply(self, history: list[tuple[str, str]], question: str) -> str:
        talk = "\n".join(f"{who}: {text}" for who, text in history[-8:])
        prompt = (f"You play a user of an app that builds camera models, chatting in Spanish. Your facts: "
                  f"{self.sc['facts']}\nConversation so far:\n{talk}\nThe app now says:\n{question}\n"
                  "Answer as the user in one or two short sentences, in Spanish. If it proposes a setup that fits "
                  "your facts, answer exactly: sí. Never invent photos or folders.")
        return self.client.chat([{"role": "user", "content": prompt}], USER_MODEL, purpose="bench_user",
                                temperature=0.3, max_tokens=200).strip()


def run_one(sc: dict, tag: str, rep: int) -> dict:
    out = OUT / tag / f"{sc['id']}_{rep}"
    out.mkdir(parents=True, exist_ok=True)
    lib = Library(HOME, HOME / "cache")
    owner = lib.db.execute("SELECT owner FROM dataset WHERE id = ?", (sc["dataset"],)).fetchone()[0]
    session = server.default_interviewer(HOME, backend="nebius", public=True)()
    session.client.usage_log = out / "usage.jsonl"
    session.uploads = out / "uploads"
    session.transcript = out / "transcript.json"
    server.attach_library(session, lib, owner, out / "uploads")
    player = Player(Client(usage_log=out / "user_usage.jsonl"), sc)
    t0, history, turns, kinds = time.time(), [], 0, []
    turn, text, err = None, sc["first"], None
    try:
        while turns < MAX_USER_TURNS:
            history.append(("user", text))
            turn = session.reply(text)
            turns += 1
            kinds.append(turn.kind)
            history.append(("app", turn.text))
            if turn.kind in ("done", "failed"):
                break
            text = player.reply(history, turn.text)
    except Exception as e:  # a crash is a result too
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    notes = [m["content"] for m in session.messages if m["role"] == "user" and isinstance(m.get("content"), str)]
    spec = turn.spec if turn is not None and turn.kind == "done" else session.proposed
    names = set()
    raw = None
    if spec is not None:
        # What the teacher will look for (SAM 3's first phrase per class), the name the labels are kept under.
        names = {p[0] for p in teacher.phrases(spec).values()} if spec.task == "count" else set(spec.labels)
        raw = spec.user_data.raw
    r = {"scenario": sc["id"], "tag": tag, "rep": rep, "seconds": round(time.time() - t0, 1), "user_turns": turns,
         "kinds": kinds, "done": bool(turn and turn.kind == "done"), "error": err,
         "format_retries": sum(n.startswith("Your answer is not valid JSON") for n in notes),
         "invalid_specs": sum(n.startswith("The spec is invalid") for n in notes),
         "relabel_notes": sum("[Labels of dataset" in n for n in notes),
         "asked_photos": "photos" in kinds,
         "names": sorted(names), "reused": sorted(names & sc["names"]), "expected": sorted(sc["names"]),
         "uses_dataset": sc["dataset"] in json.dumps(raw),
         "prompt_tokens": sum(json.loads(l).get("prompt_tokens", 0) for l in (out / "usage.jsonl").read_text().splitlines())
         if (out / "usage.jsonl").exists() else 0}
    (out / "result.json").write_text(json.dumps(r, indent=1))
    (out / "talk.txt").write_text("\n\n".join(f"{w}: {t}" for w, t in history))
    return r


def report() -> None:
    rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*/*/result.json"))]
    print(f"{'tag':10} {'scenario':9} {'s':>6} {'turns':>5} {'done':>4} {'fmt':>3} {'inv':>3} {'ask':>3} "
          f"{'photo':>5} {'reused':>7} names")
    for r in rows:
        print(f"{r['tag']:10} {r['scenario']:9} {r['seconds']:6.0f} {r['user_turns']:5} {int(r['done']):4} "
              f"{r['format_retries']:3} {r['invalid_specs']:3} {r['relabel_notes']:3} {int(r['asked_photos']):5} "
              f"{len(r['reused'])}/{len(r['expected']):<5} {', '.join(r['names'])}{'  ERROR ' + r['error'] if r['error'] else ''}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--only", default="")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report()
        return 0
    todo = [(sc, rep) for sc in SCENARIOS if not a.only or sc["id"] in a.only.split(",") for rep in range(a.reps)]
    with ThreadPoolExecutor(4) as pool:
        for r in pool.map(lambda x: run_one(x[0], a.tag, x[1]), todo):
            print(json.dumps(r))
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
