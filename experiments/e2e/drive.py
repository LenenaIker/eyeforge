"""End to end through the web API, as a user: chat (a simulated user answers), confirm the setup, accept the
teacher's check if one comes, then follow the run to its end and sum it up.

    python experiments/e2e/drive.py --base http://127.0.0.1:8100 --case bins
    python experiments/e2e/drive.py --base http://127.0.0.1:8100 --job <name>     # follow a job only

The server is any `sim2chip serve` (local or Nebius backend) whose home holds the library the cases name; a
non-public server needs no sign-in. Writes runs/e2e/<job>.json with the chat, the timeline and the result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "library_bench"))

from bench import Player  # noqa: E402
from sim2chip.llm import Client  # noqa: E402

OUT = ROOT / "runs" / "e2e"
CASES = {
    "bins": {"first": "Quiero contar los contenedores de basura por color (azul, amarillo, marrón, gris y verde) en "
                      "las fotos del dataset 'Chat photos 09 Oct'. Lo usaré en el móvil.",
             "facts": "Phone. Only the five colored bins. Use the photos of that dataset, you have no others. Balanced "
                      "errors. If asked about reusing the dataset's labels, say yes."},
    "table": {"first": "Quiero contar los objetos que hay encima de la mesa con mi dataset 'contador mesa', para el "
                       "móvil.",
              "facts": "Phone. Any object lying on the table counts, one class is enough. Keep the labels the dataset "
                       "already has."},
}


def chat(api: httpx.Client, case: dict, log: list) -> tuple[dict, str]:
    sid = api.post("/api/interview").json()["session"]
    player, text, history = Player(Client(), case), case["first"], []
    for _ in range(10):
        t0 = time.time()
        r = api.post(f"/api/interview/{sid}", json={"text": text}, timeout=600)
        r.raise_for_status()
        turn = r.json()
        log.append({"user": text, "app": turn.get("intro") or turn["text"], "kind": turn["kind"],
                    "s": round(time.time() - t0, 1)})
        print(f"[chat {log[-1]['s']} s] {turn['kind']}: {log[-1]['app'][:300]}", flush=True)
        history += [("user", text), ("app", turn["text"])]
        if turn["kind"] == "done":
            return turn["spec"], sid
        if turn["kind"] == "failed":
            raise RuntimeError(f"chat failed: {turn['text']}")
        text = player.reply(history, turn["text"])
    raise RuntimeError("chat did not end in 10 turns")


def follow(api: httpx.Client, job: str, timeline: list, deadline_h: float = 3) -> dict:
    t0, seen, accepted = time.time(), None, False
    while time.time() - t0 < deadline_h * 3600:
        s = api.get(f"/api/jobs/{job}", timeout=60).json()
        cal = s.get("calibration")
        now = (s.get("phase"), cal and cal.get("state"), tuple((x["name"], x["state"]) for x in s["stages"]))
        if now != seen:
            seen = now
            line = {"t": round(time.time() - t0), "phase": s.get("phase"), "check": cal and cal.get("state"),
                    "stages": {x["name"]: x["state"] for x in s["stages"]}}
            timeline.append(line)
            print(f"[{line['t']:5d} s] phase={line['phase']} check={line['check']} "
                  + " ".join(f"{k}:{v}" for k, v in line["stages"].items()), flush=True)
        if cal and cal.get("state") == "failed":
            return {"outcome": "check failed", "error": cal.get("error"), "log": s["log"][-15:]}
        if cal and cal.get("state") == "ready" and not accepted:
            accepted = True  # the user looks at the teacher's marks and trains as they are
            r = api.post(f"/api/jobs/{job}/train", json={"labels": {}, "points": {}}, timeout=120)
            print(f"[check] train as marked: {r.status_code} {r.text[:200]}", flush=True)
            if r.status_code >= 400:
                return {"outcome": "train refused", "error": r.text}
        if not s["running"] and not cal and (s.get("report") or any(x["state"] == "failed" for x in s["stages"])
                                             or any(l.startswith("FAILED") for l in s["log"][-5:])):
            rep = s.get("report") or {}
            real = rep.get("real_test") or {}
            return {"outcome": "done" if rep else "failed", "metric": real.get("metric"), "score": real.get("mean"),
                    "shipped": rep.get("shipped_test"), "left_out": rep.get("left_out"), "kb": rep.get("model_kb"),
                    "usd": s.get("cloud_usd"), "engine": (s.get("version") or {}).get("engine"),
                    "failed": [x["name"] for x in s["stages"] if x["state"] == "failed"], "log": s["log"][-15:]}
        time.sleep(15)
    return {"outcome": "timeout"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8100")
    p.add_argument("--case", choices=sorted(CASES))
    p.add_argument("--job")
    a = p.parse_args()
    api = httpx.Client(base_url=a.base)
    record: dict = {"base": a.base, "case": a.case, "chat": [], "timeline": []}
    t0 = time.time()
    if a.job:
        job = a.job
    else:
        spec, sid = chat(api, CASES[a.case], record["chat"])
        record["spec"] = spec
        r = api.post("/api/jobs", data={"spec": json.dumps(spec), "calibrate": "true", "session": sid}, timeout=600)
        if r.status_code >= 400:
            print("job refused:", r.text)
            return 1
        job = r.json()["job"]
        print(f"[job] {job}: {r.json()}", flush=True)
    record["job"] = job
    record["result"] = follow(api, job, record["timeline"])
    record["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{job}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in record["result"].items() if k != "log"}, indent=1))
    print("\n".join(record["result"].get("log") or []))
    return 0 if record["result"]["outcome"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
