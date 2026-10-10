"""Interview bench: the real interview Session against simulated users, one call setting per variant.

The Nemotron interview runs as in the web app (live 3D model search, live Poly Haven tag check); the photo
analysis and the photo ranker are stubbed with canned answers so only the interview model varies. A second
model plays each persona (personas.py) and, at the end, a judge marks every question the user saw as allowed,
forbidden by the system prompt, or a repeat of something already answered.

    python experiments/interview_bench/bench.py --variants A,B,C --reps 2
    python experiments/interview_bench/bench.py --report       # table of the runs on disk

Per run: questions before the first spec, user turns, rejected specs, format retries, truncated answers,
forbidden or repeated questions, spec mismatches against the persona, seconds and dollars of the interview.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from personas import PERSONAS  # noqa: E402
from sim2chip import interview, scene_profile  # noqa: E402
from sim2chip.llm import Client, parse_json  # noqa: E402

FIXTURES = ROOT / "data" / "interview_bench"
OUT = ROOT / "runs" / "interview_bench"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
USER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
MAX_USER_TURNS = 14

OFF = {"enable_thinking": False}
LOW = {"enable_thinking": True, "low_effort": True}
# interview: overrides for the interview calls; short: chat_template_kwargs for the other text calls, None for
# the model's default (without "short", the code's own setting). Production since 2026-10-03 is "P" (= A8).
# Token Factory ignores reasoning_budget (checked 2026-10-03: 582 reasoning tokens with a budget of 256).
VARIANTS = {
    "P": {"desc": "production settings as in the code"},
    "P1010": {"desc": "production settings, code of 2026-10-10 (scene and slips fixed in code)"},
    "P1010b": {"desc": "P1010 plus LVIS, count, distractor and pitch slips fixed in code, English ids kept"},
    "A": {"desc": "before 2026-10-03: full reasoning, T 0.3, max 4096", "interview": {"max_tokens": 4096},
          "short": None},
    "A8": {"desc": "full reasoning, T 0.3, max 8192, short calls no reasoning",
           "interview": {"max_tokens": 8192}, "short": OFF},
    "B": {"desc": "low effort, T 0.3, max 8192, short calls no reasoning",
          "interview": {"max_tokens": 8192, "chat_template_kwargs": LOW}, "short": OFF},
    "C": {"desc": "no reasoning, T 0.3, max 8192, short calls no reasoning",
          "interview": {"max_tokens": 8192, "chat_template_kwargs": OFF}, "short": OFF},
    "A8T": {"desc": "A8 with T 1.0, top_p 0.95",
            "interview": {"max_tokens": 8192, "temperature": 1.0, "top_p": 0.95}, "short": OFF},
    "BT": {"desc": "B with T 1.0, top_p 0.95",
           "interview": {"max_tokens": 8192, "temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": LOW},
           "short": OFF},
    "CT": {"desc": "C with T 1.0, top_p 0.95",
           "interview": {"max_tokens": 8192, "temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": OFF},
           "short": OFF},
}


class BenchClient:
    """Stands in for llm.Client in the Session: applies the variant and records every call's usage."""

    usage_log = None

    def __init__(self, base: Client, variant: dict):
        self.base, self.variant, self.calls = base, variant, []

    def chat(self, messages, model, purpose, temperature=0.2, max_tokens=2048, json_mode=False, timeout=None,
             retries=None, thinking=None):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if purpose == "interview":
            body |= self.variant.get("interview", {})
        elif "short" in self.variant:
            body.pop("chat_template_kwargs", None)
            if self.variant["short"]:
                body["chat_template_kwargs"] = self.variant["short"]
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        resp = self.base._request("POST", "/chat/completions", body, timeout=timeout, retries=retries)
        choice = resp["choices"][0]
        u = resp.get("usage") or {}
        self.calls.append({
            "purpose": purpose, "latency_s": round(time.time() - t0, 2),
            "prompt": u.get("prompt_tokens") or 0, "completion": u.get("completion_tokens") or 0,
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "finish": choice.get("finish_reason")})
        return choice["message"].get("content") or ""


def make_fixtures() -> None:
    for p in PERSONAS:
        for f in p["folders"].values():
            d = FIXTURES / f["path"]
            d.mkdir(parents=True, exist_ok=True)
            for i in range(f.get("loose", 0)):
                (d / f"photo_{i:03d}.jpg").touch()
            for sub, n in f.get("subs", {}).items():
                (d / sub).mkdir(exist_ok=True)
                for i in range(n):
                    (d / sub / f"{sub}_{i:03d}.jpg").touch()


def fx(text: str) -> str:
    return text.replace("{fx}", str(FIXTURES))


def profiler(persona):
    def profile(folder, classes, progress=None):
        prof = copy.deepcopy(persona["profile"])
        if prof is None:
            raise ValueError("no photos to describe")
        n = len(prof["suggest"]["photos"])
        return prof, scene_profile.summary(prof, n)
    return profile


def ranker(folder, query, progress=None):
    files = sorted(Path(folder).glob("*.jpg"))[:4]
    return [{"file": str(f), "score": 0.8, "reason": f"a clear {query}, whole, plain background"} for f in files]


USER_PROMPT = """You play a user of EyeForge, an app that trains a small camera model for you. You are not a
vision expert. Facts about you (answer from them; reveal a fact only when asked or when it is natural):
{facts}
Rules: reply to the assistant's last message in {lang}, briefly (1 or 2 sentences), like a real person typing in a
chat. If asked something your facts do not cover, give a plausible short answer or say you do not know. Never
invent folder paths; to give photos, write the exact folder from your facts. Reply with your message only."""

CONFIRM_PROMPT = """{facts}
The assistant proposes this setup for your camera model:
---
{summary}
---
If it gets something important wrong compared with your facts (what to detect or count, the device, using only
your own photos when you said so), reply in {lang} with what to change, in one sentence. Otherwise reply exactly
OK."""


def simulated_user(base: Client, persona, seen: list[dict], prompt: str) -> str:
    msgs = [{"role": "system", "content": prompt}] + seen
    return base.chat(msgs, USER_MODEL, purpose="bench_user", temperature=0.3, max_tokens=300).strip()


def run_one(persona, variant_name: str, rep: int) -> dict:
    base = Client()
    client = BenchClient(base, VARIANTS[variant_name])
    uploads = FIXTURES / "uploads" / f"{variant_name}_{persona['id']}_{rep}"
    session = interview.Session(client, MODEL, CATS, scene_check=interview.check_scene, photo_ranker=ranker,
                                uploads=uploads, scene_profiler=profiler(persona))
    facts, lang = fx(persona["facts"]), persona["lang"]
    user_prompt = USER_PROMPT.format(facts=facts, lang=lang)
    seen = []  # the dialogue as the user saw it (assistant = EyeForge)
    shown = []  # {"kind", "text"} for every turn shown to the user
    rejected_confirms = 0
    first_spec_at = None
    wall = 0.0
    error = None

    def step(fn, *args):
        nonlocal wall
        t = time.time()
        try:
            return fn(*args)
        finally:
            wall += time.time() - t

    text = fx(persona["first"])
    seen.append({"role": "assistant", "content": interview.FIRST_QUESTION})
    seen.append({"role": "user", "content": text})
    try:
        turn = step(session.reply, text)
        while turn.kind not in ("done", "failed") and len(shown) < MAX_USER_TURNS:
            shown.append({"kind": turn.kind, "text": turn.intro if turn.kind == "confirm" else turn.text})
            if turn.kind == "confirm":
                first_spec_at = first_spec_at or len(shown)
                verdict = base.chat([{"role": "user", "content": CONFIRM_PROMPT.format(
                    facts=facts, summary=turn.text, lang=lang)}], USER_MODEL, purpose="bench_user",
                    temperature=0.0, max_tokens=200).strip()
                if verdict.strip(". ").upper() == "OK" or rejected_confirms >= 2:
                    turn = step(session.reply, "ok")
                    continue
                rejected_confirms += 1
                seen += [{"role": "assistant", "content": turn.text}, {"role": "user", "content": verdict}]
                turn = step(session.reply, verdict)
            elif turn.kind == "pick":
                files = [c["file"] for c in turn.request["candidates"]]
                seen += [{"role": "assistant", "content": turn.text}, {"role": "user", "content": "(chose all)"}]
                turn = step(session.choose, files, turn.request["class"])
            elif turn.kind == "photos" and (f := persona["folders"].get(turn.request["for"])):
                folder = FIXTURES / f["path"]
                n = interview.count_photos(folder)
                seen += [{"role": "assistant", "content": turn.text},
                         {"role": "user", "content": f"(uploaded {n} photos)"}]
                turn = step(session.add_photos, str(folder), n, turn.request["for"], turn.request.get("class"))
            else:
                seen.append({"role": "assistant", "content": turn.text})
                answer = simulated_user(base, persona, seen, user_prompt)
                seen.append({"role": "user", "content": answer})
                turn = step(session.reply, answer)
        outcome = turn.kind if turn.kind in ("done", "failed") else "too_long"
    except Exception as e:  # a crash is a result too
        outcome, error = "error", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    spec = session.proposed if outcome == "done" else None
    notes = [m["content"] for m in session.messages if m["role"] == "user" and isinstance(m["content"], str)]
    rec = {
        "persona": persona["id"], "variant": variant_name, "rep": rep, "outcome": outcome, "error": error,
        "shown": shown, "dialogue": seen, "confirm_rejections": rejected_confirms,
        "questions_before_spec": (first_spec_at - 1) if first_spec_at else None,
        "user_turns": len(shown),
        "rejected_specs": sum(n.startswith("The spec is invalid") for n in notes),
        "format_retries": sum(n.startswith(("Reply with exactly one JSON", "Your answer is not valid JSON",
                                               "Unknown action", "Your say message"))
                              for n in notes),
        "calls": client.calls, "wall_s": round(wall, 1),
        "spec": json.loads(spec.to_json()) if spec else None,
        "mismatches": mismatches(persona["expect"], spec) if spec else ["no spec"],
        "messages": session.messages,
    }
    rec["judge"] = judge(base, persona, rec)
    return rec


def mismatches(expect: dict, s) -> list[str]:
    out = []
    for key, want in expect.items():
        if key == "task" and s.task != want:
            out.append(f"task {s.task}")
        elif key == "hardware" and s.hardware != want:
            out.append(f"hardware {s.hardware}")
        elif key == "background_class" and s.background_class != want:
            out.append(f"background_class {s.background_class}")
        elif key == "frames" and s.generation.frames != want:
            out.append(f"frames {s.generation.frames}")
        elif key == "errors" and s.errors != want:
            out.append(f"errors {s.errors}")
        elif key == "orientation" and (s.training.orientation or "upright") != want:
            out.append(f"orientation {s.training.orientation}")
        elif key == "names" and not all(c.names for c in s.classes):
            out.append("names missing")
        elif key == "user_data":
            has = "labeled" if s.user_data.labeled else "raw" if s.user_data.raw else None
            if has != want:
                out.append(f"user_data {has}")
    return out


JUDGE_PROMPT = """An assistant interviews a non-expert user to set up a small camera model. Its rules allow
asking ONLY: what the camera must detect or count and what counts as one; where the user's photos are (or
asking them to upload photos, for training or for building 3D models); the device, if the user did not say
it; which mistake hurts more (missing one or a false alarm), when the goal does not make it clear; and whether
a substitute or merge matters when 3D models of an object are lacking. Everything else (sizes, camera height or
distance, lighting, weather, scene, background, orientation, how photos are organised, whether to use
synthetic images, confirmations along the way) it must decide itself, never ask.

Dialogue (assistant = EyeForge); the assistant messages to judge are marked [Q1], [Q2]...:
{dialogue}

For every marked message decide: "forbidden" if it asks something outside the allowed list, "repeat" if it asks
something the user had already answered in a message BEFORE it (even in other words), else "ok". Only what comes
before a message counts for "repeat". A message that only informs, or proposes a setup to confirm, is "ok".
Reply with one JSON object: {{"verdicts": [{{"n": 1, "verdict": "ok|forbidden|repeat", "why": "<short>"}}]}}"""


def judge(base: Client, persona, rec: dict) -> list[dict]:
    asks = {s["text"] for s in rec["shown"] if s["kind"] in ("say", "photos")}
    if not asks:
        return []
    lines, k = [], 0
    for m in rec["dialogue"]:
        if m["role"] == "assistant" and m["content"] in asks and k < len(asks) + 5:
            k += 1
            lines.append(f"[Q{k}] EyeForge: {m['content']}")
        else:
            lines.append(f"{'EyeForge' if m['role'] == 'assistant' else 'User'}: {m['content']}")
    dialogue = "\n".join(lines)
    try:
        v = parse_json(base.chat([{"role": "user", "content": JUDGE_PROMPT.format(dialogue=dialogue)}], JUDGE_MODEL, purpose="bench_judge", temperature=0.0,
            max_tokens=1500))
        return v.get("verdicts", []) if isinstance(v, dict) else []
    except Exception as e:
        return [{"n": 0, "verdict": "judge_error", "why": str(e)[:200]}]


def price(calls: list[dict], prices: dict) -> float:
    p = prices.get(MODEL, {"input": 0, "output": 0})
    return sum(c["prompt"] * p["input"] + c["completion"] * p["output"] for c in calls)


def row(rec: dict, prices: dict) -> dict:
    iv = [c for c in rec["calls"] if c["purpose"] == "interview"]
    verdicts = [v.get("verdict") for v in rec["judge"]]
    return {"outcome": rec["outcome"], "q_before_spec": rec["questions_before_spec"],
            "turns": rec["user_turns"], "rejected": rec["rejected_specs"], "format": rec["format_retries"],
            "truncated": sum(c["finish"] == "length" for c in iv), "forbidden": verdicts.count("forbidden"),
            "repeat": verdicts.count("repeat"), "mismatch": len([m for m in rec["mismatches"] if m != "no spec"]),
            "reasoning": sum(c["reasoning"] for c in iv), "calls": len(iv),
            "llm_s": sum(c["latency_s"] for c in rec["calls"] if not c["purpose"].startswith("bench")),
            "usd": price(rec["calls"], prices)}


def report(variants: list[str] | None = None) -> None:
    prices = Client().prices()
    runs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*/*.json"))]
    names = variants or sorted({r["variant"] for r in runs})
    print(f"{'var':5} {'n':>3} {'done':>5} {'q<spec':>6} {'turns':>5} {'rej':>4} {'fmt':>4} {'trunc':>5} "
          f"{'forb':>4} {'rep':>4} {'mism':>4} {'reas/call':>9} {'s/run':>6} {'$/run':>7}")
    for v in names:
        rows = [row(r, prices) for r in runs if r["variant"] == v]
        if not rows:
            continue
        n = len(rows)
        done = sum(r["outcome"] == "done" for r in rows)
        qs = [r["q_before_spec"] for r in rows if r["q_before_spec"] is not None]
        mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
        calls = sum(r["calls"] for r in rows) or 1
        print(f"{v:5} {n:>3} {done:>5} {sum(qs) / max(1, len(qs)):>6.2f} {mean('turns'):>5.2f} "
              f"{mean('rejected'):>4.2f} {mean('format'):>4.2f} {mean('truncated'):>5.2f} {mean('forbidden'):>4.2f} "
              f"{mean('repeat'):>4.2f} {mean('mismatch'):>4.2f} {sum(r['reasoning'] for r in rows) / calls:>9.0f} "
              f"{mean('llm_s'):>6.1f} {mean('usd'):>7.4f}")
    print()
    for v in names:
        for r in runs:
            if r["variant"] == v and (r["outcome"] != "done" or r["mismatches"]):
                print(f"{v:5} {r['persona']:10} rep{r['rep']} {r['outcome']:8} {r['mismatches']}"
                      + (f" {r['error'].splitlines()[0]}" if r.get("error") else ""))


def main() -> None:
    global CATS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--personas", default="all")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    variants = a.variants.split(",")
    if a.report:
        report(variants if a.variants != "A,B,C" else None)
        return
    make_fixtures()
    objaverse_dir = ROOT / "data" / "cache" / "objaverse"
    CATS = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    people = PERSONAS if a.personas == "all" else [p for p in PERSONAS if p["id"] in a.personas.split(",")]
    jobs = [(p, v, r) for r in range(a.reps) for v in variants for p in people
            if not (OUT / v / f"{p['id']}_{r}.json").exists()]
    print(f"{len(jobs)} runs")

    def go(job):
        p, v, r = job
        rec = run_one(p, v, r)
        (OUT / v).mkdir(parents=True, exist_ok=True)
        (OUT / v / f"{p['id']}_{r}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
        print(f"{v:4} {p['id']:10} rep{r} {rec['outcome']:8} turns {rec['user_turns']} rej {rec['rejected_specs']} "
              f"{rec['mismatches']}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(go, jobs))
    report(variants)


CATS = None

if __name__ == "__main__":
    main()
