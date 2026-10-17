"""S.6 for the bins with their holders (docs/roadmap_photosphere.md, S.6c): the context of the user's train photos
with the boxes the user reviewed (the teacher missed the hanging buckets), what holds each bin (support.py, the
local vision model), the holders it names segmented (steps/photo_holders.py) and measured into rigs
(context.measure_rigs), for s6_sheet.py --context ... --rigs ....

    python experiments/sphere/bins_rigs.py [--job bins_count] [--out data/sphere_test/bins_train]

Only train photos: the test stays out of what the synthetic copies.
"""
import argparse
import json
from pathlib import Path

from sim2chip.photo2scene import Flow
from sim2chip.photo2scene.sphere import GROUND_KINDS, STRUCTURE_PHRASES, context, support

ROOT = Path(__file__).resolve().parents[2]
HOME = Path("/media/lenena-iker/Crucial/runs/eyeforge/p2s_test")
STEPS = ROOT / "sim2chip/photo2scene/sphere/steps"


def verdicts(points: dict) -> dict:
    """ingest's points ({hash: {"key", "points": [[x, y, class, w, h]]}}) as label/verdicts.json's objects."""
    return {v["key"]: {"objects": [{"class": c, "box": [round(x - w / 2, 4), round(y - h / 2, 4),
                                                          round(x + w / 2, 4), round(y + h / 2, 4)]}
                                   for x, y, c, w, h in v["points"]]}
            for v in points.values() if v["points"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="bins_count")
    ap.add_argument("--photos", type=Path, default=ROOT / "data/real/container")
    ap.add_argument("--out", type=Path, default=ROOT / "data/sphere_test/bins_train")
    args = ap.parse_args()
    out = args.out
    flow = Flow(out)
    v = verdicts(json.loads((HOME / "jobs" / args.job / "ingest/train/points.json").read_text()))
    names = ",".join(sorted(v))
    print(f"{len(v)} train photos with targets, {sum(len(x['objects']) for x in v.values())} targets", flush=True)
    flow.run("bins_depth", "mesh", str(STEPS / "photo_depth.py"),
             ["--photos", args.photos, "--names", names, "--out", out / "depth"])
    groups = [a for kind, ph in GROUND_KINDS.items() for a in ("--group", f"{kind}={','.join(ph)}")]
    flow.run("bins_stuff", "sam3", str(STEPS / "photo_stuff.py"),
             ["--photos", args.photos, "--names", names, "--out", out / "stuff", *groups,
              "--group", "structure=" + ",".join(STRUCTURE_PHRASES)])
    measured = context.measure(args.photos, v, out / "depth", out / "stuff", out / "context.json")
    if (out / "support.json").exists():
        sup = json.loads((out / "support.json").read_text())
    else:
        sup = support.run(args.photos, measured, out / "support.json")
        support.sheet(args.photos, sup, out / "sheet_support.jpg")
    ts = [t for x in sup.values() for t in x]
    known = [t for t in ts if t["agree"] is not None]
    print(f"support: {len(ts)} targets, {sum(t['support'] is None for t in ts)} unsure, agree with the base on "
          f"{sum(t['agree'] for t in known)} of {len(known)}", flush=True)
    # a target hangs where the model and the measured base agree; the others stay out of the rigs
    agreed = {n: [{**t, "mounted": bool(t.get("mounted")) and s["support"] == "hangs"} for t, s in zip(x, sup[n])]
              for n, x in measured.items() if n in sup}
    (out / "context_agreed.json").write_text(json.dumps(agreed, indent=1))  # s6_sheet.py --context
    # each photo looks for what holds its own hanging targets, in the model's words
    wanted = {n: sorted({s["holder"] for t, s in zip(agreed[n], sup[n]) if t["mounted"] and s["holder"]})
              for n in agreed}
    wanted = {n: p for n, p in wanted.items() if p}
    print(f"holders named: {wanted}", flush=True)
    if not wanted:
        return
    (out / "holder_phrases.json").write_text(json.dumps(wanted, indent=1))
    flow.run("bins_holders", "sam3", str(STEPS / "photo_holders.py"),
             ["--photos", args.photos, "--out", out / "holders", "--phrases", out / "holder_phrases.json"])
    rigs = context.measure_rigs(args.photos, agreed, out / "depth", out / "stuff", out / "holders",
                                out / "rigs.json")
    for name, rs in rigs.items():
        for r in rs:
            print(name, "standing" if r["standing"] else "hung", len(r["holders"]), "holders",
                  [(t["class"], round(t["x"], 2), t["base_m"]) for t in r["targets"]], flush=True)


if __name__ == "__main__":
    main()
