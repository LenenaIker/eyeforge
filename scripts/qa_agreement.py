"""Agreement between the automatic asset QA and a human review.

    python scripts/qa_agreement.py data/jobs/lora_alarm_traffic_autoqa data/review/qa_v3

The human review folder holds exclude.txt (discarded uids) and optionally class_override.json.
Prints keep/discard agreement, the confusion matrix, class agreement on kept models and the
disagreements with the model's reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(job: Path, review: Path) -> None:
    assets = json.loads((job / "convert_assets" / "assets.json").read_text())
    verdicts = json.loads((job / "qa_assets" / "verdicts.json").read_text())
    human_out = set((review / "exclude.txt").read_text().split())
    ov_file = review / "class_override.json"
    human_cls = {u: v["class"] for u, v in json.loads(ov_file.read_text()).items()} if ov_file.exists() else {}

    cm = {("keep", "keep"): 0, ("keep", "discard"): 0, ("discard", "keep"): 0, ("discard", "discard"): 0}
    cls_ok = cls_n = 0
    rows = []
    for a in assets:
        u = a["uid"]
        h = "discard" if u in human_out else "keep"
        v = verdicts.get(u) or {"decision": "discard", "reason": "no verdict"}
        m = v["decision"]
        cm[(h, m)] += 1
        if h == m == "keep":
            cls_n += 1
            cls_ok += v.get("class", a["class"]) == human_cls.get(u, a["class"])
        if h != m:
            rows.append((a["class"], h, m, a.get("name"), v.get("reason", "")))
    n = sum(cm.values())
    agree = cm[("keep", "keep")] + cm[("discard", "discard")]
    print(f"assets {n}, keep/discard agreement {agree}/{n} = {agree / n:.1%}")
    print(f"  human keep:    model keep {cm[('keep', 'keep')]:4d}, model discard {cm[('keep', 'discard')]:4d}")
    print(f"  human discard: model keep {cm[('discard', 'keep')]:4d}, model discard {cm[('discard', 'discard')]:4d}")
    if cls_n:
        print(f"class agreement on models both keep: {cls_ok}/{cls_n} = {cls_ok / cls_n:.1%}")
    per_class: dict[str, list[int]] = {}
    for a in assets:
        h = "discard" if a["uid"] in human_out else "keep"
        m = (verdicts.get(a["uid"]) or {"decision": "discard"})["decision"]
        per_class.setdefault(a["class"], [0, 0])
        per_class[a["class"]][0] += h == m
        per_class[a["class"]][1] += 1
    for c, (ok, tot) in per_class.items():
        print(f"  {c:16s} {ok}/{tot} = {ok / tot:.0%}")
    print("\ndisagreements (class, human, model, name, model reason):")
    for r in sorted(rows):
        print("  " + " | ".join(str(x) for x in r))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
