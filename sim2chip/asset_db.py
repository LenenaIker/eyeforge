"""Asset knowledge base: what we learned about 3D models, not the models themselves.

Models are found live and downloaded per job; the file cache can be thrown away at any time. What is kept
is one record per (asset, concept) review: where the asset comes from (uid, link, license, author), the
quality verdict for that concept (keep or discard, why, whether it can be repainted) and preparation data
(measured size). Later jobs reuse the verdicts instead of rendering previews and calling the vision model
again, and live search puts known good assets first and skips known bad ones.

A concept is what the asset was searched for: an Objaverse-LVIS category or a live search query, so
"forklift" asked by two different users hits the same records. Storage is append-only JSON lines, one
file per job run (cache/db/reviews/<job>-<time>-<random>.jsonl): no write conflicts between jobs, and it
syncs to Object Storage like any other folder. On read, the newest record per (uid, concept) wins.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

FIELDS = ("uid", "concept", "decision", "no_color", "problems", "reason", "judge", "name", "license", "author",
          "source", "dims")


def concept_key(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text.strip().lower())


class AssetDB:
    def __init__(self, root: Path):
        self.dir = Path(root) / "reviews"
        self._records: dict[tuple[str, str], dict] | None = None

    @property
    def records(self) -> dict[tuple[str, str], dict]:
        if self._records is None:
            recs: dict[tuple[str, str], dict] = {}
            for f in sorted(self.dir.glob("*.jsonl")) if self.dir.exists() else []:
                for line in f.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        key = (r["uid"], r["concept"])
                        if key not in recs or r.get("t", 0) >= recs[key].get("t", 0):
                            recs[key] = r
            self._records = recs
        return self._records

    def lookup(self, uid: str, concept: str) -> dict | None:
        return self.records.get((uid, concept_key(concept)))

    def known(self, concept: str, decision: str) -> list[dict]:
        """Records of a concept with the given decision, newest first."""
        c = concept_key(concept)
        return sorted((r for (u, k), r in self.records.items() if k == c and r["decision"] == decision),
                      key=lambda r: -r.get("t", 0))

    def add(self, records: list[dict], job: str) -> Path | None:
        if not records:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        out = self.dir / f"{job}-{int(now * 1000)}-{uuid.uuid4().hex[:8]}.jsonl"  # never overwrite another run
        with open(out, "w") as f:
            for r in records:
                rec = {k: r.get(k) for k in FIELDS if r.get(k) is not None}
                rec.update(t=round(now, 6), concept=concept_key(r["concept"]), job=job)
                f.write(json.dumps(rec) + "\n")
                self.records[(rec["uid"], rec["concept"])] = rec
        return out
