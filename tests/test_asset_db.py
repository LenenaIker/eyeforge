"""Asset knowledge base: verdicts per (asset, concept), reused by QA instead of judging again."""
import json

from sim2chip.asset_db import AssetDB, concept_key
from sim2chip.spec import from_dict
from sim2chip.stages import Context
from sim2chip.stages.qa_assets import QAAssets
from sim2chip.workspace import Cache, open_job


def test_records_newest_wins_and_concepts_normalize(tmp_path):
    db = AssetDB(tmp_path)
    db.add([{"uid": "a", "concept": "Forklift  Truck", "decision": "keep", "license": "by"}], "job1")
    db.add([{"uid": "a", "concept": "forklift truck", "decision": "discard", "reason": "toy"}], "job2")
    db.add([{"uid": "b", "concept": "forklift_truck", "decision": "keep"}], "job2")
    fresh = AssetDB(tmp_path)  # read back from the files
    assert fresh.lookup("a", "forklift truck")["decision"] == "discard"
    assert [r["uid"] for r in fresh.known("FORKLIFT truck", "keep")] == ["b"]
    assert fresh.lookup("a", "excavator") is None
    assert concept_key(" Car_(automobile) ") == "car (automobile)"
    assert len(list((tmp_path / "reviews").glob("*.jsonl"))) == 3


def test_qa_reuses_known_verdicts_and_records_new_ones(tmp_path, monkeypatch):
    spec = from_dict({"name": "t", "qa": {"mode": "auto"},
                      "classes": [{"id": "forklift", "size_m": {"measure": "length", "range": [2.5, 3.5]},
                                   "assets": {"source": "search", "queries": ["forklift"], "count": 5}}]})
    ctx = Context(spec, open_job(tmp_path, "job"), Cache(tmp_path / "cache"))
    assets = [{"uid": u, "class": "forklift", "category": "forklift", "usd": f"usd/{u}.usd", "dims": [3, 1, 2],
               "name": f"model {u}", "license": "by"} for u in "abc"]
    (ctx.out("convert_assets") / "assets.json").write_text(json.dumps(assets))
    AssetDB(ctx.cache.db).add([{"uid": "a", "concept": "forklift", "decision": "keep"},
                               {"uid": "b", "concept": "forklift", "decision": "discard", "reason": "toy"}], "old")
    judged = []

    def fake_judge(self, ctx, todo):
        judged.extend(a["uid"] for a in todo)
        return {"c": {"decision": "keep", "class": "forklift", "no_color": True, "reason": "ok", "judge": "m"}}

    monkeypatch.setattr(QAAssets, "judge", fake_judge)
    info = QAAssets().run(ctx)
    assert judged == ["c"] and info["kept"] == {"forklift": 2}
    rec = AssetDB(ctx.cache.db).lookup("c", "forklift")
    assert rec["decision"] == "keep" and rec["no_color"] is True and rec["judge"] == "m" and rec["license"] == "by"
    judged.clear()
    QAAssets().run(ctx)  # second run: everything known, nothing judged
    assert judged == []
