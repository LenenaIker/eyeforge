"""Live asset search against a fake Sketchfab: pagination, licenses, mirror filter, interleaving."""
import pytest

from sim2chip import asset_search, interview
from sim2chip.asset_search import Mirror, find, search
from sim2chip.spec import SpecError, from_dict

CC_BY = {"label": "CC Attribution"}


def result(uid, lic=CC_BY, **kw):
    return {"uid": uid, "name": f"model {uid}", "license": lic, "isDownloadable": True, "user": {"username": "a"},
            **kw}


PAGES = {
    "boar": [{"results": [result("b1"), result("b2", {"label": "CC Attribution-NonCommercial"}),
                          result("b3", isAgeRestricted=True)], "next": "page2"},
             {"results": [result("b4"), result("b5")], "next": None}],
    "pig": [{"results": [result("p1"), result("b1"), result("p2")], "next": None}],
}


def fake_fetch(url):
    if url == "page2":
        return PAGES["boar"][1]
    q = "boar" if "q=boar" in url else "pig"
    return PAGES[q][0]


def test_search_filters_licenses_and_paginates():
    hits = search("boar", fetch=fake_fetch)
    assert [h.uid for h in hits] == ["b1", "b4", "b5"]
    assert hits[0].license == "by" and hits[0].source.endswith("/b1") and hits[0].query == "boar"


def test_find_interleaves_dedupes_and_keeps_mirror_only():
    mirror = Mirror(uids={"b1", "b4", "p1", "p2"})
    hits = find(["boar", "pig"], count=10, mirror=mirror, fetch=fake_fetch)
    assert [h.uid for h in hits] == ["b1", "p1", "b4", "p2"]
    assert [h.uid for h in find(["boar", "pig"], 2, mirror, fake_fetch)] == ["b1", "p1"]
    assert [h.uid for h in find(["boar"], 5, mirror, fake_fetch, exclude={"b1"})] == ["b4"]


def test_spec_search_source():
    base = {"name": "t", "classes": [{"id": "boar", "size_m": {"measure": "length", "range": [1, 1.8]},
                                      "assets": {"source": "search", "queries": ["wild boar"]}}]}
    assert from_dict(base).classes[0].assets.queries == ["wild boar"]
    bad = {**base, "classes": [{**base["classes"][0], "assets": {"source": "search"}}]}
    with pytest.raises(SpecError, match="need queries"):
        from_dict(bad)
    with pytest.raises(SpecError, match="assets.source"):
        from_dict({**base, "classes": [{**base["classes"][0], "assets": {"source": "web", "queries": ["x"]}}]})


def test_interview_search_reports_live_counts():
    live = lambda q: [asset_search.Hit(f"{q}{i}", f"{q} {i}", "by", "a", "", q) for i in range(3)]  # noqa: E731
    cats = interview.CategoryIndex(counts={"cow": 62}, live=live)
    r = cats.search(["cow", "wild boar"])
    assert r["curated_categories"] == {"cow": 62}
    assert r["live_search"]["wild boar"] == {"models": 3, "examples": ["wild boar 0", "wild boar 1", "wild boar 2"]}


def test_polyhaven_picks():
    from sim2chip import polyhaven
    hdris = {"a": {"categories": ["outdoor", "nature", "clear"], "tags": ["forest"]},
             "b": {"categories": ["outdoor", "nature", "night"], "tags": ["forest"]},
             "c": {"categories": ["indoor", "studio"], "tags": []},
             "d": {"categories": ["indoor", "artificial light"], "tags": ["warehouse"]}}
    assert polyhaven.pick_skies(hdris, ["forest"], ["clear"], 5, 0) == ["a"]
    assert sorted(polyhaven.pick_skies(hdris, ["forest"], ["clear", "night"], 5, 0)) == ["a", "b"]
    assert polyhaven.pick_skies(hdris, ["indoor"], ["clear"], 5, 0) == ["d"]  # weather ignored, no studios
    assert polyhaven.pick_skies(hdris, ["indoor", "studio"], ["clear"], 5, 0) == ["c"]
    tex = {"g": {"categories": ["terrain"], "tags": ["grass", "Leaves"]}, "r": {"categories": ["road"], "tags": []}}
    assert polyhaven.pick_textures(tex, ["leaves"], 5, 0) == ["g"]
