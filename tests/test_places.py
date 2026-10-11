"""Procedural places and the scene profile, without Isaac Sim or a model."""
import json
import random

import pytest

from sim2chip import interview, scene_profile
from sim2chip.scenes import places_layout as pl
from sim2chip.spec import SpecError, from_dict

SPEC = {"name": "t", "task": "count", "classes": [{"id": "pot", "size_m": {"measure": "height", "range": [0.2, 0.4]},
                                                   "assets": {"source": "search", "queries": ["flower pot"]}}],
        "scene": {"template": "generic"}}


def test_building_has_the_supports_and_sane_sizes():
    kinds = set()
    for seed in range(20):
        lay = pl.building(random.Random(seed), ["bars", "solid"])
        W, H = lay.size
        assert 9 <= W <= 27 and 8 <= H <= 19.2
        kinds |= {s.kind for s in lay.surfaces}
        for s in lay.surfaces:
            assert s.x[0] <= s.x[1] and s.y[0] <= s.y[1] and s.z >= 0
            assert s.y[1] <= 0  # everything stands in front of the building front
        for boxes in lay.boxes.values():
            for _, size in boxes:
                assert min(size) > 0
    assert kinds == {"ground", "balcony_floor", "railing", "window_sill"}


def test_room_surfaces_are_inside():
    lay = pl.room(random.Random(3))
    ox, oy = pl.ROOM_AT
    sx, sy = lay.size
    for s in lay.surfaces:
        assert ox - sx / 2 <= s.x[0] and s.x[1] <= ox + sx / 2 and oy - sy / 2 <= s.y[0] and s.y[1] <= oy + sy / 2
    assert {"floor", "window_sill", "furniture"} == {s.kind for s in lay.surfaces}


def test_place_no_overlap_and_hanging_outside():
    rng = random.Random(0)
    floor = pl.Surface("balcony_floor", (0, 2), (-1.5, -0.2), 3.0)
    hung = pl.Surface("railing", (0, 2), (-1.7, -1.7), 3.65, hang=True)
    spots = pl.place([floor, hung], [(0.15, 0.3)] * 8, rng)
    got = [p for p in spots if p]
    assert len(got) >= 6 and spots[0][2] == 3.0
    for i, a in enumerate(got):
        for b in got[i + 1:]:
            assert a[2] != b[2] or ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 > 0.3
    assert all(p[1] < -1.7 for p in got if p[2] == 3.65)  # hung on the street side of the railing


def test_shots_look_at_what_they_place():
    rng = random.Random(1)
    lay = pl.building(rng, ["bars"])
    eye, look, surf = pl.facade_shot(lay, ["window_sill", "railing"], (6, 20), 70, 4 / 3, rng)
    assert eye[1] < -5 and surf[0].kind in ("window_sill", "railing")
    for s in surf[1:]:
        assert pl.in_view(eye, look, pl._mid(s), 70, 4 / 3)
    b = pl.balcony_shot(lay, ["balcony_floor"], rng)
    if b:
        eye, look, surf = b
        assert all(s.kind == "balcony_floor" for s in surf) and look[1] < eye[1]


def test_in_view():
    assert pl.in_view((0, -10, 1.5), (0, 0, 1.5), (1, 0, 2), 60, 4 / 3)
    assert not pl.in_view((0, -10, 1.5), (0, 0, 1.5), (9, 0, 2), 60, 4 / 3)
    assert not pl.in_view((0, -10, 1.5), (0, 0, 1.5), (0, -20, 1.5), 60, 4 / 3)  # behind


def test_mesh_counts_and_texture_scale():
    pts, counts, idx, uv = pl.mesh([((0, 0, 1), (4, 0.2, 2))])
    assert len(pts) == 24 and counts == [4] * 6 and max(idx) == 23
    front_u = [u for u, _ in uv[:4]]
    assert max(front_u) - min(front_u) == pytest.approx(4 / pl.TILE_M)  # 4 m wide: two texture tiles


def test_contexts_in_spec():
    s = from_dict({**SPEC, "scene": {"template": "generic", "wall_tags": ["brick"], "railings": ["bars"],
                                     "contexts": [{"kind": "facade", "share": 0.7, "distance_m": [5, 20],
                                                   "supports": ["railing"]}, {"kind": "indoor", "share": 0.3}]}})
    assert [c.kind for c in s.scene.contexts] == ["facade", "indoor"] and s.scene.contexts[0].distance_m == (5, 20)
    assert json.loads(s.to_json())["scene"]["contexts"][1]["supports"] == []
    bad = [({"contexts": [{"kind": "roof"}]}, "kind must be"),
           ({"contexts": [{"kind": "indoor", "supports": ["railing"]}]}, "not possible"),
           ({"contexts": [{"kind": "facade", "share": 0}]}, "share"),
           ({"wall_tags": ["marble"]}, "wall_tags"),
           ({"template": "road", "contexts": [{"kind": "facade"}]}, "generic")]
    for extra, msg in bad:
        with pytest.raises(SpecError, match=msg):
            from_dict({**SPEC, "scene": {"template": "generic", **extra}})
    # A spec without contexts keeps its fingerprint: the new fields are empty and pruned.
    assert from_dict(SPEC).section("scene") == from_dict({**SPEC, "scene": {"template": "generic", "contexts": []}}
                                                         ).section("scene")


def test_profile_aggregate():
    d = [{"context": "facade", "distance_m": 10, "supports": ["railing", "window_sill"], "wall": "brick",
          "railing": "bars"},
         {"context": "facade", "distance_m": "20", "supports": ["railing", "floor"], "wall": "plaster",
          "railing": "glass"},
         {"context": "balcony", "distance_m": 1.5, "supports": ["balcony_floor"], "wall": "brick", "railing": "solid"},
         {"context": "balcony", "distance_m": 2, "supports": [], "wall": "brick", "railing": "solid"},
         {"context": "indoor", "distance_m": None, "supports": [], "wall": "none", "railing": "none"}] + \
        [{"context": "facade", "distance_m": 15, "wall": "brick", "railing": "bars"}] * 7
    prof = scene_profile.aggregate(d)
    kinds = {c["kind"]: c for c in prof["contexts"]}
    assert set(kinds) == {"facade", "balcony"}  # a single room in 12 photos is under MIN_SHARE
    assert kinds["facade"]["distance_m"] == [7.0, 26.0] and kinds["facade"]["supports"] == ["railing", "window_sill"]
    assert prof["wall_tags"][0] == "brick" and prof["railings"] == ["bars", "solid"]  # glass counts as bars
    scene = {k: v for k, v in prof.items() if k != "suggest"}
    from_dict({**SPEC, "scene": {"template": "generic", **scene}})  # the proposal is a valid spec
    assert "75% building fronts" in scene_profile.summary(prof, 12)


def test_profile_before_the_classes_does_not_size_them():
    """A lawn described before anyone chose the classes: its most numerous thing is the leaves, whose count and
    size must not be proposed for the class (grass got 35 per photo of 0.08-0.75 m, 2026-10-11)."""
    lawn = [{"context": "ground", "count": 35, "size_m": s, "height_m": 0.01, "distance_m": 1.5,
             "objects": [{"name": "leaf", "count": 35, "size_m": s}, {"name": "grass", "count": 1, "size_m": 5}]}
            for s in (0.08, 0.1, 0.75)]
    sug = scene_profile.aggregate(lawn, general=True)["suggest"]
    assert not {"size_m", "height_m", "objects_per_photo"} & set(sug)
    assert {"name": "leaf", "share": 1.0, "count": 35, "size_m": 0.1} in sug["seen"]
    # Described for the user's classes, the size and count are of those classes.
    sug = scene_profile.aggregate(lawn)["suggest"]
    assert sug["size_m"]["range"] == [0.06, 0.94] and sug["objects_per_photo"] == {"median": 35, "max": 35}
    # Surfaces answer null: nothing proposed either.
    flat = [{**d, "count": None, "size_m": None, "height_m": None} for d in lawn]
    assert not {"size_m", "height_m", "objects_per_photo"} & set(scene_profile.aggregate(flat)["suggest"])


class Scripted:
    def __init__(self, actions):
        self.actions, self.seen = list(actions), []

    def chat(self, messages, model, purpose, **kw):
        self.seen.append(messages[-1]["content"])
        return json.dumps(self.actions.pop(0))


def test_interview_describes_the_photos(tmp_path):
    fields = {"contexts": [{"kind": "facade", "share": 1.0}], "wall_tags": ["brick"]}
    calls = []

    def profiler(folder, classes):
        calls.append((folder, classes))
        return fields, "In 5 of your photos I see: 100% building fronts."
    client = Scripted([{"describe_scene": {"folder": str(tmp_path), "classes": "flower pots"}},
                       {"say": "Mostly building fronts. Should the images look like that?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}), scene_profiler=profiler)
    turn = s.reply("I want to count pots, photos in the folder")
    assert turn.kind == "say" and calls == [(str(tmp_path), "flower pots")]
    assert "building fronts" in client.seen[-1] and '"wall_tags": ["brick"]' in client.seen[-1]


def test_photos_are_analysed_as_soon_as_the_user_names_the_folder(tmp_path):
    (tmp_path / "sub").mkdir()
    calls = []

    def profiler(folder, classes):
        calls.append((folder, classes))
        return ({"contexts": [{"kind": "balcony", "share": 1.0}],
                 "suggest": {"size_m": {"measure": "length", "range": [0.2, 0.5]}, "props": ["chair"],
                             "photos": ["a.jpg: two pots on a balcony floor"]}}, "In 16 of your photos I see: balconies.")
    client = Scripted([{"say": "I saw pots on balconies. Which device?"}])
    s = interview.Session(client, "m", interview.CategoryIndex(counts={}), scene_profiler=profiler)
    s.reply(f"I want to count pots, my photos are in {tmp_path} and some in {tmp_path}/sub.")
    assert calls == [(str(tmp_path.resolve()), f"I want to count pots, my photos are in {tmp_path} and some in "
                      f"{tmp_path}/sub.")]  # once: the subfolder is part of it
    note = client.seen[-1]
    assert "two pots on a balcony floor" in note and '"size_m"' in note and "without asking" in note
