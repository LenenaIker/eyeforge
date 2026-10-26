import json
from pathlib import Path

import pytest
from PIL import Image

from sim2chip import review
from sim2chip.library import KEEP, Library
from sim2chip.workspace import Job


def photos(folder, colors):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, c in enumerate(colors):
        out.append(folder / f"{i}.jpg")
        Image.new("RGB", (32, 24), c).save(out[-1])
    return out


BINS = [{"key": "blue_bin", "phrases": ["blue bin"]}, {"key": "gray_bin", "phrases": ["gray bin"]}]


def dataset(tmp_path, colors=("red", "blue", "green")):
    lib = Library(tmp_path, tmp_path / "cache")
    ds = lib.new_dataset("u", "Street", photos(tmp_path / "up", list(colors)))
    return lib, ds, [p["id"] for p in lib.photos(ds)]


def test_a_dataset_keeps_each_photo_once_and_chat_photos_make_their_own(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    assert lib.add_photos("u", ds, photos(tmp_path / "again", ["red", "white"])) == 1  # red was there
    assert len(lib.photos(ds)) == 4 and lib.add_photos("other", ds, []) == 0
    chat = lib.add_chat_photos("u", "s1", photos(tmp_path / "chat", ["red"]))
    assert chat != ds and lib.chat_dataset("u", "s1") == chat and len(lib.photos(chat)) == 1
    assert {d["id"] for d in lib.datasets("u")} == {ds, chat} and lib.datasets("other") == []


def test_a_labeling_has_classes_marks_and_reviews(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    lid = lib.create_labeling(ds, "Bins", "count", BINS)
    with pytest.raises(ValueError):
        lib.create_labeling(ds, "Twice", "count", [{"key": "a"}, {"key": "a"}])
    lib.review_photo(lid, ids[0], [{"class": "blue_bin", "x": 0.2, "y": 0.3}])
    # The teacher's marks never land on a reviewed photo.
    assert lib.take_teacher_marks(lid, {ids[0]: [{"class": "gray_bin", "x": 0.5, "y": 0.5, "score": 0.9}],
                                        ids[1]: [{"class": "gray_bin", "x": 0.5, "y": 0.5, "score": 0.9}]}, 0.45) == 1
    lab = lib.labeling(lid)
    assert (lab["reviewed"], lab["marked"], lab["threshold"]) == (1, 2, 0.45)
    assert [(c["key"], c["marks"], c["phrases"]) for c in lab["classes"]] == [("blue_bin", 1, ["blue bin"]),
                                                                             ("gray_bin", 1, ["gray bin"])]
    got = {p["id"]: p for p in lib.labeling_photos(lid)}
    assert got[ids[0]]["reviewed"] and got[ids[0]]["marks"][0]["source"] == "user"
    assert not got[ids[2]]["reviewed"] and got[ids[2]]["marks"] == []
    # One object, one class; a class the labeling does not have; classify takes exactly one class.
    with pytest.raises(review.StackedMarks):
        lib.review_photo(lid, ids[1], [{"class": "blue_bin", "x": 0.5, "y": 0.5}, {"class": "gray_bin", "x": 0.5, "y": 0.5}])
    with pytest.raises(ValueError):
        lib.review_photo(lid, ids[1], [{"class": "litter", "x": 0.5, "y": 0.5}])
    cls = lib.create_labeling(ds, "Kind", "classify", [{"key": "bin"}, {"key": "empty"}])
    with pytest.raises(ValueError):
        lib.review_photo(cls, ids[1], [{"class": "bin"}, {"class": "empty"}])
    assert lib.find_labeling(ds, "count", ["gray_bin", "blue_bin"]) == lid and lib.find_labeling(ds, "count", ["x"]) is None


def test_editing_classes_copies_a_labeling_a_model_was_trained_with(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    lid = lib.create_labeling(ds, "Bins", "count", BINS)
    lib.review_photo(lid, ids[0], [{"class": "blue_bin", "x": 0.2, "y": 0.3}])
    gray = next(c["id"] for c in lib.classes(lid) if c["key"] == "gray_bin")
    assert lib.edit_class(lid, gray, key="grey_bin") == lid  # no model yet: changed in place
    assert [c["key"] for c in lib.classes(lid)] == ["blue_bin", "grey_bin"]
    lib.db.execute("INSERT INTO model VALUES ('j1', ?, 1, 1)", (lid,))
    new = lib.add_class(lid, "green_bin")
    assert new != lid and [c["key"] for c in lib.classes(lid)] == ["blue_bin", "grey_bin"]  # the model's stays
    lab = lib.labeling(new)
    # The copy keeps the marks; the new class was looked for nowhere, so no photo counts as reviewed.
    assert lab["copied_from"] == lid and lab["marked"] == 1 and lab["reviewed"] == 0
    assert lib.labeling(lid)["reviewed"] == 1
    # The spec's classes: the same ones use it, others make it fit (a copy when a model uses it).
    assert lib.fit_classes(lid, BINS[:1] + [{"key": "grey_bin"}]) == lid
    fit = lib.fit_classes(lid, BINS[:1])
    assert fit not in (lid, new) and [c["key"] for c in lib.classes(fit)] == ["blue_bin"]
    with pytest.raises(ValueError):
        lib.delete_class(fit, lib.classes(fit)[0]["id"])  # its last class


def test_a_job_takes_a_copy_of_its_labeling_and_gives_back_the_teachers_marks(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    lid = lib.create_labeling(ds, "Bins", "count", BINS)
    lib.review_photo(lid, ids[0], [{"class": "blue_bin", "x": 0.2, "y": 0.3}])
    job = Job(tmp_path / "jobs" / "j")
    folder = lib.export(lid, job.input)
    snap = json.loads((job.input / KEEP).read_text())
    assert folder == f"labeling_{lid}" and sorted(snap["photos"]) == sorted(f"{i}.jpg" for i in ids)
    assert snap["photos"][f"{ids[0]}.jpg"]["reviewed"] and snap["classes"][0]["phrases"] == ["blue bin"]
    assert len(list((job.input / folder).iterdir())) == 3
    label = job.stage_dir("label")
    label.mkdir(parents=True)
    obj = {"x": 0.5, "y": 0.5, "class": "gray_bin", "box": [0.4, 0.4, 0.6, 0.6], "score": 0.8}
    (label / "verdicts.json").write_text(json.dumps({f"{i}.jpg": {"objects": [obj], "assigned": "images"} for i in ids}))
    (label / "teacher.json").write_text(json.dumps({"threshold": 0.4}))
    assert lib.take_teacher(job) == 2  # not on the reviewed photo
    marks = {p["id"]: p["marks"] for p in lib.labeling_photos(lid)}
    assert marks[ids[0]][0]["class"] == "blue_bin" and marks[ids[1]][0]["w"] == 0.2
    assert lib.labeling(lid)["threshold"] == 0.4
    # A finished model says which labeling it was trained with; its classes then stay.
    (job.stage_dir("export")).mkdir(parents=True)
    (job.stage_dir("export") / "report.json").write_text("{}")
    lib.sync("u", [job])
    assert lib.labeling(lid)["models"] == ["j"] and lib.used(lid)


def test_deleted_items_leave_kits_and_take_their_files(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    lid = lib.create_labeling(ds, "Bins", "count", BINS)
    lib.review_photo(lid, ids[0], [{"class": "blue_bin", "x": 0.2, "y": 0.3}])
    kit = lib.save_kit("u", None, "K", {"datasets": [ds, "not-mine"]})
    assert lib.kits("u")[0]["datasets"] == [ds]
    assert lib.delete("u", "dataset", ds) == 1
    assert lib.kits("u")[0]["datasets"] == [] and lib.kits("u")[0]["id"] == kit
    assert not list((tmp_path / "library" / "files" / "photos").rglob("*.jpg"))
    for table in ("photo", "labeling", "class", "class_name", "mark", "review"):
        assert lib.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_the_chat_may_only_use_listed_library_ids():
    from types import SimpleNamespace

    from sim2chip import spec as spec_mod
    from sim2chip.interview import _library_refs
    data = {"name": "t", "classes": [{"id": "pot", "size_m": {"measure": "length", "range": [0.1, 0.3]},
                                      "assets": {"source": "library", "models": ["aaaaaaaaaaaaaaaa"]}}],
            "user_data": {"raw": "library:bbbbbbbbbbbbbbbb"}}
    s = spec_mod.from_dict(data)
    assert _library_refs(SimpleNamespace(library_ids={"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"}), s) is None
    assert "bbbbbbbbbbbbbbbb" in _library_refs(SimpleNamespace(library_ids={"aaaaaaaaaaaaaaaa"}), s)


def test_a_job_using_library_objects_does_not_copy_them(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    glb = tmp_path / "m.glb"
    glb.write_bytes(b"glTF fake")
    oid = lib.new_object("u", "Pot", "pot", glb, None)
    job = Job(tmp_path / "jobs" / "j")
    (job.stage_dir("convert_assets")).mkdir(parents=True)
    (job.stage_dir("convert_assets") / "assets.json").write_text(json.dumps(
        [{"uid": oid, "class": "pot", "glb": f"library/{oid}.glb"}]))
    lib.sync("u", [job])
    (obj,) = lib.objects("u")
    assert obj["id"] == oid and obj["jobs"] == ["j"]


def test_internet_models_stay_out_of_the_owners_objects(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = Job(tmp_path / "jobs" / "j")
    (job.stage_dir("convert_assets")).mkdir(parents=True)
    (job.stage_dir("convert_assets") / "assets.json").write_text(json.dumps(
        [{"uid": "objaverse1", "class": "bin"}, {"uid": "built1", "class": "bin", "generated": True}]))
    lib.sync("u", [job])
    (obj,) = lib.objects("u")
    assert obj["source"] == "photos" and obj["concept"] == "bin"
    # A library from before keeps its internet models as objects: they go when it opens again.
    lib.db.execute("INSERT INTO objects VALUES ('old1', 'u', 'uid:x', 'Bin', 'bin', 'online', NULL, NULL, 1, '{}', 0, NULL)")
    lib.save_kit("u", None, "Street", {"objects": ["old1", obj["id"]]})
    lib.db.execute("PRAGMA user_version = 4")
    lib = Library(tmp_path, tmp_path / "cache")
    assert [o["id"] for o in lib.objects("u")] == [obj["id"]]
    assert lib.kits("u")[0]["objects"] == [obj["id"]]


def bundle(folder: Path) -> Path:
    """A scene job's bundle (wsroute/bundle.py), small."""
    folder.mkdir(parents=True)
    for name in ("splat.ply", "splat.usdz", "free.npz"):
        (folder / name).write_bytes(b"x")
    Image.new("RGB", (64, 40)).save(folder / "sky.jpg")
    Image.new("RGB", (64, 40)).save(folder / "card.jpg")
    (folder / "scene.json").write_text(json.dumps({"route": "walk", "radius": 10.0, "free_m2": 171.0}))
    return folder


def test_a_walk_scene_goes_from_building_to_ready_and_to_the_jobs_that_name_it(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    sid = lib.start_scene("u", ids[0])
    (scene,) = lib.scenes("u")
    assert scene["id"] == sid and scene["status"] == "building" and scene["dataset"] == ds and scene["mine"]
    assert lib.start_scene("u", ids[0]) == sid and len(lib.scenes("u")) == 1  # one scene per photo
    lib.add_scene(sid, bundle(tmp_path / "b0"))
    (scene,) = Library(tmp_path, tmp_path / "cache").scenes("u")  # it outlives a server restart
    assert scene["status"] == "ready" and scene["thumb"] and scene["free_m2"] == 171.0 and not scene["public"]
    # A training gets only the files it reads: no USDZ.
    given = lib.export_scenes("u", [sid, "nope"], tmp_path / "jobs" / "k" / "input")
    assert given == [sid]
    files = {f.name for f in (tmp_path / "jobs" / "k" / "input" / "scenes" / sid).iterdir()}
    assert files == {"scene.json", "splat.ply", "sky.jpg", "free.npz"}
    # Another owner sees it only once it is public, and may use it then.
    assert lib.scenes("v") == [] and lib.export_scenes("v", [sid], tmp_path / "jobs" / "m" / "input") == []
    lib.set_public(sid, True)
    (other,) = lib.scenes("v")
    assert other["public"] and not other["mine"] and other["dataset"] is None
    assert lib.export_scenes("v", [sid], tmp_path / "jobs" / "m" / "input") == [sid]
    assert lib.card_file("v", "scene", sid) is not None
    # A failed one says why; deleting is the owner's.
    sid2 = lib.start_scene("u", ids[1])
    lib.fail_scene(sid2, "no focal length")
    assert lib.scene_status("u", sid2) == {"status": "failed", "why": "no focal length"}
    lib.delete("u", "scene", sid)
    assert [s["id"] for s in lib.scenes("u")] == [sid2] and lib.scenes("v") == []


def test_a_bundle_made_apart_is_imported_as_a_ready_scene(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    sid = lib.import_scene("admin", bundle(tmp_path / "f3c_photo21" / "bundle"), "Road by the fence")
    (scene,) = lib.scenes("admin")
    assert scene["id"] == sid and scene["status"] == "ready" and scene["name"] == "Road by the fence"
    assert lib.import_scene("admin", tmp_path / "f3c_photo21" / "bundle", "again") == sid


def test_a_dataset_profile_is_made_once_and_read_by_the_chat(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    asked = []

    def run(files):
        asked.extend(files)
        return [{"photo": f.name, "context": "indoor", "distance_m": 0.8, "light": "indoor", "floor": "wood",
                 "objects": [{"name": "Bottle", "count": 4}, {"name": "shoe", "count": 1}],
                 "background": "dark wooden table"} for f in files]
    assert lib.profile_stale("u", ds)
    lib.profile_dataset("u", ds, run)
    assert len(asked) == 3 and not lib.profile_stale("u", ds)
    prof = lib.datasets("u")[0]["profile"]
    assert prof["objects"][0] == {"name": "bottle", "share": 1.0, "count": 4}
    card = lib.dataset_card("u", ds)
    assert "dark wooden table" in prof["summary"] and "dark wooden table" in card["what_its_photos_show"]
    lib.profile_dataset("u", ds, run)  # every photo already described
    assert len(asked) == 3
    lib.delete("u", "dataset", ds)
    assert lib.db.execute("SELECT COUNT(*) FROM profile").fetchone()[0] == 0


def test_the_chat_sees_a_datasets_labelings(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    lid = lib.create_labeling(ds, "Bins", "count", BINS)
    lib.review_photo(lid, ids[0], [])
    (card,) = lib.dataset_card("u", ds)["labelings"]
    assert card["id"] == lid and card["photos_reviewed_by_user"] == 1
    assert card["classes"][0] == {"key": "blue_bin", "names": ["blue bin"], "description": ""}
    assert "1 labelings" in lib.for_chat("u")[0] and lib.dataset_card("other", ds) is None


def test_forgetting_an_owner_leaves_others_alone(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    lib.new_dataset("u", "Mine", photos(tmp_path / "a", ["red"]))
    lib.new_dataset("v", "Theirs", photos(tmp_path / "b", ["blue"]))
    lib.save_kit("u", None, "K", {})
    lib.forget("u", ["m"])
    assert lib.datasets("u") == [] and lib.kits("u") == [] and len(lib.datasets("v")) == 1


def test_a_library_from_before_labelings_starts_empty(tmp_path):
    import sqlite3
    (tmp_path / "library").mkdir()
    old = sqlite3.connect(tmp_path / "library" / "library.db")
    old.execute("CREATE TABLE notes (owner TEXT)")
    old.execute("PRAGMA user_version = 3")
    old.commit()
    old.close()
    lib = Library(tmp_path, tmp_path / "cache")
    assert lib.db.execute("SELECT name FROM sqlite_master WHERE name = 'notes'").fetchone() is None
    assert lib.datasets("u") == []
