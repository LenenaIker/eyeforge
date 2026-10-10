import json

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


def test_a_scene_built_by_a_job_goes_to_the_next_job_on_its_dataset(tmp_path):
    lib, ds, ids = dataset(tmp_path)
    job = Job(tmp_path / "jobs" / "j")
    stage = job.stage_dir("scene_from_photo")
    (stage / "s0" / "assets" / "sam3d").mkdir(parents=True)
    (stage / "s0" / "background.png").write_bytes(b"png")
    (stage / "s0" / "assets" / "sam3d" / "000_canon.glb").write_bytes(b"glTF")
    (stage / "s0" / "logs").mkdir()
    (stage / "s0" / "logs" / "x.log").write_text("noise")
    (stage / "scenes.json").write_text(json.dumps([
        {"photo_id": ids[0], "route": "support", "status": "ready", "dir": "s0", "name": "Kitchen table",
         "objects": [{"index": 0, "name": "water bottle", "concept": "bottle", "role": "target",
                      "mesh": "assets/sam3d/000_canon.glb"}]},
        {"photo_id": ids[1], "route": "street", "status": "failed", "why": "no ground"}]))
    lib.sync("u", [job])
    (scene,) = lib.scenes("u")
    assert scene["name"] == "Kitchen table" and scene["dataset"] == ds and scene["jobs"] == ["j"]
    (obj,) = lib.objects("u")
    assert obj["source"] == "scene" and obj["scene"] == scene["id"] and scene["objects"] == [obj["id"]]
    nxt = tmp_path / "jobs" / "k" / "input"
    lid = lib.create_labeling(ds, "Bottles", "count", [{"key": "bottle"}])
    lib.export(lid, nxt)
    given = json.loads((nxt / "scenes.json").read_text())
    assert given["failed_photos"] == [ids[1]] and given["ready"][0]["objects"][0]["concept"] == "bottle"
    assert (nxt / "scenes" / scene["id"] / "background.png").exists()
    assert not (nxt / "scenes" / scene["id"] / "logs").exists()
    # Deleting the scene keeps the objects taken out of it.
    lib.delete("u", "scene", scene["id"])
    assert lib.scenes("u") == [] and lib.objects("u")[0]["scene"] is None


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
