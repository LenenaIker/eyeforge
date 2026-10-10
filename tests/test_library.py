import json

from PIL import Image

from sim2chip.library import Library
from sim2chip.workspace import Job


def photos(folder, colors):
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, c in enumerate(colors):
        out.append(folder / f"{i}.jpg")
        Image.new("RGB", (32, 24), c).save(out[-1])
    return out


def job_with(home, name, colors):
    job = Job(home / "jobs" / name)
    photos(job.input / "photos", colors)
    job.root.mkdir(parents=True, exist_ok=True)
    (job.root / "spec.json").write_text(json.dumps({"classes": [{"id": "a"}], "user_data": {"raw": "photos"}}))
    return job


def test_jobs_with_the_same_photos_share_one_dataset(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    a, b = job_with(tmp_path, "a", ["red", "blue"]), job_with(tmp_path, "b", ["red", "blue"])
    lib.sync("u", [a, b])
    (ds,) = lib.datasets("u")
    assert ds["photos"] == 2 and ds["jobs"] == ["a", "b"]
    assert lib.datasets("other") == []


def test_chat_photos_make_a_dataset_the_job_then_uses(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    upload = photos(tmp_path / "uploads", ["red", "blue", "green"])
    chat = lib.add_chat_photos("u", "s1", upload, "train", None)
    job = job_with(tmp_path, "j", ["red", "blue"])  # two of the chat's three photos
    lib.sync("u", [job])
    (ds,) = lib.datasets("u")
    assert ds["id"] == chat and ds["photos"] == 3 and ds["jobs"] == ["j"]


def test_deleted_items_leave_kits_and_do_not_come_back(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "j", ["red"])
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    kit = lib.save_kit("u", None, "K", {"datasets": [ds, "not-mine"]})
    assert lib.kits("u")[0]["datasets"] == [ds]
    assert lib.delete("u", "dataset", ds) == 1
    assert lib.kits("u")[0]["datasets"] == [] and lib.kits("u")[0]["id"] == kit
    assert not list((tmp_path / "library" / "files" / "photos").rglob("*.jpg"))
    (job.root / "spec.json").touch()  # the job changes: it is read again, the dataset stays deleted
    lib.db.execute("DELETE FROM synced")
    lib.sync("u", [job])
    assert lib.datasets("u") == []


def test_a_dataset_goes_into_a_job_by_label(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = Job(tmp_path / "jobs" / "j")
    photos(job.input / "lab" / "cat", ["red"])
    photos(job.input / "lab" / "dog", ["blue"])
    (job.root / "spec.json").write_text(json.dumps({"classes": [], "user_data": {"labeled": "lab"}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]
    assert ds["labels"] == ["cat", "dog"]
    out = tmp_path / "out"
    assert lib.materialize_dataset("u", ds["id"], out, by_label=True) == 2
    assert sorted(p.parent.name for p in out.rglob("*.jpg")) == ["cat", "dog"]
    assert lib.materialize_dataset("intruder", ds["id"], tmp_path / "x", by_label=False) == 0


def test_a_dataset_goes_into_a_job_side_by_side(tmp_path):
    """Only the asked sides of a dataset, and the photos 3D models were made from never as labeled photos."""
    lib = Library(tmp_path, tmp_path / "cache")
    job = Job(tmp_path / "jobs" / "j")
    photos(job.input / "lab" / "cat", ["red"])
    photos(job.input / "tst" / "cat", ["blue"])
    photos(job.input / "obj", ["green"])
    (job.root / "spec.json").write_text(json.dumps({
        "classes": [{"id": "cat", "assets": {"photos": "obj"}}], "user_data": {"labeled": "lab", "test": "tst"}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]
    assert ds["splits"] == {"3d": 1, "test": 1, "train": 1}
    out = tmp_path / "all"
    assert lib.materialize_dataset("u", ds["id"], out, by_label=True) == 3
    assert sorted(p.parent.name for p in out.rglob("*.jpg")) == ["all", "cat", "cat"]
    out = tmp_path / "test"
    assert lib.materialize_dataset("u", ds["id"], out, by_label=True, splits=("test",)) == 1
    out = tmp_path / "train"
    assert lib.materialize_dataset("u", ds["id"], out, by_label=True, splits=("train", "3d")) == 2
    assert not {p.name for p in (tmp_path / "test").rglob("*.jpg")} & {p.name for p in out.rglob("*.jpg")}


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
    job = job_with(tmp_path, "j", ["red"])
    (job.stage_dir("convert_assets")).mkdir(parents=True)
    (job.stage_dir("convert_assets") / "assets.json").write_text(json.dumps(
        [{"uid": oid, "class": "pot", "glb": f"library/{oid}.glb"}]))
    lib.sync("u", [job])
    (obj,) = lib.objects("u")
    assert obj["id"] == oid and obj["jobs"] == ["j"]


def test_a_dataset_profile_is_made_once_and_read_by_the_chat(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    ds = lib.new_dataset("u", "Table", photos(tmp_path / "up", ["red", "blue", "green"]))
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
    assert "dark wooden table" in prof["summary"] and "dark wooden table" in lib.dataset_card("u", ds)["what_its_photos_show"]
    lib.profile_dataset("u", ds, run)  # every photo already described
    assert len(asked) == 3
    assert lib.notes("other", [p for p in lib.dataset_notes("u", ds)], "scene", "x") == {}
    lib.delete("u", "dataset", ds)
    assert lib.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0


def test_corrections_of_one_job_come_back_in_the_next(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "a", ["red", "blue"])
    (job.root / "spec.json").write_text(json.dumps({"task": "count", "classes": [{"id": "botella"}],
                                                    "user_data": {"raw": "photos"}}))
    label = job.stage_dir("label")
    label.mkdir(parents=True)
    (label / "teacher.json").write_text(json.dumps({"phrases": {"botella": ["bottle"]}}))
    (label / "verdicts.json").write_text(json.dumps({
        "0.jpg": {"objects": [{"x": 0.5, "y": 0.5, "class": "botella", "box": [0.4, 0.4, 0.6, 0.6], "score": 0.9}],
                  "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"},
        "1.jpg": {"objects": [], "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"}}))
    (job.input / "corrections.json").write_text(json.dumps({"points": {"1.jpg": [[0.2, 0.3, "botella"]]}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    kinds = sorted((n["kind"], n["source"]) for v in lib.dataset_notes("u", ds).values() for n in v)
    assert kinds == [("boxes", "model"), ("boxes", "model"), ("points", "user")]
    # A new job with the same photos in another folder, its class named differently but the same in English.
    other = photos(tmp_path / "b" / "fotos", ["blue", "red"])
    got = lib.corrections_for("u", other[0].parent, ["bottle_cls"], {"bottle_cls": "bottle"}, [])
    assert got == {"points": {"0.jpg": [[0.2, 0.3, "bottle_cls"]]}}
    # A second class the user never marked: the marks stand for the first class only, the teacher's for the cup.
    got = lib.corrections_for("u", other[0].parent, ["bottle_cls", "cup"], {"bottle_cls": "bottle", "cup": "cup"}, [])
    assert got == {"points": {"0.jpg": [[0.2, 0.3, "bottle_cls"]]}, "checked": {"0.jpg": ["bottle_cls"]}}


def test_several_raw_sources_become_one_folder(tmp_path):
    from sim2chip import spec as spec_mod
    from sim2chip.server import merge_sources
    job = Job(tmp_path / "j")
    photos(job.input / "dataset_x", ["red", "blue"])
    photos(job.input / "photos", ["blue", "green"])
    assert merge_sources(job, ["dataset_x", "photos"]) == "raw"
    assert len(list((job.input / "raw").rglob("*.jpg"))) == 3
    s = spec_mod.from_dict({"name": "t", "classes": [{"id": "a"}], "user_data": {"raw": ["library:aaaaaaaaaaaaaaaa"]}})
    assert s.user_data.raw == "library:aaaaaaaaaaaaaaaa"


def test_photos_added_to_a_job_join_its_dataset(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    upload = photos(tmp_path / "uploads", ["red", "blue"])
    chat = lib.add_chat_photos("u", "s1", upload, "train", None)
    job = job_with(tmp_path, "j", ["red", "blue"])
    lib.sync("u", [job])
    photos(job.input / "photos" / "added", ["green"])  # taken in the live test
    (job.root / "spec.json").touch()
    lib.db.execute("DELETE FROM synced")
    lib.sync("u", [job])
    (ds,) = lib.datasets("u")
    assert ds["id"] == chat and ds["photos"] == 3


def test_forgetting_an_owner_leaves_others_alone(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    mine, theirs = job_with(tmp_path, "m", ["red"]), job_with(tmp_path, "t", ["blue"])
    lib.sync("u", [mine])
    lib.sync("v", [theirs])
    lib.save_kit("u", None, "K", {})
    lib.forget("u", ["m"])
    assert lib.datasets("u") == [] and lib.kits("u") == [] and len(lib.datasets("v")) == 1


def test_deleted_labels_are_no_labeling_any_more(tmp_path):
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "a", ["red", "blue"])
    spec = {"task": "count", "classes": [{"id": "botella"}, {"id": "vaso"}], "user_data": {"raw": "photos"}}
    for f in ("spec.json", "request.json"):
        (job.root / f).write_text(json.dumps(spec))
    label = job.stage_dir("label")
    label.mkdir(parents=True)
    (label / "teacher.json").write_text(json.dumps({"phrases": {"botella": ["bottle"], "vaso": ["glass"]}}))
    (label / "verdicts.json").write_text(json.dumps({
        "0.jpg": {"objects": [], "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"}}))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    assert [(l["classes"], l["phrases"]) for l in lib.labelings(ds)] == [
        (["botella", "vaso"], {"botella": "bottle", "vaso": "glass"})]
    assert lib.forget_labels("u", ds, "boxes", "glass") == 1
    assert [l["classes"] for l in lib.labelings(ds)] == [["botella"]]
    lib.forget_labels("u", ds, "boxes", "bottle")
    assert lib.labelings(ds) == []


def test_deleted_and_renamed_labels_stay_so_when_the_job_syncs_again(tmp_path):
    """forget_labels and rename_labels changed only the notes: the next sync of the job (it changed, or VERSION
    went up) took its labels in again under the old names."""
    import os
    import time
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "a", ["red", "blue"])
    (job.root / "spec.json").write_text(json.dumps({"task": "count", "classes": [{"id": "bin"}, {"id": "cup"}],
                                                    "user_data": {"raw": "photos"}}))
    label = job.stage_dir("label")
    label.mkdir(parents=True)
    (label / "teacher.json").write_text(json.dumps({"phrases": {"bin": ["paper bin"], "cup": ["cup"]}}))
    marks = [{"x": 0.5, "y": 0.5, "class": c, "box": [0.4, 0.4, 0.6, 0.6], "score": 0.9} for c in ("bin", "cup")]
    (label / "verdicts.json").write_text(json.dumps({
        f"{i}.jpg": {"objects": marks, "model": "facebook/sam3", "threshold": 0.5, "assigned": "images"}
        for i in range(2)}))
    (job.input / "corrections.json").write_text(json.dumps({"points": {"0.jpg": [[0.2, 0.3, "bin"]]}}))
    past = time.time() - 60
    for f in (label / "verdicts.json", job.input / "corrections.json"):
        os.utime(f, (past, past))
    lib.sync("u", [job])
    ds = lib.datasets("u")[0]["id"]
    assert lib.label_phrases("u", ds) == {"cup": 2, "paper bin": 2}
    lib.rename_labels("u", ds, "paper bin", "blue bin")
    lib.forget_labels("u", ds, "boxes", "cup")
    lib.db.execute("DELETE FROM synced")  # what a VERSION bump does
    lib.sync("u", [job])
    assert lib.label_phrases("u", ds) == {"blue bin": 2}
    kinds = sorted((n["kind"], n["source"]) for v in lib.dataset_notes("u", ds).values() for n in v)
    assert kinds == [("boxes", "model"), ("boxes", "model"), ("points", "user")]
    # The job labels the photos again after that: its new labels are taken in.
    (label / "verdicts.json").touch()
    lib.sync("u", [job])
    assert lib.label_phrases("u", ds) == {"blue bin": 2, "cup": 2, "paper bin": 2}


def test_photos_sorted_into_labels_do_not_reuse_an_unlabeled_dataset(tmp_path):
    """The same photos, first raw, then in label folders: the labels make a dataset of their own (the raw one
    had none to give), and a photo on two sides counts once."""
    lib = Library(tmp_path, tmp_path / "cache")
    raw = job_with(tmp_path, "raw", ["red", "blue"])
    lib.sync("u", [raw])
    job = Job(tmp_path / "jobs" / "lab")
    photos(job.input / "lab" / "cat", ["red"])
    photos(job.input / "lab" / "dog", ["blue"])
    photos(job.input / "tst" / "cat", ["red"])
    (job.root / "spec.json").write_text(json.dumps({"classes": [], "user_data": {"labeled": "lab", "test": "tst"}}))
    lib.sync("u", [job])
    sets = {d["name"]: d for d in lib.datasets("u")}
    assert sets["lab"]["labels"] == ["cat", "dog"] and sets["raw"]["labels"] == []
    assert sets["lab"]["photos"] == 2 and sets["lab"]["splits"] == {"test": 1, "train": 2}


def test_one_object_under_several_classes_is_not_the_users_truth(tmp_path):
    """A check that showed one bin under every bin phrase saved it as every color at once (2026-10-09): those
    classes count as not checked on that photo, in sync, in a later job and in the library kept before."""
    lib = Library(tmp_path, tmp_path / "cache")
    job = job_with(tmp_path, "a", ["red", "blue"])
    (job.root / "spec.json").write_text(json.dumps({"task": "count", "classes": [{"id": "blue_bin"}, {"id": "gray_bin"},
                                                    {"id": "bag"}], "user_data": {"raw": "photos"}}))
    (job.input / "corrections.json").write_text(json.dumps({"points": {
        "0.jpg": [[0.5, 0.5, "blue_bin"], [0.5002, 0.5001, "gray_bin"], [0.2, 0.8, "bag"]]}}))
    lib.sync("u", [job])
    other = photos(tmp_path / "b" / "fotos", ["red", "blue"])[0].parent
    phrases = {"blue_bin": "blue bin", "gray_bin": "gray bin", "bag": "bag"}
    got = lib.corrections_for("u", other, list(phrases), phrases, [])
    assert got == {"points": {"0.jpg": [[0.2, 0.8, "bag"]]}, "checked": {"0.jpg": ["bag"]}}
    # A library from before: the stacked notes are dropped when it opens.
    lib.db.executemany("INSERT OR REPLACE INTO notes VALUES (?, ?, 'points', ?, 'user', NULL, ?, 0)",
                       [("u", k, p, json.dumps({"points": [[0.5, 0.5]]})) for p in ("blue bin", "gray bin")
                        for k in ["k1"]])
    lib.db.execute("PRAGMA user_version = 2")
    lib = Library(tmp_path, tmp_path / "cache")
    assert lib.db.execute("SELECT COUNT(*) FROM notes WHERE key = 'k1'").fetchone()[0] == 0


def test_kept_boxes_leave_out_one_object_under_several_classes(tmp_path):
    from sim2chip.library import BOXES, content_key
    lib = Library(tmp_path, tmp_path / "cache")
    fs = photos(tmp_path / "raw", ["red", "blue"])
    bin_box = {"box": [0.4, 0.4, 0.6, 0.8], "score": 0.9}
    for p in ("blue bin", "gray bin"):  # one bin under both colors on the first photo
        lib.note("u", content_key(fs[0]), BOXES, p, {"boxes": [bin_box], "threshold": 0.5})
        lib.note("u", content_key(fs[1]), BOXES, p, {"boxes": [], "threshold": 0.5})
    lib.note("u", content_key(fs[0]), BOXES, "bag", {"boxes": [{"box": [0.0, 0.0, 0.1, 0.1], "score": 0.8}]})
    got = lib.kept_boxes("u", fs[0].parent, {"blue_bin": "blue bin", "gray_bin": "gray bin", "bag": "bag"})
    first, second = fs[0].name, fs[1].name
    assert list(got["photos"][first]) == ["bag"] and got["photos"][second] == {"blue_bin": [], "gray_bin": []}
    assert got["thresholds"] == {"blue_bin": 0.5, "gray_bin": 0.5}
