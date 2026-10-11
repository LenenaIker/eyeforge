"""The user's library: datasets of real photos and how they are labeled, 3D objects, scenes and kits, kept for
good (user, 2026-10-08; labelings redesigned 2026-10-10, docs/plan_etiquetado.md).

- A dataset is a set of photos (uploaded on the Datasets page or in a chat). A photo belongs to one dataset.
- A labeling is a list of classes looked for in one dataset's photos, with their marks: "Blue bins by color",
  count or classify. A dataset may have several. Each mark is one class on one photo (a point with the box
  size when known, or the whole photo when classifying), from the teacher or from the user. A photo the user
  reviewed in a labeling has the user's marks as its truth; the teacher never writes on it again.
- One object, one class: marks of two classes on the same spot of a photo are refused (review.stacked).
- The classes of a labeling a model was trained with do not change under it: editing them makes a copy.
- A job gets a fixed copy of its labeling (export: input/labeling.json and the photos) and gives back the
  teacher's marks (take_teacher). Nothing else travels between jobs and the library.
- The owner's 3D objects: uploaded, or built by a job from the owner's photos, with a picture for its card.
  Models found on the internet are our catalog for every user (asset_db.py), never an owner's object.
- A scene is a place rebuilt from one of the owner's photos (photo2scene, docs/plan_photo2scene.md), built once by
  a job (take_scenes) and given to the next jobs on that dataset (export_scenes). The things taken out of the
  photo become the owner's objects, tied to their scene.
- A kit has no files: a named list of references to datasets, objects and scenes.
- What a vision model said about a photo's scene is kept by photo content (photo_description), and each dataset
  has a profile (what objects show up, the backgrounds, the places), made before anyone says what to look for.

Files are stored once by content under <home>/library/files; the SQLite file <home>/library/library.db says who
owns what. Nothing here runs a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from .review import StackedMarks, stacked
from .workspace import IMAGE_EXT, Job, content_key

VERSION = 5  # 4: datasets, labelings, classes, marks and reviews (2026-10-10); older libraries were emptied
# 5: models found on the internet left the owners' objects (2026-10-11): they are our catalog (asset_db.py)
# 6: scenes built from a photo (photo2scene, docs/plan_photo2scene.md) and the objects taken out of them
THUMB = 512  # side of the picture kept for an object's or a scene's card
TASKS = ("count", "classify")
# Colors of the classes' marks, by position: distinct on photos and in both themes.
COLORS = ["#2f6db5", "#d4a017", "#8a5a2b", "#7a7f87", "#3f8a3a", "#a83e72", "#d2691e", "#2d8aa0", "#7a4f9a",
          "#b5482f"]
KEEP = "labeling.json"  # in a job's input: the fixed copy of its labeling (export)

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photo (
    id INTEGER PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL, file TEXT NOT NULL, added REAL NOT NULL, UNIQUE (dataset_id, content_hash));
CREATE TABLE IF NOT EXISTS labeling (
    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES dataset(id) ON DELETE CASCADE, name TEXT NOT NULL,
    task TEXT NOT NULL CHECK (task IN ('count', 'classify')), threshold REAL,
    copied_from TEXT REFERENCES labeling(id) ON DELETE SET NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS class (
    id INTEGER PRIMARY KEY, labeling_id TEXT NOT NULL REFERENCES labeling(id) ON DELETE CASCADE,
    key TEXT NOT NULL, description TEXT NOT NULL, color TEXT NOT NULL, position INTEGER NOT NULL,
    UNIQUE (labeling_id, key));
CREATE TABLE IF NOT EXISTS class_name (
    id INTEGER PRIMARY KEY, class_id INTEGER NOT NULL REFERENCES class(id) ON DELETE CASCADE,
    phrase TEXT NOT NULL, rank INTEGER NOT NULL, UNIQUE (class_id, phrase));
CREATE TABLE IF NOT EXISTS review (
    labeling_id TEXT NOT NULL REFERENCES labeling(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE, reviewed_at REAL NOT NULL,
    PRIMARY KEY (labeling_id, photo_id));
CREATE TABLE IF NOT EXISTS mark (
    id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL REFERENCES class(id) ON DELETE CASCADE,
    x REAL, y REAL, w REAL, h REAL, score REAL, source TEXT NOT NULL CHECK (source IN ('teacher', 'user')));
CREATE INDEX IF NOT EXISTS mark_photo ON mark (photo_id);
CREATE INDEX IF NOT EXISTS mark_class ON mark (class_id);
CREATE TABLE IF NOT EXISTS model (
    job TEXT PRIMARY KEY, labeling_id TEXT REFERENCES labeling(id) ON DELETE SET NULL, trained_at REAL NOT NULL,
    reviewed_photos INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS profile (
    dataset_id TEXT PRIMARY KEY REFERENCES dataset(id) ON DELETE CASCADE, data TEXT NOT NULL, summary TEXT NOT NULL,
    photos INTEGER NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS photo_description (
    owner TEXT NOT NULL, content_hash TEXT NOT NULL, classes TEXT NOT NULL, model TEXT, data TEXT NOT NULL,
    created REAL NOT NULL, PRIMARY KEY (owner, content_hash, classes));
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, photo_id INTEGER REFERENCES photo(id) ON DELETE SET NULL,
    name TEXT NOT NULL, route TEXT NOT NULL, status TEXT NOT NULL, why TEXT, dir TEXT, thumb TEXT,
    info TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, origin TEXT NOT NULL, name TEXT NOT NULL, concept TEXT NOT NULL,
    source TEXT NOT NULL, file TEXT, thumb TEXT, kept INTEGER NOT NULL, info TEXT NOT NULL, created REAL NOT NULL,
    scene_id TEXT REFERENCES scenes(id) ON DELETE SET NULL, UNIQUE (owner, origin));
CREATE TABLE IF NOT EXISTS kits (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, refs TEXT NOT NULL, created REAL NOT NULL,
    updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS uses (
    job TEXT NOT NULL, kind TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY (job, kind, item));
CREATE TABLE IF NOT EXISTS removed (
    owner TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY (owner, item));
CREATE TABLE IF NOT EXISTS synced (
    job TEXT PRIMARY KEY, stamp REAL NOT NULL);
"""
OLD_TABLES = ("datasets", "photos", "notes", "label_edits", "profiles", "uses", "synced")  # before VERSION 4
TASK_NAME = {"count": "counts objects", "roi_classification": "one label per photo", "classify": "one label per photo"}
TITLE = "title.txt"  # in a job folder: the name the user gave the model
SCENES = "scenes.json"  # in a job's input: the ready scenes of its dataset; in scene_from_photo: the ones it built
# A scene's working files nobody reads again, and one job's layout (photo2scene.COMPOSED).
SCENE_SKIP = ("logs", "isaac", "report.json", "sheet.jpg", "scene.glb", "sheet_isaac.jpg", "review.jpg", "review.json")


def _id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _images(folder: Path) -> list[Path]:
    """Image files under folder, following links (a job's input photos may be links to the upload)."""
    out = []
    for root, _, files in os.walk(folder, followlinks=True):
        out += [Path(root) / f for f in files if Path(f).suffix.lower() in IMAGE_EXT and not f.startswith(".")]
    return sorted(out)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _store(src: Path, dst: Path) -> None:
    """A file into the library: a hard link when it can (same disk, no extra space), else a copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _picture(src: Path, dst: Path) -> None:
    """The card picture of an object or a scene: a JPEG of at most THUMB pixels a side."""
    from PIL import Image, ImageOps
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    im.save(dst, quality=88)


def task_of(spec_task: str | None) -> str:
    """The labeling task of a spec task: count, or classify (one label per photo)."""
    return "count" if spec_task == "count" else "classify"


def stage_signature(job: Job) -> float:
    """When the job last changed what the library takes from it (its spec, the asset stages, the teacher's marks
    and the model)."""
    paths = [job.spec_path, job.root / "request.json", job.stage_dir("convert_assets") / "assets.json",
             job.stage_dir("qa_assets") / "assets.json", job.stage_dir("label") / "verdicts.json",
             job.stage_dir("scene_from_photo") / "scenes.json", job.stage_dir("export") / "report.json"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


class Library:
    def __init__(self, home: Path, cache_root: Path):
        self.root = home / "library"
        self.files = self.root / "files"
        self.cache_root = cache_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "library.db"), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys = ON")
        old = self.db.execute("PRAGMA user_version").fetchone()[0]
        if old < 4:
            # Labels were notes per English phrase until 2026-10-10; the user's data was emptied for the new model.
            for t in OLD_TABLES:
                self.db.execute(f"DROP TABLE IF EXISTS {t}")
        if 0 < old < 6:
            # No job made scenes before photo2scene: the old table was always empty.
            self.db.execute("DROP TABLE IF EXISTS scenes")
            cols = [c[1] for c in self.db.execute("PRAGMA table_info(objects)")]
            if cols and "scene_id" not in cols:
                self.db.execute("ALTER TABLE objects ADD COLUMN scene_id TEXT REFERENCES scenes(id) ON DELETE SET NULL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        if old < 5:
            self._drop_online()
        self.db.execute(f"PRAGMA user_version = {VERSION}")

    # Datasets and photos -----------------------------------------------------------------------------

    def owns(self, owner: str, kind: str, item: str) -> bool:
        table = {"dataset": "dataset", "object": "objects", "scene": "scenes", "kit": "kits"}[kind]
        return self.db.execute(f"SELECT 1 FROM {table} WHERE owner = ? AND id = ?", (owner, item)).fetchone() is not None

    def owns_labeling(self, owner: str, lid: str) -> bool:
        return self.db.execute("SELECT 1 FROM labeling l JOIN dataset d ON d.id = l.dataset_id WHERE d.owner = ? "
                               "AND l.id = ?", (owner, lid)).fetchone() is not None

    def _put_photos(self, ds: str, files: list[Path]) -> int:
        """Photos into a dataset, once each by content. Returns how many were new."""
        n = 0
        for f in files:
            key = content_key(f)
            rel = f"photos/{key[:2]}/{key}{f.suffix.lower()}"
            _store(f, self.files / rel)
            n += self.db.execute("INSERT OR IGNORE INTO photo (dataset_id, content_hash, file, added) VALUES (?, ?, ?, ?)",
                                 (ds, key, rel, time.time())).rowcount
        return n

    def new_dataset(self, owner: str, name: str, files: list[Path]) -> str:
        """Photos uploaded on the Datasets page: a dataset of their own."""
        ds = _id(owner, "upload", str(time.time_ns()))
        with self.lock:
            self.db.execute("INSERT INTO dataset VALUES (?, ?, ?, ?)", (ds, owner, name, time.time()))
            self._put_photos(ds, files)
        return ds

    def add_chat_photos(self, owner: str, session: str, files: list[Path]) -> str:
        """Photos attached in a chat go to that chat's dataset, made with the first ones."""
        ds = _id(owner, "chat", session)
        with self.lock:
            self.db.execute("DELETE FROM removed WHERE owner = ? AND item = ?", (owner, ds))
            self.db.execute("INSERT OR IGNORE INTO dataset VALUES (?, ?, ?, ?)",
                            (ds, owner, time.strftime("Chat photos %d %b %Y %H:%M"), time.time()))
            self._put_photos(ds, files)
        return ds

    def chat_dataset(self, owner: str, session: str) -> str | None:
        ds = _id(owner, "chat", session)
        return ds if self.owns(owner, "dataset", ds) else None

    def add_photos(self, owner: str, ds: str, files: list[Path]) -> int:
        if not self.owns(owner, "dataset", ds):
            return 0
        with self.lock:
            return self._put_photos(ds, files)

    def photo_id(self, ds: str, content_hash: str) -> int | None:
        row = self.db.execute("SELECT id FROM photo WHERE dataset_id = ? AND content_hash = ?", (ds, content_hash)).fetchone()
        return row[0] if row else None

    def photos(self, ds: str) -> list[dict]:
        return [{"id": pid, "hash": h, "file": f} for pid, h, f in self.db.execute(
            "SELECT id, content_hash, file FROM photo WHERE dataset_id = ? ORDER BY id", (ds,))]

    def photo_file(self, owner: str, photo_id: int) -> Path | None:
        row = self.db.execute("SELECT p.file FROM photo p JOIN dataset d ON d.id = p.dataset_id "
                              "WHERE d.owner = ? AND p.id = ?", (owner, photo_id)).fetchone()
        return self.files / row[0] if row else None

    def delete_photo(self, owner: str, photo_id: int) -> bool:
        with self.lock:
            row = self.db.execute("SELECT p.file FROM photo p JOIN dataset d ON d.id = p.dataset_id "
                                  "WHERE d.owner = ? AND p.id = ?", (owner, photo_id)).fetchone()
            if row is None:
                return False
            self.db.execute("DELETE FROM photo WHERE id = ?", (photo_id,))
            self._drop_files([row[0]])
        return True

    def _drop_files(self, files: list[str]) -> None:
        for f in files:
            if not self.db.execute("SELECT 1 FROM photo WHERE file = ?", (f,)).fetchone():
                (self.files / f).unlink(missing_ok=True)

    def datasets(self, owner: str) -> list[dict]:
        out = []
        for ds, name, created in self.db.execute(
                "SELECT id, name, created FROM dataset WHERE owner = ? ORDER BY created DESC", (owner,)).fetchall():
            n = self.db.execute("SELECT COUNT(*) FROM photo WHERE dataset_id = ?", (ds,)).fetchone()[0]
            sample = [pid for (pid,) in self.db.execute("SELECT id FROM photo WHERE dataset_id = ? ORDER BY id LIMIT 4",
                                                        (ds,))]
            out.append({"id": ds, "name": name, "created": created, "photos": n, "sample": sample,
                        "labelings": self.labelings(ds), "jobs": self._used("dataset", ds),
                        "profile": self.profile(owner, ds)})
        return out

    # Labelings -------------------------------------------------------------------------------------------

    def create_labeling(self, ds: str, name: str, task: str, classes: list[dict], copied_from: str | None = None) -> str:
        """A labeling of a dataset: classes [{key, description, phrases}] in order (keys unique)."""
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        keys = [c["key"] for c in classes]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("a labeling needs classes with different names")
        lid = _id(ds, "labeling", str(time.time_ns()))
        with self.lock:
            self.db.execute("INSERT INTO labeling VALUES (?, ?, ?, ?, NULL, ?, ?)",
                            (lid, ds, name, task, copied_from, time.time()))
            for c in classes:
                self._add_class(lid, c)
        return lid

    def _add_class(self, lid: str, c: dict) -> int:
        pos = self.db.execute("SELECT COALESCE(MAX(position) + 1, 0) FROM class WHERE labeling_id = ?",
                              (lid,)).fetchone()[0]
        cid = self.db.execute("INSERT INTO class (labeling_id, key, description, color, position) VALUES (?, ?, ?, ?, ?)",
                              (lid, c["key"], c.get("description") or "", COLORS[pos % len(COLORS)], pos)).lastrowid
        self._set_phrases(cid, c.get("phrases") or [c["key"].replace("_", " ")])
        return cid

    def _set_phrases(self, cid: int, phrases: list[str]) -> None:
        self.db.execute("DELETE FROM class_name WHERE class_id = ?", (cid,))
        for i, p in enumerate(dict.fromkeys(p.strip().lower() for p in phrases if p and p.strip())):
            self.db.execute("INSERT INTO class_name (class_id, phrase, rank) VALUES (?, ?, ?)", (cid, p, i))

    def classes(self, lid: str) -> list[dict]:
        out = []
        for cid, key, desc, color in self.db.execute(
                "SELECT id, key, description, color FROM class WHERE labeling_id = ? ORDER BY position", (lid,)).fetchall():
            phrases = [p for (p,) in self.db.execute("SELECT phrase FROM class_name WHERE class_id = ? ORDER BY rank",
                                                     (cid,))]
            out.append({"id": cid, "key": key, "description": desc, "color": color, "phrases": phrases})
        return out

    def labelings(self, ds: str) -> list[dict]:
        """The dataset's labelings, newest first, with their classes, how many photos the user reviewed, how many
        have marks, the marks per class and the models trained with each."""
        out = []
        total = self.db.execute("SELECT COUNT(*) FROM photo WHERE dataset_id = ?", (ds,)).fetchone()[0]
        for lid, name, task, threshold, copied, created in self.db.execute(
                "SELECT id, name, task, threshold, copied_from, created FROM labeling WHERE dataset_id = ? "
                "ORDER BY created DESC", (ds,)).fetchall():
            classes = self.classes(lid)
            counts = dict(self.db.execute("SELECT class_id, COUNT(*) FROM mark WHERE class_id IN "
                                          "(SELECT id FROM class WHERE labeling_id = ?) GROUP BY class_id", (lid,)))
            for c in classes:
                c["marks"] = counts.get(c["id"], 0)
            reviewed = self.db.execute("SELECT COUNT(*) FROM review WHERE labeling_id = ?", (lid,)).fetchone()[0]
            marked = self.db.execute("SELECT COUNT(DISTINCT photo_id) FROM mark WHERE class_id IN "
                                     "(SELECT id FROM class WHERE labeling_id = ?)", (lid,)).fetchone()[0]
            models = [j for (j,) in self.db.execute("SELECT job FROM model WHERE labeling_id = ? ORDER BY trained_at",
                                                    (lid,))]
            out.append({"id": lid, "dataset": ds, "name": name, "task": task, "threshold": threshold,
                        "copied_from": copied, "created": created, "classes": classes, "photos": total,
                        "reviewed": reviewed, "marked": marked, "models": models})
        return out

    def labeling(self, lid: str) -> dict | None:
        row = self.db.execute("SELECT dataset_id FROM labeling WHERE id = ?", (lid,)).fetchone()
        if row is None:
            return None
        return next(l for l in self.labelings(row[0]) if l["id"] == lid)

    def labeling_photos(self, lid: str) -> list[dict]:
        """Every photo of the labeling's dataset with its marks for the labeling's classes and whether the user
        reviewed it: [{id, hash, file, reviewed, marks: [{class, x, y, w, h, score, source}]}]."""
        ds = self.db.execute("SELECT dataset_id FROM labeling WHERE id = ?", (lid,)).fetchone()
        if ds is None:
            return []
        reviewed = {p for (p,) in self.db.execute("SELECT photo_id FROM review WHERE labeling_id = ?", (lid,))}
        marks: dict[int, list] = {}
        for pid, key, x, y, w, h, score, source in self.db.execute(
                "SELECT m.photo_id, c.key, m.x, m.y, m.w, m.h, m.score, m.source FROM mark m JOIN class c ON c.id = "
                "m.class_id WHERE c.labeling_id = ? ORDER BY m.id", (lid,)):
            marks.setdefault(pid, []).append({"class": key, "x": x, "y": y, "w": w, "h": h, "score": score,
                                              "source": source})
        return [{**p, "reviewed": p["id"] in reviewed, "marks": marks.get(p["id"], [])} for p in self.photos(ds[0])]

    def find_labeling(self, ds: str, task: str, keys: list[str]) -> str | None:
        """The newest labeling of the dataset for this task with exactly these classes."""
        for l in self.labelings(ds):
            if l["task"] == task and sorted(c["key"] for c in l["classes"]) == sorted(keys):
                return l["id"]
        return None

    def used(self, lid: str) -> bool:
        """A model was trained with it: its classes stay as they were (edits make a copy)."""
        return self.db.execute("SELECT 1 FROM model WHERE labeling_id = ?", (lid,)).fetchone() is not None

    def rename_labeling(self, lid: str, name: str) -> None:
        with self.lock:
            self.db.execute("UPDATE labeling SET name = ? WHERE id = ?", (name, lid))

    def delete_labeling(self, lid: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM labeling WHERE id = ?", (lid,))

    def copy_labeling(self, lid: str, name: str | None = None) -> str:
        """A new labeling with the same classes, marks and reviews, to change without touching this one."""
        src = self.labeling(lid)
        new = self.create_labeling(src["dataset"], name or f"{src['name']} (2)", src["task"],
                                   [{"key": c["key"], "description": c["description"], "phrases": c["phrases"]}
                                    for c in src["classes"]], copied_from=lid)
        with self.lock:
            ids = {c["key"]: c["id"] for c in self.classes(new)}
            self.db.execute("UPDATE labeling SET threshold = ? WHERE id = ?", (src["threshold"], new))
            for c in src["classes"]:
                self.db.execute("INSERT INTO mark (photo_id, class_id, x, y, w, h, score, source) SELECT photo_id, ?, x, y, "
                                "w, h, score, source FROM mark WHERE class_id = ?", (ids[c["key"]], c["id"]))
            self.db.execute("INSERT INTO review SELECT ?, photo_id, reviewed_at FROM review WHERE labeling_id = ?",
                            (new, lid))
        return new

    def _editable(self, lid: str) -> str:
        """The labeling itself, or a copy when a model was trained with it."""
        return self.copy_labeling(lid) if self.used(lid) else lid

    def add_class(self, lid: str, key: str, description: str = "", phrases: list[str] | None = None) -> str:
        """A new class: the photos the user reviewed are reviewed again (nobody looked for it there). Returns the
        labeling that has it (a copy when a model uses this one)."""
        lid = self._editable(lid)
        with self.lock:
            if self.db.execute("SELECT 1 FROM class WHERE labeling_id = ? AND key = ?", (lid, key)).fetchone():
                raise ValueError(f"the labeling has a class {key} already")
            self._add_class(lid, {"key": key, "description": description, "phrases": phrases})
            self.db.execute("DELETE FROM review WHERE labeling_id = ?", (lid,))
        return lid

    def edit_class(self, lid: str, cid: int, key: str | None = None, description: str | None = None,
                   phrases: list[str] | None = None) -> str:
        """Rename a class or change its description or English names: its marks follow it."""
        key_of = self.db.execute("SELECT key FROM class WHERE id = ? AND labeling_id = ?", (cid, lid)).fetchone()
        if key_of is None:
            raise ValueError("no such class in this labeling")
        new = self._editable(lid)
        if new != lid:
            cid = self.db.execute("SELECT id FROM class WHERE labeling_id = ? AND key = ?", (new, key_of[0])).fetchone()[0]
        with self.lock:
            if key and key != key_of[0]:
                if self.db.execute("SELECT 1 FROM class WHERE labeling_id = ? AND key = ?", (new, key)).fetchone():
                    raise ValueError(f"the labeling has a class {key} already")
                self.db.execute("UPDATE class SET key = ? WHERE id = ?", (key, cid))
            if description is not None:
                self.db.execute("UPDATE class SET description = ? WHERE id = ?", (description, cid))
            if phrases is not None:
                self._set_phrases(cid, phrases)
        return new

    def delete_class(self, lid: str, cid: int) -> str:
        key_of = self.db.execute("SELECT key FROM class WHERE id = ? AND labeling_id = ?", (cid, lid)).fetchone()
        if key_of is None:
            raise ValueError("no such class in this labeling")
        if self.db.execute("SELECT COUNT(*) FROM class WHERE labeling_id = ?", (lid,)).fetchone()[0] <= 1:
            raise ValueError("a labeling keeps at least one class: delete the labeling instead")
        new = self._editable(lid)
        with self.lock:
            self.db.execute("DELETE FROM class WHERE labeling_id = ? AND key = ?", (new, key_of[0]))
        return new

    def fit_classes(self, lid: str, classes: list[dict]) -> str:
        """The labeling to use for these classes [{key, description, phrases}]: this one when its classes are the
        same, else this one changed to them (a copy when a model uses it). Classes it has keep their names and
        marks; new ones are added (the reviewed photos are then reviewed again); missing ones are dropped."""
        have = {c["key"]: c for c in self.classes(lid)}
        want = [c["key"] for c in classes]
        if sorted(have) == sorted(want):
            return lid
        lid = self._editable(lid)
        have = {c["key"]: c for c in self.classes(lid)}
        with self.lock:
            for key, c in have.items():
                if key not in want:
                    self.db.execute("DELETE FROM class WHERE id = ?", (c["id"],))
            added = [c for c in classes if c["key"] not in have]
            for c in added:
                self._add_class(lid, c)
            if added:
                self.db.execute("DELETE FROM review WHERE labeling_id = ?", (lid,))
        return lid

    # Marks -------------------------------------------------------------------------------------------------

    def _class_ids(self, lid: str) -> dict[str, int]:
        return dict(self.db.execute("SELECT key, id FROM class WHERE labeling_id = ?", (lid,)).fetchall())

    def _photo_in(self, lid: str, photo_id: int) -> bool:
        return self.db.execute("SELECT 1 FROM photo p JOIN labeling l ON l.dataset_id = p.dataset_id WHERE l.id = ? "
                               "AND p.id = ?", (lid, photo_id)).fetchone() is not None

    def review_photo(self, lid: str, photo_id: int, marks: list[dict]) -> None:
        """The user's marks of a photo, its truth from now on: [{class, x, y(, w, h)}] for count (empty: nothing
        there), [{class}] (one) for classify. One object, one class."""
        ids = self._class_ids(lid)
        task = self.db.execute("SELECT task FROM labeling WHERE id = ?", (lid,)).fetchone()[0]
        if not self._photo_in(lid, photo_id):
            raise ValueError("that photo is not in this labeling's dataset")
        bad = sorted({m.get("class") for m in marks} - set(ids))
        if bad:
            raise ValueError(f"no class {', '.join(map(str, bad))} in this labeling")
        if task == "classify" and len(marks) != 1:
            raise ValueError("classifying gives each photo exactly one class")
        if task == "count":
            if pile := stacked([[m["x"], m["y"], m["class"]] for m in marks]):
                raise StackedMarks(f"one object is marked as several classes ({', '.join(sorted(pile))}): keep only "
                                   "the class it is")
        with self.lock:
            self.db.execute(f"DELETE FROM mark WHERE photo_id = ? AND class_id IN ({','.join('?' * len(ids))})",
                            (photo_id, *ids.values()))
            for m in marks:
                self.db.execute("INSERT INTO mark (photo_id, class_id, x, y, w, h, score, source) VALUES "
                                "(?, ?, ?, ?, ?, ?, NULL, 'user')",
                                (photo_id, ids[m["class"]], *(round(float(m[k]), 4) if m.get(k) is not None else None
                                                              for k in ("x", "y", "w", "h"))))
            self.db.execute("INSERT OR REPLACE INTO review VALUES (?, ?, ?)", (lid, photo_id, time.time()))

    def unreview_photo(self, lid: str, photo_id: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM review WHERE labeling_id = ? AND photo_id = ?", (lid, photo_id))

    def take_teacher_marks(self, lid: str, by_photo: dict[int, list[dict]], threshold: float | None = None) -> int:
        """The teacher's marks of these photos (replacing its earlier ones), on photos the user has not reviewed.
        Returns how many photos took them."""
        ids = self._class_ids(lid)
        reviewed = {p for (p,) in self.db.execute("SELECT photo_id FROM review WHERE labeling_id = ?", (lid,))}
        n = 0
        with self.lock:
            for pid, marks in by_photo.items():
                if pid in reviewed or not self._photo_in(lid, pid):
                    continue
                self.db.execute(f"DELETE FROM mark WHERE photo_id = ? AND class_id IN ({','.join('?' * len(ids))})",
                                (pid, *ids.values()))
                for m in marks:
                    if m.get("class") in ids:
                        self.db.execute("INSERT INTO mark (photo_id, class_id, x, y, w, h, score, source) VALUES "
                                        "(?, ?, ?, ?, ?, ?, ?, 'teacher')",
                                        (pid, ids[m["class"]], m.get("x"), m.get("y"), m.get("w"), m.get("h"),
                                         m.get("score")))
                n += 1
            if threshold is not None:
                self.db.execute("UPDATE labeling SET threshold = ? WHERE id = ?", (threshold, lid))
        return n

    # Into a job and back -----------------------------------------------------------------------------------

    def export(self, lid: str, job_input: Path) -> str:
        """The labeling as a job takes it: its dataset's photos in job_input/labeling_<id>/ (named by photo id) and
        job_input/labeling.json with its classes and each photo's marks and review. Returns the folder's name."""
        lab = self.labeling(lid)
        folder = f"labeling_{lid}"
        dst = job_input / folder
        dst.mkdir(parents=True, exist_ok=True)
        photos = {}
        for p in self.labeling_photos(lid):
            name = f"{p['id']}{Path(p['file']).suffix}"
            _store(self.files / p["file"], dst / name)
            photos[name] = {"photo_id": p["id"], "hash": p["hash"], "reviewed": p["reviewed"], "marks": p["marks"]}
        keep = {name for name in photos}
        for f in dst.iterdir():  # photos deleted from the dataset since the last export
            if f.name not in keep:
                f.unlink()
        snapshot = {"labeling": lid, "name": lab["name"], "task": lab["task"], "threshold": lab["threshold"],
                    "classes": [{"key": c["key"], "description": c["description"], "phrases": c["phrases"]}
                                for c in lab["classes"]], "photos": photos}
        (job_input / KEEP).write_text(json.dumps(snapshot, indent=1))
        self.export_scenes(lab["dataset"], job_input)
        return folder

    def take_teacher(self, job: Job) -> int:
        """The teacher's marks a job made (label/verdicts.json) into the labeling it was given, for the photos the
        user has not reviewed."""
        snap = _load(job.input / KEEP)
        verdicts = _load(job.stage_dir("label") / "verdicts.json")
        if not snap or not verdicts or self.labeling(snap["labeling"]) is None:
            return 0
        learned = _load(job.stage_dir("label") / "teacher.json") or {}
        by_photo: dict[int, list[dict]] = {}
        for rel, v in verdicts.items():
            p = snap["photos"].get(rel)
            if p is None or p["reviewed"]:
                continue
            if "objects" in v:  # count
                objs = v.get("teacher_objects", v["objects"]) if v.get("judge") == "user" else v["objects"]
                if objs is None:
                    continue
                by_photo[p["photo_id"]] = [
                    {"class": o["class"], "x": o["x"], "y": o["y"],
                     "w": round(o["box"][2] - o["box"][0], 4) if o.get("box") else None,
                     "h": round(o["box"][3] - o["box"][1], 4) if o.get("box") else None, "score": o.get("score")}
                    for o in objs]
            elif v.get("class") and v["class"] != "no_answer":
                cls = v.get("teacher", v["class"]) if v.get("judge") == "user" else v["class"]
                by_photo[p["photo_id"]] = [{"class": cls, "score": v.get("teacher_confidence", v.get("confidence"))}]
        return self.take_teacher_marks(snap["labeling"], by_photo, learned.get("threshold"))

    def take_model(self, job: Job) -> None:
        """A finished model: which labeling it was trained with, and how many of its photos the user reviewed."""
        snap = _load(job.input / KEEP)
        report = job.stage_dir("export") / "report.json"
        if not snap or not report.exists() or self.labeling(snap["labeling"]) is None:
            return
        reviewed = sum(p["reviewed"] for p in snap["photos"].values())
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO model VALUES (?, ?, ?, ?)",
                            (job.name, snap["labeling"], report.stat().st_mtime, reviewed))

    def sync(self, owner: str, jobs: list[Job]) -> None:
        """Take in what new or changed jobs of this owner made: 3D objects, the teacher's marks, the model. Cheap
        when nothing changed."""
        for job in jobs:
            stamp = stage_signature(job)
            row = self.db.execute("SELECT stamp FROM synced WHERE job = ?", (job.name,)).fetchone()
            if row and row[0] >= stamp:
                continue
            self._take_objects(owner, job)
            self.take_scenes(owner, job)
            self.take_teacher(job)
            self.take_model(job)
            with self.lock:
                self.db.execute("INSERT OR REPLACE INTO synced VALUES (?, ?)", (job.name, stamp))

    def note_use(self, job: str, kind: str, item: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, ?, ?)", (job, kind, item))

    # 3D objects --------------------------------------------------------------------------------------------

    def _drop_online(self) -> None:
        """Models found on the internet were kept as every owner's objects until 2026-10-11. They are our catalog
        (the asset DB keeps their verdicts, the cache their files), not the owner's: gone from the library."""
        for oid, owner in self.db.execute("SELECT id, owner FROM objects WHERE source = 'online'").fetchall():
            self.delete(owner, "object", oid)
            self.db.execute("DELETE FROM removed WHERE owner = ? AND item = ?", (owner, oid))

    def _take_objects(self, owner: str, job: Job) -> None:
        """The models a job built from the owner's photos become the owner's objects; a model the job took from
        the library is noted as used. Models found on the internet stay in our catalog (asset_db.py)."""
        converted = _load(job.stage_dir("convert_assets") / "assets.json") or []
        kept = {a["uid"] for a in _load(job.stage_dir("qa_assets") / "assets.json") or []}
        qa_ran = (job.stage_dir("qa_assets") / "assets.json").exists()
        with self.lock:
            for a in converted:
                uid = a["uid"]
                # A model the job took from this library (fetch_assets uses the object's id as uid) is that object.
                mine = self.owns(owner, "object", uid)
                if not mine and not a.get("generated"):
                    continue
                oid = uid if mine else _id(owner, "object", uid)
                if self._removed(owner, oid):
                    continue
                known = self.db.execute("SELECT kept FROM objects WHERE id = ?", (oid,)).fetchone()
                is_kept = int(uid in kept or not qa_ran)
                if known:  # seen in another job: approved once is enough
                    if is_kept and not known[0]:
                        self.db.execute("UPDATE objects SET kept = 1 WHERE id = ?", (oid,))
                else:
                    glb = self.cache_root / a["glb"] if a.get("glb") else None
                    file = None
                    if glb and glb.exists():
                        file = f"objects/{oid}/model.glb"
                        _store(glb, self.files / file)
                    picture = next((p for p in (job.stage_dir("qa_assets") / "sheets" / f"{uid}.jpg",
                                                job.stage_dir("preview_assets") / f"{uid}_0.png",
                                                job.stage_dir("generate_assets") / a["class"] / f"{uid}.png")
                                    if p.exists()), None)
                    thumb = None
                    if picture:
                        thumb = f"objects/{oid}/card.jpg"
                        _picture(picture, self.files / thumb)
                    info = {k: a[k] for k in ("license", "author", "dims", "category", "photo") if a.get(k)}
                    if a.get("source"):
                        info["url"] = a["source"]
                    self.db.execute("INSERT INTO objects VALUES (?, ?, ?, ?, ?, 'photos', ?, ?, ?, ?, ?, NULL)",
                                    (oid, owner, f"uid:{uid}", a.get("name") or a["class"], a["class"], file,
                                     thumb, is_kept, json.dumps(info), time.time()))
                self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'object', ?)", (job.name, oid))

    def _removed(self, owner: str, item: str) -> bool:
        return self.db.execute("SELECT 1 FROM removed WHERE owner = ? AND item = ?", (owner, item)).fetchone() is not None

    def new_object(self, owner: str, name: str, concept: str, glb: Path, card: Path | None) -> str:
        """A 3D model the owner uploads (GLB), with the picture the app rendered of it for its card."""
        oid = _id(owner, "upload", content_key(glb))
        with self.lock:
            if self.db.execute("SELECT 1 FROM objects WHERE id = ?", (oid,)).fetchone():
                return oid
            self.db.execute("DELETE FROM removed WHERE owner = ? AND item = ?", (owner, oid))
            file = f"objects/{oid}/model.glb"
            _store(glb, self.files / file)
            thumb = None
            if card is not None:
                thumb = f"objects/{oid}/card.jpg"
                _picture(card, self.files / thumb)
            self.db.execute("INSERT INTO objects VALUES (?, ?, ?, ?, ?, 'upload', ?, ?, 1, '{}', ?, NULL)",
                            (oid, owner, f"upload:{oid}", name, concept, file, thumb, time.time()))
        return oid

    def materialize_objects(self, owner: str, ids: list[str], dst: Path) -> list[str]:
        """Objects' GLB files into a job folder, with objects.json (name, concept, license, author, page) that
        fetch_assets reads. Returns the file names, in order; objects without a file are skipped."""
        meta_path = dst / "objects.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        names = []
        for oid in ids:
            row = self.db.execute("SELECT name, concept, file, info FROM objects WHERE owner = ? AND id = ?",
                                  (owner, oid)).fetchone()
            if not row or not row[2]:
                continue
            name, concept, file, info = row
            _store(self.files / file, dst / f"{oid}.glb")
            meta[oid] = {"name": name, "concept": concept, **{k: v for k, v in json.loads(info).items()
                                                              if k in ("license", "author", "url")}}
            names.append(f"{oid}.glb")
        dst.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1))
        return names

    # Scenes from a photo (photo2scene) -------------------------------------------------------------------

    def export_scenes(self, ds: str, job_input: Path) -> int:
        """The dataset's ready scenes into a job: job_input/scenes/<id>/ with their files and job_input/scenes.json
        (id, photo, route, folder, objects) for scene_from_photo, which then builds none for those photos."""
        rows = self.db.execute("SELECT s.id, s.photo_id, s.route, s.dir, s.info FROM scenes s JOIN photo p "
                               "ON p.id = s.photo_id WHERE p.dataset_id = ? AND s.status = 'ready' "
                               "ORDER BY s.created", (ds,)).fetchall()
        tried = [pid for (pid,) in self.db.execute("SELECT s.photo_id FROM scenes s JOIN photo p ON p.id = s.photo_id "
                                                   "WHERE p.dataset_id = ? AND s.status = 'failed'", (ds,))]
        out = []
        for sid, pid, route, folder, info in rows:
            src = self.files / folder
            for f in src.rglob("*"):
                if f.is_file():
                    _store(f, job_input / "scenes" / sid / f.relative_to(src))
            objects = [{"id": oid, "name": name, "concept": concept, "file": file}
                       for oid, name, concept, file in self.db.execute(
                           "SELECT id, name, concept, file FROM objects WHERE scene_id = ? AND file IS NOT NULL", (sid,))]
            out.append({"id": sid, "photo_id": pid, "route": route, "dir": f"scenes/{sid}",
                        "info": json.loads(info), "objects": objects})
        if out or tried:
            (job_input / SCENES).write_text(json.dumps({"ready": out, "failed_photos": tried}, indent=1))
        return len(out)

    def take_scenes(self, owner: str, job: Job) -> int:
        """The scenes a job built (scene_from_photo/scenes.json) into the owner's library, with the things taken
        out of each photo as the owner's objects. A failed one is kept too, so the photo is not tried again."""
        stage = job.stage_dir("scene_from_photo")
        built = _load(stage / SCENES) or []
        n = 0
        with self.lock:
            for b in built:
                pid = b.get("photo_id")
                if pid is None or not self.db.execute("SELECT 1 FROM photo p JOIN dataset d ON d.id = p.dataset_id "
                                                      "WHERE p.id = ? AND d.owner = ?", (pid, owner)).fetchone():
                    continue
                sid = _id(owner, "scene", str(pid))
                if self.db.execute("SELECT 1 FROM scenes WHERE id = ?", (sid,)).fetchone() or self._removed(owner, sid):
                    continue
                folder, thumb = None, None
                if b["status"] == "ready":
                    src = stage / b["dir"]
                    folder = f"scenes/{sid}"
                    for f in src.rglob("*"):
                        rel = f.relative_to(src)
                        if f.is_file() and rel.parts[0] not in SCENE_SKIP:
                            _store(f, self.files / folder / rel)
                    if b.get("thumb") and (stage / b["thumb"]).exists():
                        thumb = f"{folder}/card.jpg"
                        _picture(stage / b["thumb"], self.files / thumb)
                self.db.execute("INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (sid, owner, pid, b.get("name") or f"Scene {pid}", b["route"], b["status"],
                                 b.get("why"), folder, thumb, json.dumps(b.get("info", {})), time.time()))
                for o in b.get("objects", []) if folder else []:
                    oid = _id(owner, "scene_object", sid, str(o["index"]))
                    file = f"objects/{oid}/model.glb"
                    _store(stage / b["dir"] / o["mesh"], self.files / file)
                    card = None
                    if o.get("card") and (stage / b["dir"] / o["card"]).exists():
                        card = f"objects/{oid}/card.jpg"
                        _picture(stage / b["dir"] / o["card"], self.files / card)
                    info = {k: o[k] for k in ("height_m", "width_m", "role") if k in o}
                    self.db.execute("INSERT OR IGNORE INTO objects VALUES (?, ?, ?, ?, ?, 'scene', ?, ?, 1, ?, ?, ?)",
                                    (oid, owner, f"scene:{sid}:{o['index']}", o["name"], o.get("concept") or o["name"],
                                     file, card, json.dumps(info), time.time(), sid))
                self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'scene', ?)", (job.name, sid))
                n += 1
            # The library's scenes the job composed again (frames named by scene id).
            for sid in {f["scene"] for f in _load(stage / "frames.json") or []}:
                if self.owns(owner, "scene", sid):
                    self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'scene', ?)", (job.name, sid))
        return n

    # The owner's own changes -------------------------------------------------------------------------------

    def rename(self, owner: str, kind: str, item: str, name: str) -> None:
        table = {"dataset": "dataset", "object": "objects", "scene": "scenes", "kit": "kits"}[kind]
        with self.lock:
            self.db.execute(f"UPDATE {table} SET name = ? WHERE owner = ? AND id = ?", (name, owner, item))

    def delete(self, owner: str, kind: str, item: str) -> int:
        """Delete a dataset (with its photos and labelings), an object or a scene (kept deleted: syncing its jobs
        again does not bring it back) or a kit. The item leaves every kit of the owner; returns how many kits it
        was in."""
        with self.lock:
            if kind == "kit":
                self.db.execute("DELETE FROM kits WHERE owner = ? AND id = ?", (owner, item))
                return 0
            if kind == "dataset":
                files = [f for (f,) in self.db.execute("SELECT file FROM photo WHERE dataset_id = ?", (item,))]
                self.db.execute("DELETE FROM dataset WHERE owner = ? AND id = ?", (owner, item))
                self._drop_files(files)
            else:
                table = {"object": "objects", "scene": "scenes"}[kind]
                self.db.execute(f"DELETE FROM {table} WHERE owner = ? AND id = ?", (owner, item))
                shutil.rmtree(self.files / f"{table}/{item}", ignore_errors=True)
            self.db.execute("DELETE FROM uses WHERE kind = ? AND item = ?", (kind, item))
            self.db.execute("INSERT OR IGNORE INTO removed VALUES (?, ?)", (owner, item))
            n = 0
            for kid, refs in self.db.execute("SELECT id, refs FROM kits WHERE owner = ?", (owner,)).fetchall():
                r = json.loads(refs)
                if item in r.get(kind + "s", []):
                    r[kind + "s"].remove(item)
                    self.db.execute("UPDATE kits SET refs = ?, updated = ? WHERE id = ?", (json.dumps(r), time.time(), kid))
                    n += 1
            return n

    def forget(self, owner: str, jobs: list[str]) -> None:
        """Everything of an owner whose account is closed: datasets (and photo files no one else holds), objects,
        scenes, kits and photo descriptions; `jobs`: the names of its jobs, whose records here go too."""
        for kind, table in (("dataset", "dataset"), ("object", "objects"), ("scene", "scenes"), ("kit", "kits")):
            for (item,) in self.db.execute(f"SELECT id FROM {table} WHERE owner = ?", (owner,)).fetchall():
                self.delete(owner, kind, item)
        with self.lock:
            self.db.execute("DELETE FROM photo_description WHERE owner = ?", (owner,))
            self.db.execute("DELETE FROM removed WHERE owner = ?", (owner,))
        self.forget_jobs(jobs)

    def forget_jobs(self, jobs: list[str]) -> None:
        """The records of deleted jobs. What they gave the library (marks, objects, scenes) stays."""
        with self.lock:
            for job in jobs:
                self.db.execute("DELETE FROM uses WHERE job = ?", (job,))
                self.db.execute("DELETE FROM synced WHERE job = ?", (job,))
                self.db.execute("DELETE FROM model WHERE job = ?", (job,))

    def save_kit(self, owner: str, kit: str | None, name: str, refs: dict[str, list[str]]) -> str:
        """Create (kit None) or replace a kit. Only references the owner holds are kept."""
        clean = {f"{k}s": [i for i in dict.fromkeys(refs.get(f"{k}s", [])) if self.owns(owner, k, i)]
                 for k in ("dataset", "object", "scene")}
        now = time.time()
        with self.lock:
            if kit is None:
                kit = _id(owner, "kit", str(time.time_ns()))
                self.db.execute("INSERT INTO kits VALUES (?, ?, ?, ?, ?, ?)", (kit, owner, name, json.dumps(clean), now, now))
            else:
                self.db.execute("UPDATE kits SET name = ?, refs = ?, updated = ? WHERE owner = ? AND id = ?",
                                (name, json.dumps(clean), now, owner, kit))
        return kit

    # What is known about the photos ------------------------------------------------------------------------

    def describe(self, owner: str, photos: list[Path], classes: str, run, model: str | None = None) -> list[dict]:
        """Descriptions of the scene in these photos for these classes (scene_profile.describe), the kept ones
        reused: run(photos) is called only for those without one and its answers are kept (run None: only the kept
        ones). A photo the model could not describe is asked again next time."""
        keys = {f: content_key(f) for f in photos}
        kept = {}
        for k, data in self.db.execute(
                f"SELECT content_hash, data FROM photo_description WHERE owner = ? AND classes = ? AND content_hash IN "
                f"({','.join('?' * len(keys))})", (owner, classes, *keys.values())) if keys else []:
            kept[k] = json.loads(data)
        todo = [f for f in photos if keys[f] not in kept]
        out = [{**kept[keys[f]], "photo": f.name} for f in photos if keys[f] in kept]
        if todo and run is not None:
            # Answers come back by file name: one name in two subfolders cannot tell its photos apart, so those
            # answers are used but not kept (asked again next time) rather than kept on the wrong photo.
            names = [f.name for f in todo]
            by_name = {f.name: f for f in todo if names.count(f.name) == 1}
            for d in run(todo):
                f = by_name.get(d.get("photo"))
                if f is not None:
                    with self.lock:
                        self.db.execute("INSERT OR REPLACE INTO photo_description VALUES (?, ?, ?, ?, ?, ?)",
                                        (owner, keys[f], classes, model,
                                         json.dumps({k: v for k, v in d.items() if k != "photo"}), time.time()))
                if d.get("photo") in names:
                    out.append(d)
        return out

    def profile_stale(self, owner: str, ds: str) -> bool:
        """Has the dataset no profile, or new photos the profile has not seen while it looked at fewer than
        it could?"""
        from .scene_profile import SAMPLE
        if not self.owns(owner, "dataset", ds):
            return False
        n = self.db.execute("SELECT COUNT(*) FROM photo WHERE dataset_id = ?", (ds,)).fetchone()[0]
        row = self.db.execute("SELECT photos FROM profile WHERE dataset_id = ?", (ds,)).fetchone()
        if not n:
            return False
        return row is None or (row[0] != n and row[0] < SAMPLE)

    def profile_dataset(self, owner: str, ds: str, run, model: str | None = None, fresh: bool = False) -> dict | None:
        """A dataset's profile before any goal (scene_profile.GENERAL): what objects show up, the backgrounds,
        the places and the light, from a sample of its photos. run(photos) describes photos; the descriptions
        are kept (fresh: made anew). None when no photo could be described."""
        from . import scene_profile
        if not self.owns(owner, "dataset", ds):
            return None
        rows = self.db.execute("SELECT content_hash, file FROM photo WHERE dataset_id = ?", (ds,)).fetchall()
        files = scene_profile.sample([self.files / f for _, f in rows])
        if fresh:  # the user asked to look again: the kept descriptions are not reused
            with self.lock:
                self.db.executemany("DELETE FROM photo_description WHERE owner = ? AND content_hash = ? AND classes = ?",
                                    [(owner, f.stem, scene_profile.GENERAL) for f in files])
        found = self.describe(owner, files, scene_profile.GENERAL, run, model)
        if not found:
            return None
        prof = scene_profile.aggregate(found, general=True)
        text = scene_profile.dataset_summary(prof, len(found))
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO profile VALUES (?, ?, ?, ?, ?)",
                            (ds, json.dumps(prof), text, len(rows), time.time()))
        return {"data": prof, "summary": text}

    def profile(self, owner: str, ds: str) -> dict | None:
        row = self.db.execute("SELECT r.data, r.summary, r.photos, r.created FROM profile r JOIN dataset d "
                              "ON d.id = r.dataset_id WHERE d.owner = ? AND d.id = ?", (owner, ds)).fetchone()
        if row is None:
            return None
        sug = json.loads(row[0]).get("suggest", {})
        return {"summary": row[1], "photos": row[2], "created": row[3], "objects": sug.get("seen", []),
                "backgrounds": sug.get("backgrounds", [])}

    # Reading out -----------------------------------------------------------------------------------------

    def _used(self, kind: str, item: str) -> list[str]:
        return [j for (j,) in self.db.execute("SELECT job FROM uses WHERE kind = ? AND item = ? ORDER BY job",
                                              (kind, item)).fetchall()]

    def objects(self, owner: str) -> list[dict]:
        rows = self.db.execute("SELECT id, name, concept, source, file, thumb, kept, info, created, scene_id FROM objects "
                               "WHERE owner = ? ORDER BY created DESC", (owner,)).fetchall()
        return [{"id": oid, "name": name, "concept": concept, "source": source, "has_file": file is not None,
                 "thumb": thumb is not None, "kept": bool(kept), **json.loads(info), "created": created,
                 "jobs": self._used("object", oid), "scene": scene}
                for oid, name, concept, source, file, thumb, kept, info, created, scene in rows]

    def scenes(self, owner: str) -> list[dict]:
        """The owner's ready scenes, with the photo and dataset each comes from and the objects taken out of it."""
        rows = self.db.execute("SELECT s.id, s.name, s.route, s.photo_id, p.dataset_id, s.thumb, s.info, s.created "
                               "FROM scenes s LEFT JOIN photo p ON p.id = s.photo_id WHERE s.owner = ? "
                               "AND s.status = 'ready' ORDER BY s.created DESC", (owner,)).fetchall()
        return [{"id": sid, "name": name, "route": route, "photo_id": pid, "dataset": ds, "thumb": thumb is not None,
                 **json.loads(info), "created": created, "jobs": self._used("scene", sid),
                 "objects": [o for (o,) in self.db.execute("SELECT id FROM objects WHERE scene_id = ?", (sid,))]}
                for sid, name, route, pid, ds, thumb, info, created in rows]

    def kits(self, owner: str) -> list[dict]:
        rows = self.db.execute("SELECT id, name, refs, created, updated FROM kits WHERE owner = ? "
                               "ORDER BY updated DESC", (owner,)).fetchall()
        return [{"id": kid, "name": name, **json.loads(refs), "created": created, "updated": updated}
                for kid, name, refs, created, updated in rows]

    def dataset_card(self, owner: str, ds: str) -> dict | None:
        """What the interview learns of one dataset when it looks at it (interview "dataset" action), read now so
        changes made on the Datasets page count: its photos, its labelings (task, classes with their English
        names, photos reviewed by the user, models trained with it) and what its photos show."""
        if not self.owns(owner, "dataset", ds):
            return None
        name, = self.db.execute("SELECT name FROM dataset WHERE id = ?", (ds,)).fetchone()
        n = self.db.execute("SELECT COUNT(*) FROM photo WHERE dataset_id = ?", (ds,)).fetchone()[0]
        prof = self.profile(owner, ds)
        return {"id": ds, "name": name, "photos": n,
                "labelings": [{"id": l["id"], "name": l["name"], "task": l["task"],
                               "classes": [{"key": c["key"], "names": c["phrases"], "description": c["description"]}
                                           for c in l["classes"]],
                               "photos_reviewed_by_user": l["reviewed"], "photos_with_marks": l["marked"],
                               "models": l["models"]} for l in self.labelings(ds)],
                "what_its_photos_show": prof["summary"] if prof else None}

    def for_chat(self, owner: str) -> tuple[str, set[str]]:
        """What the interview is told of the library up front (interview.LIBRARY): an index of every kit with its
        items and the datasets outside kits. A dataset's labelings come when the interview looks at it
        (dataset_card). Objects outside kits are left out (hundreds of them, kept from earlier jobs). Returns the
        text and the ids it lists."""
        datasets = {d["id"]: d for d in self.datasets(owner)}
        objects = {o["id"]: o for o in self.objects(owner)}

        def ds_line(d):
            return f'dataset "{d["name"]}" (id {d["id"]}, {d["photos"]} photos, {len(d["labelings"])} labelings)'

        def obj_line(o):
            return f'object "{o["name"]}" (id {o["id"]}, a {o["concept"].replace("_", " ")})'
        lines, ids, in_kits = [], set(), set()
        for k in self.kits(owner):
            parts = [ds_line(datasets[i]) for i in k["datasets"] if i in datasets] + \
                    [obj_line(objects[i]) for i in k["objects"] if i in objects and objects[i]["has_file"]]
            ids |= {i for i in k["datasets"] if i in datasets} | {i for i in k["objects"] if i in objects}
            in_kits |= set(k["datasets"])
            lines.append(f'- kit "{k["name"]}": ' + ("; ".join(parts) if parts else "empty"))
        loose = [d for i, d in datasets.items() if i not in in_kits]
        if loose:
            lines.append("- datasets outside kits: " + "; ".join(ds_line(d) for d in loose))
            ids |= {d["id"] for d in loose}
        return ("\n".join(lines) if lines else "- nothing yet"), ids

    def materialize_dataset(self, owner: str, ds: str, dst: Path) -> int:
        """A dataset's photos as a plain folder (hard links), named by photo id: the chat looks at them."""
        rows = self.db.execute("SELECT p.id, p.file FROM photo p JOIN dataset d ON d.id = p.dataset_id "
                               "WHERE d.owner = ? AND d.id = ?", (owner, ds)).fetchall()
        for pid, file in rows:
            _store(self.files / file, dst / f"{pid}{Path(file).suffix}")
        return len(rows)

    def card_file(self, owner: str, kind: str, item: str) -> Path | None:
        table = {"object": "objects", "scene": "scenes"}[kind]
        row = self.db.execute(f"SELECT thumb FROM {table} WHERE owner = ? AND id = ?", (owner, item)).fetchone()
        return self.files / row[0] if row and row[0] else None


def model_summary(job: Job, running: bool) -> dict:
    """A job as a model card: what it does and how well its model did on the real test photos."""
    spec = _load(job.spec_path) or _load(job.root / "request.json") or {}
    created = min((p.stat().st_mtime for p in (job.spec_path, job.root / "request.json") if p.exists()),
                  default=job.root.stat().st_mtime)
    title = job.root / TITLE
    report = _load(job.stage_dir("export") / "report.json")
    model = None
    if report:
        test = report.get("shipped_test") or {}
        real = report.get("real_test") or report.get("real_accuracy") or {}
        model = {"metric": real.get("metric") or report.get("metric"), "score": test.get("score", real.get("mean")),
                 "kb": report.get("model_kb"), "hardware": report.get("hardware")}
    snap = _load(job.input / KEEP) or {}
    return {"job": job.name, "title": title.read_text().strip() if title.exists() else None, "created": created,
            "running": running, "task": spec.get("task"),
            "hardware": spec.get("hardware"), "classes": [c.get("id") for c in spec.get("classes", [])],
            "labeling": {"id": snap["labeling"], "name": snap.get("name"),
                         "reviewed": sum(p["reviewed"] for p in snap["photos"].values()),
                         "photos": len(snap["photos"])} if snap.get("labeling") else None,
            "model": model}
