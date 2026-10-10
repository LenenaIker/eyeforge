"""The user's library: datasets of real photos, 3D objects and scenes, kept for good, plus kits (user,
2026-10-08).

- Every real photo belongs to a dataset. The photos a job was given (to train, to test, to build 3D models
  from) become one dataset; a job with the very same photos as an existing dataset reuses it.
- Every 3D object a job found online or built from photos is kept, with one picture of it for its card and,
  for models from the internet, the license and author (CC-BY asks to credit them).
- Scenes (photo2scene) will be kept the same way, with a picture of the built place.
- A kit has no files: a named list of references to datasets, objects and scenes, so the chat can start
  another model from it ("using kit X, count Y").
- Synthetic images are not kept here: only the few samples a job shows stay in that job.
- What is known about the photos is kept with them (user, 2026-10-09), so nothing is asked twice: per photo,
  notes (a vision model's description of the scene, the teacher's marks, the user's corrections), by the photo's
  content, so they serve every dataset and job that holds it; per dataset, a profile (what objects show up,
  the backgrounds, the places), made before anyone says what to look for.

Files are stored once by content under <home>/library/files and the SQLite file <home>/library/library.db
says who owns what. Jobs are read into the library when the owner opens it (sync), so jobs made before the
library existed are taken in too. Nothing here runs a model.
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

from .review import checked
from .workspace import IMAGE_EXT, Job, content_key

VERSION = 2  # what sync takes from a job; 2: the teacher's marks and the user's corrections
THUMB = 512  # side of the picture kept for an object's or a scene's card

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, labeled INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS photos (
    dataset TEXT NOT NULL, key TEXT NOT NULL, file TEXT NOT NULL, split TEXT NOT NULL, label TEXT,
    PRIMARY KEY (dataset, key, split));
CREATE TABLE IF NOT EXISTS objects (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, origin TEXT NOT NULL, name TEXT NOT NULL, concept TEXT NOT NULL,
    source TEXT NOT NULL, file TEXT, thumb TEXT, kept INTEGER NOT NULL, info TEXT NOT NULL, created REAL NOT NULL,
    UNIQUE (owner, origin));
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, photo TEXT, thumb TEXT, info TEXT NOT NULL,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS kits (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, refs TEXT NOT NULL, created REAL NOT NULL,
    updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS uses (
    job TEXT NOT NULL, kind TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY (job, kind, item));
CREATE TABLE IF NOT EXISTS removed (
    owner TEXT NOT NULL, item TEXT NOT NULL, PRIMARY KEY (owner, item));
CREATE TABLE IF NOT EXISTS synced (
    job TEXT PRIMARY KEY, stamp REAL NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    owner TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL, source TEXT NOT NULL,
    model TEXT, data TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (owner, key, kind, ref, source));
CREATE TABLE IF NOT EXISTS profiles (
    dataset TEXT PRIMARY KEY, data TEXT NOT NULL, summary TEXT NOT NULL, photos INTEGER NOT NULL,
    created REAL NOT NULL);
"""
# Kinds of notes on a photo. ref says what they answer: the classes the scene was described for, the English
# phrase the teacher looked for, or the labels a class was chosen among.
SCENE = "scene"    # a vision model's description of the photo (scene_profile.PROMPT)
BOXES = "boxes"    # objects found: [{"box": [x0, y0, x1, y1], "score"}] in 0..1 of the upright photo
POINTS = "points"  # objects marked: [[x, y], ...], the user's corrections of a count
CLASS = "class"    # the photo's label among the labels in ref
TASK_NAME = {"count": "counts objects", "roi_classification": "one label per photo"}
TITLE = "title.txt"  # in a job folder: the name the user gave the model


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


def stage_signature(job: Job) -> float:
    """When the job last changed what the library takes from it (its spec and the asset stages)."""
    paths = [job.spec_path, job.root / "request.json", job.stage_dir("convert_assets") / "assets.json",
             job.stage_dir("qa_assets") / "assets.json", job.stage_dir("label") / "verdicts.json",
             job.input / "corrections.json"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


class Library:
    def __init__(self, home: Path, cache_root: Path):
        self.root = home / "library"
        self.files = self.root / "files"
        self.cache_root = cache_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "library.db"), check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
        if self.db.execute("PRAGMA user_version").fetchone()[0] < VERSION:  # jobs hold more for this version
            self.db.execute("DELETE FROM synced")
            self.db.execute(f"PRAGMA user_version = {VERSION}")
        self.lock = threading.Lock()

    # Reading jobs in -------------------------------------------------------------------------------

    def sync(self, owner: str, jobs: list[Job]) -> None:
        """Take in what new or changed jobs of this owner made. Cheap when nothing changed."""
        for job in jobs:
            stamp = stage_signature(job)
            row = self.db.execute("SELECT stamp FROM synced WHERE job = ?", (job.name,)).fetchone()
            if row and row[0] >= stamp:
                continue
            with self.lock:
                self._take_photos(owner, job)
                self._take_objects(owner, job)
                self._take_labels(owner, job)
                self.db.execute("INSERT OR REPLACE INTO synced VALUES (?, ?)", (job.name, stamp))

    def _spec(self, job: Job) -> dict:
        return _load(job.spec_path) or _load(job.root / "request.json") or {}

    def _take_photos(self, owner: str, job: Job) -> None:
        spec = self._spec(job)
        ud = spec.get("user_data") or {}
        rows = []  # (key, source file, split, label)

        def folder(rel: str | None, split: str, labeled: bool):
            if not rel:
                return
            try:
                base = job.resolve_input(rel)
            except ValueError:
                return
            for f in _images(base):
                label = f.relative_to(base).parts[0] if labeled and len(f.relative_to(base).parts) > 1 else None
                rows.append((content_key(f), f, split, label))

        folder(ud.get("labeled"), "train", True)
        folder(ud.get("raw"), "train", False)
        folder(ud.get("test"), "test", True)
        for c in spec.get("classes", []):
            photos = (c.get("assets") or {}).get("photos")
            if photos and photos != "user_data":
                folder(photos, "3d", False)
                rows[:] = [(k, f, s, c["id"] if s == "3d" and lab is None else lab) for k, f, s, lab in rows]
        if not rows:
            return
        keys = sorted({k for k, *_ in rows})
        # Photos this owner already has in one dataset (the chat's, or another job's with the same photos): the
        # job uses the smallest such dataset and nothing new is made.
        same, size = None, None
        for (ds,) in self.db.execute("SELECT id FROM datasets WHERE owner = ?", (owner,)).fetchall():
            have = {k for (k,) in self.db.execute("SELECT key FROM photos WHERE dataset = ?", (ds,)).fetchall()}
            if set(keys) <= have and (size is None or len(have) < size):
                same, size = ds, len(have)
        if same is None:  # photos added to a job (the live test, "Make it better") join the dataset it used
            for (ds,) in self.db.execute("SELECT item FROM uses WHERE job = ? AND kind = 'dataset'", (job.name,)).fetchall():
                have = {k for (k,) in self.db.execute("SELECT key FROM photos WHERE dataset = ?", (ds,)).fetchall()}
                if have and have <= set(keys) and self.owns(owner, "dataset", ds):
                    self._add_rows(ds, [r for r in rows if r[0] not in have])
                    return
        if same is None:
            same = _id(owner, "dataset", *keys)
            if self._removed(owner, same):  # the owner deleted it: a later sync does not bring it back
                return
            created = min((p.stat().st_mtime for p in (job.spec_path, job.root / "request.json") if p.exists()),
                          default=time.time())
            self.db.execute("INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, ?, ?)",
                            (same, owner, job.name.replace("_", " "), created, int(bool(ud.get("labeled")))))
            self._add_rows(same, rows)
        self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'dataset', ?)", (job.name, same))

    def _add_rows(self, ds: str, rows: list) -> None:
        for key, f, split, label in rows:
            rel = f"photos/{key[:2]}/{key}{f.suffix.lower()}"
            _store(f, self.files / rel)
            self.db.execute("INSERT OR IGNORE INTO photos VALUES (?, ?, ?, ?, ?)", (ds, key, rel, split, label))

    def _take_labels(self, owner: str, job: Job) -> None:
        """The teacher's marks and the user's corrections of a job's raw photos, as notes on those photos: a
        count's boxes and points by the English phrase of each class, a label among the job's labels."""
        spec = self._spec(job)
        raw = (spec.get("user_data") or {}).get("raw")
        verdicts = _load(job.stage_dir("label") / "verdicts.json") or {}
        fixes = _load(job.input / "corrections.json") or {}
        if not raw or not isinstance(raw, str) or not (verdicts or fixes):
            return
        try:
            base = job.resolve_input(raw)
        except ValueError:
            return
        learned = _load(job.stage_dir("label") / "teacher.json") or {}
        ids = [c.get("id") for c in spec.get("classes", []) if c.get("id")]
        phrase = {c: (learned.get("phrases") or {}).get(c, [c.replace("_", " ")])[0] for c in ids}
        labels = json.dumps(sorted(set(ids) | ({spec["background_class"]} if spec.get("background_class") else set())))
        keys: dict[str, str] = {}

        def key(rel: str) -> str | None:
            if rel not in keys:
                f = base / rel
                keys[rel] = content_key(f) if f.is_file() else None
            return keys[rel]
        rows = []
        for rel, v in verdicts.items():
            if not (k := key(rel)):
                continue
            if "objects" in v:  # count
                teacher_marks = v.get("teacher_objects", v["objects"]) if v.get("judge") == "user" else v["objects"]
                if teacher_marks is None:
                    continue
                for c in ids:
                    boxes = [{"box": o["box"], "score": o.get("score")} for o in teacher_marks
                             if o.get("class") == c and o.get("box")]
                    rows.append((k, BOXES, phrase[c], "model", v.get("model"),
                                 {"boxes": boxes, "threshold": v.get("threshold")}))
            elif v.get("class") and v["class"] != "no_answer":  # classification
                cls = v.get("teacher", v["class"]) if v.get("judge") == "user" else v["class"]
                rows.append((k, CLASS, labels, "model", v.get("model"), {"label": cls, "confidence": v.get("confidence")}))
        fixes = {"points": {}, **fixes}
        for rel, marks in fixes["points"].items():
            if k := key(rel):
                for c in checked(fixes, rel, ids):  # only the classes the user looked at on this photo
                    rows.append((k, POINTS, phrase[c], "user", None, {"points": [[x, y] for x, y, cl in marks if cl == c]}))
        for rel, label in (fixes.get("labels") or {}).items():
            if k := key(rel):
                rows.append((k, CLASS, labels, "user", None, {"label": label}))
        now = time.time()
        self.db.executemany("INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [(owner, k, kind, ref, src, model, json.dumps(data), now)
                             for k, kind, ref, src, model, data in rows])

    def _take_objects(self, owner: str, job: Job) -> None:
        converted = _load(job.stage_dir("convert_assets") / "assets.json") or []
        kept = {a["uid"] for a in _load(job.stage_dir("qa_assets") / "assets.json") or []}
        qa_ran = (job.stage_dir("qa_assets") / "assets.json").exists()
        for a in converted:
            uid = a["uid"]
            # A model the job took from this library (fetch_assets uses the object's id as uid) is that object.
            oid = uid if self.owns(owner, "object", uid) else _id(owner, "object", uid)
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
                source = "photos" if a.get("generated") else "online"
                self.db.execute("INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (oid, owner, f"uid:{uid}", a.get("name") or a["class"], a["class"], source, file,
                                 thumb, is_kept, json.dumps(info), time.time()))
            self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'object', ?)", (job.name, oid))

    def _removed(self, owner: str, item: str) -> bool:
        return self.db.execute("SELECT 1 FROM removed WHERE owner = ? AND item = ?", (owner, item)).fetchone() is not None

    def _put_photos(self, ds: str, files: list[Path], split: str, label: str | None) -> int:
        for f in files:
            key = content_key(f)
            rel = f"photos/{key[:2]}/{key}{f.suffix.lower()}"
            _store(f, self.files / rel)
            self.db.execute("INSERT OR IGNORE INTO photos VALUES (?, ?, ?, ?, ?)", (ds, key, rel, split, label))
        return len(files)

    def add_chat_photos(self, owner: str, session: str, files: list[Path], split: str, label: str | None) -> str:
        """Photos attached in a chat go to that chat's dataset, made with the first ones."""
        ds = _id(owner, "chat", session)
        with self.lock:
            self.db.execute("DELETE FROM removed WHERE owner = ? AND item = ?", (owner, ds))
            self.db.execute("INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, ?, 0)",
                            (ds, owner, time.strftime("Chat photos %d %b %Y %H:%M"), time.time()))
            self._put_photos(ds, files, split, label)
        return ds

    # The owner's own changes -------------------------------------------------------------------------

    def owns(self, owner: str, kind: str, item: str) -> bool:
        table = {"dataset": "datasets", "object": "objects", "scene": "scenes", "kit": "kits"}[kind]
        return self.db.execute(f"SELECT 1 FROM {table} WHERE owner = ? AND id = ?", (owner, item)).fetchone() is not None

    def new_dataset(self, owner: str, name: str, files: list[Path]) -> str:
        """Photos uploaded on the Datasets page: a dataset of their own."""
        ds = _id(owner, "upload", str(time.time_ns()))
        with self.lock:
            self.db.execute("INSERT INTO datasets VALUES (?, ?, ?, ?, 0)", (ds, owner, name, time.time()))
            self._put_photos(ds, files, "train", None)
        return ds

    def add_photos(self, owner: str, ds: str, files: list[Path]) -> int:
        with self.lock:
            return self._put_photos(ds, files, "train", None)

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
            self.db.execute("INSERT INTO objects VALUES (?, ?, ?, ?, ?, 'upload', ?, ?, 1, '{}', ?)",
                            (oid, owner, f"upload:{oid}", name, concept, file, thumb, time.time()))
        return oid

    def rename(self, owner: str, kind: str, item: str, name: str) -> None:
        table = {"dataset": "datasets", "object": "objects", "scene": "scenes", "kit": "kits"}[kind]
        with self.lock:
            self.db.execute(f"UPDATE {table} SET name = ? WHERE owner = ? AND id = ?", (name, owner, item))

    def delete(self, owner: str, kind: str, item: str) -> int:
        """Delete a dataset, object or scene (kept deleted: syncing its jobs again does not bring it back) or a
        kit. The item leaves every kit of the owner; returns how many kits it was in. A photo's file goes only
        when no dataset holds it any more."""
        with self.lock:
            if kind == "kit":
                self.db.execute("DELETE FROM kits WHERE owner = ? AND id = ?", (owner, item))
                return 0
            if kind == "dataset":
                files = [f for (f,) in self.db.execute("SELECT DISTINCT file FROM photos WHERE dataset = ?", (item,))]
                keys = [k for (k,) in self.db.execute("SELECT DISTINCT key FROM photos WHERE dataset = ?", (item,))]
                self.db.execute("DELETE FROM photos WHERE dataset = ?", (item,))
                self.db.execute("DELETE FROM datasets WHERE owner = ? AND id = ?", (owner, item))
                self.db.execute("DELETE FROM profiles WHERE dataset = ?", (item,))
                for k in keys:  # what was known about a photo goes with the owner's last copy of it
                    if not self.db.execute("SELECT 1 FROM photos p JOIN datasets d ON d.id = p.dataset "
                                           "WHERE d.owner = ? AND p.key = ?", (owner, k)).fetchone():
                        self.db.execute("DELETE FROM notes WHERE owner = ? AND key = ?", (owner, k))
                for f in files:
                    if not self.db.execute("SELECT 1 FROM photos WHERE file = ?", (f,)).fetchone():
                        (self.files / f).unlink(missing_ok=True)
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
        scenes, kits and notes; `jobs`: the names of its jobs, whose records here go too."""
        for kind, table in (("dataset", "datasets"), ("object", "objects"), ("scene", "scenes"), ("kit", "kits")):
            for (item,) in self.db.execute(f"SELECT id FROM {table} WHERE owner = ?", (owner,)).fetchall():
                self.delete(owner, kind, item)
        with self.lock:
            self.db.execute("DELETE FROM notes WHERE owner = ?", (owner,))
            self.db.execute("DELETE FROM removed WHERE owner = ?", (owner,))
        self.forget_jobs(jobs)

    def forget_jobs(self, jobs: list[str]) -> None:
        """The records of deleted jobs. What they gave the library (photos, labels, objects, scenes) stays."""
        with self.lock:
            for job in jobs:
                self.db.execute("DELETE FROM uses WHERE job = ?", (job,))
                self.db.execute("DELETE FROM synced WHERE job = ?", (job,))

    def _label_kinds(self, kind: str) -> tuple[str, ...]:
        if kind not in (BOXES, CLASS):
            raise ValueError(f"no labels of kind {kind}")
        return (BOXES, POINTS) if kind == BOXES else (CLASS,)

    def forget_labels(self, owner: str, ds: str, kind: str, ref: str) -> int:
        """One labeling of a dataset's photos (the teacher's boxes and the user's marks for a phrase, or the labels
        chosen among some labels), wherever those photos are used: labels belong to the photo's content."""
        kinds = self._label_kinds(kind)
        with self.lock:
            return self.db.execute(
                f"DELETE FROM notes WHERE owner = ? AND kind IN ({','.join('?' * len(kinds))}) AND ref = ? "
                "AND key IN (SELECT key FROM photos WHERE dataset = ?)", (owner, *kinds, ref, ds)).rowcount

    def rename_labels(self, owner: str, ds: str, ref: str, to: str) -> int:
        """A count labeling of a dataset's photos under another English phrase, e.g. "paper bin" -> "blue bin": a
        later job looking for the new phrase reuses them. A photo labeled for both keeps the new phrase's."""
        keys = "key IN (SELECT key FROM photos WHERE dataset = ?)"
        with self.lock:
            moved = self.db.execute(f"UPDATE OR IGNORE notes SET ref = ? WHERE owner = ? AND kind IN (?, ?) AND ref = ? "
                                    f"AND {keys}", (to, owner, BOXES, POINTS, ref, ds)).rowcount
            self.db.execute(f"DELETE FROM notes WHERE owner = ? AND kind IN (?, ?) AND ref = ? AND {keys}",
                            (owner, BOXES, POINTS, ref, ds))
        return moved

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

    # What is known about the photos ----------------------------------------------------------------

    def notes(self, owner: str, keys: list[str], kind: str, ref: str) -> dict[str, dict]:
        """Notes of one kind on these photos, the user's winning over a model's: {key: note}."""
        out: dict[str, dict] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = self.db.execute(
                f"SELECT key, source, model, data, created FROM notes WHERE owner = ? AND kind = ? AND ref = ? "
                f"AND key IN ({','.join('?' * len(part))}) ORDER BY source = 'user'", (owner, kind, ref, *part))
            for key, source, model, data, created in rows:
                out[key] = {"source": source, "model": model, "data": json.loads(data), "created": created}
        return out

    def note(self, owner: str, key: str, kind: str, ref: str, data, source: str = "model",
             model: str | None = None) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (owner, key, kind, ref, source, model, json.dumps(data), time.time()))

    def describe(self, owner: str, photos: list[Path], classes: str, run, model: str | None = None) -> list[dict]:
        """Descriptions of the scene in these photos for these classes (scene_profile.describe), the kept ones
        reused: run(photos) is called only for those without one and its answers are kept. A photo the model
        could not describe is asked again next time."""
        keys = {f: content_key(f) for f in photos}
        kept = self.notes(owner, list(keys.values()), SCENE, classes)
        todo = [f for f in photos if keys[f] not in kept]
        out = [{**n["data"], "photo": f.name} for f in photos if (n := kept.get(keys[f]))]
        if todo:
            by_name = {f.name: f for f in todo}
            for d in run(todo):
                f = by_name.get(d.get("photo"))
                if f is not None:
                    self.note(owner, keys[f], SCENE, classes, {k: v for k, v in d.items() if k != "photo"},
                              model=model)
                    out.append(d)
        return out

    def profile_stale(self, owner: str, ds: str) -> bool:
        """Has the dataset no profile, or new photos the profile has not seen while it looked at fewer than
        it could?"""
        from .scene_profile import SAMPLE
        n = self.db.execute("SELECT COUNT(DISTINCT key) FROM photos WHERE dataset = ?", (ds,)).fetchone()[0]
        row = self.db.execute("SELECT photos FROM profiles WHERE dataset = ?", (ds,)).fetchone()
        if not self.owns(owner, "dataset", ds) or not n:
            return False
        return row is None or (row[0] != n and row[0] < SAMPLE)

    def profile_dataset(self, owner: str, ds: str, run, model: str | None = None, fresh: bool = False) -> dict | None:
        """A dataset's profile before any goal (scene_profile.GENERAL): what objects show up, the backgrounds,
        the places and the light, from a sample of its photos. run(photos) describes photos; the descriptions
        are kept as notes (fresh: made anew). None when no photo could be described."""
        from . import scene_profile
        rows = self.db.execute("SELECT DISTINCT p.key, p.file FROM photos p JOIN datasets d ON d.id = p.dataset "
                               "WHERE d.owner = ? AND d.id = ?", (owner, ds)).fetchall()
        files = scene_profile.sample([self.files / f for _, f in rows])
        if fresh:  # the user asked to look again: the kept descriptions are not reused
            with self.lock:
                self.db.executemany("DELETE FROM notes WHERE owner = ? AND key = ? AND kind = ? AND ref = ?",
                                    [(owner, f.stem, SCENE, scene_profile.GENERAL) for f in files])
        found = self.describe(owner, files, scene_profile.GENERAL, run, model)
        if not found:
            return None
        prof = scene_profile.aggregate(found)
        text = scene_profile.dataset_summary(prof, len(found))
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO profiles VALUES (?, ?, ?, ?, ?)",
                            (ds, json.dumps(prof), text, len({k for k, _ in rows}), time.time()))
        return {"data": prof, "summary": text}

    def dataset_notes(self, owner: str, ds: str) -> dict[str, list[dict]]:
        """Every note on the dataset's photos: {key: [{kind, ref, source, model, data}]}."""
        if not self.owns(owner, "dataset", ds):
            return {}
        out: dict[str, list[dict]] = {}
        rows = self.db.execute("SELECT n.key, n.kind, n.ref, n.source, n.model, n.data FROM notes n WHERE n.owner = ? "
                               "AND n.key IN (SELECT key FROM photos WHERE dataset = ?) ORDER BY n.key, n.kind",
                               (owner, ds))
        for key, kind, ref, source, model, data in rows:
            out.setdefault(key, []).append({"kind": kind, "ref": ref, "source": source, "model": model,
                                            "data": json.loads(data)})
        return out

    def profile(self, owner: str, ds: str) -> dict | None:
        row = self.db.execute("SELECT r.data, r.summary, r.photos, r.created FROM profiles r JOIN datasets d "
                              "ON d.id = r.dataset WHERE d.owner = ? AND d.id = ?", (owner, ds)).fetchone()
        if row is None:
            return None
        sug = json.loads(row[0]).get("suggest", {})
        return {"summary": row[1], "photos": row[2], "created": row[3], "objects": sug.get("seen", []),
                "backgrounds": sug.get("backgrounds", [])}

    # Into a job ------------------------------------------------------------------------------------

    def materialize_dataset(self, owner: str, ds: str, dst: Path, by_label: bool) -> int:
        """A dataset's photos as a plain folder of a job (hard links): one subfolder per label for labeled
        photos (by_label), else flat. Photos without a label go flat even then."""
        rows = self.db.execute("SELECT p.key, p.file, p.label FROM photos p JOIN datasets d ON d.id = p.dataset "
                               "WHERE d.owner = ? AND d.id = ?", (owner, ds)).fetchall()
        for key, file, label in rows:
            sub = dst / label if by_label and label else dst
            _store(self.files / file, sub / Path(file).name)
        return len(rows)

    def corrections_for(self, owner: str, folder: Path, ids: list[str], phrases: dict[str, str],
                        labels: list[str]) -> dict:
        """The owner's earlier corrections of the photos in a job's raw folder, as that job's corrections.json
        (keys relative to the folder): a count's points for the classes the user marked on the photo before
        (phrases: class id -> English phrase), named in "checked" when not every class of the job, a photo's
        label when it was chosen among the very same labels."""
        files = {str(f.relative_to(folder)): content_key(f) for f in _images(folder)}
        out: dict = {"labels": {}, "points": {}, "checked": {}}
        same = json.dumps(sorted(labels))
        chosen = self.notes(owner, list(files.values()), CLASS, same)
        marked = {c: self.notes(owner, list(files.values()), POINTS, p) for c, p in phrases.items()}
        for rel, k in files.items():
            if (n := chosen.get(k)) and n["source"] == "user":
                out["labels"][rel] = n["data"]["label"]
            mine = [c for c in ids if k in marked[c] and marked[c][k]["source"] == "user"]
            if mine:
                out["points"][rel] = [[x, y, c] for c in mine for x, y in marked[c][k]["data"]["points"]]
                if len(mine) < len(ids):
                    out["checked"][rel] = mine
        return {k: v for k, v in out.items() if v}

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

    # Reading out -----------------------------------------------------------------------------------

    def _used(self, kind: str, item: str) -> list[str]:
        return [j for (j,) in self.db.execute("SELECT job FROM uses WHERE kind = ? AND item = ? ORDER BY job",
                                              (kind, item)).fetchall()]

    def labelings(self, ds: str) -> list[dict]:
        """How earlier jobs had the teacher label this dataset, newest first: {job, task, classes (ids), phrases
        (count: the English phrase the teacher looked for, per class), background, checked (photos the user
        corrected), models}. Only jobs that went past the teacher's check
        (a check left open is no labeling the user accepted). A job with the same task and classes gets the
        same labels again without a check (server.follow_labeling). models: per class or prop of the job, its
        3D model search queries and the models kept for it, so a new job can find them again."""
        out = []
        # Labels the user deleted (forget_labels) are no labeling any more: only jobs already taken in (synced)
        # are judged by them, a job that just ended may not be.
        row = self.db.execute("SELECT owner FROM datasets WHERE id = ?", (ds,)).fetchone()
        have = set(self.db.execute("SELECT DISTINCT kind, ref FROM notes WHERE owner = ? AND key IN "
                                   "(SELECT key FROM photos WHERE dataset = ?)", (row[0] if row else "", ds)).fetchall())
        synced = {j for (j,) in self.db.execute("SELECT job FROM synced").fetchall()}
        for job in self._used("dataset", ds):
            root = self.root.parent / "jobs" / job
            spec = _load(root / "request.json")
            if not spec or not (spec.get("user_data") or {}).get("raw") or (root / "calibration.json").exists() \
                    or not (root / "label" / "verdicts.json").exists():
                continue
            fixes = _load(root / "input" / "corrections.json") or {}
            kept = dict(self.db.execute("SELECT o.concept, SUM(o.kept) FROM uses u JOIN objects o ON o.id = u.item "
                                        "WHERE u.job = ? AND u.kind = 'object' GROUP BY o.concept", (job,)).fetchall())
            models = {}
            for c in [*spec.get("classes", []), *spec.get("props", [])]:
                queries = (c.get("assets") or {}).get("queries") if "assets" in c else c.get("queries")
                if queries:
                    models[c["id"]] = {"queries": queries, "kept": int(kept.get(c["id"]) or 0)}
            ids = [c["id"] for c in spec.get("classes", [])]
            learned = (_load(root / "label" / "teacher.json") or {}).get("phrases") or {}
            if job in synced:
                if spec.get("task") == "count":
                    ids = [c for c in ids if {(BOXES, learned.get(c, [c.replace("_", " ")])[0]),
                                              (POINTS, learned.get(c, [c.replace("_", " ")])[0])} & have]
                elif (CLASS, json.dumps(sorted(set(ids) | ({spec["background_class"]} if spec.get("background_class")
                                                              else set())))) not in have:
                    ids = []
                if not ids:
                    continue
            out.append({"job": job, "task": spec.get("task"), "classes": ids,
                        "phrases": {c: learned.get(c, [c.replace("_", " ")])[0] for c in ids},
                        "background": spec.get("background_class"),
                        "checked": len(set(fixes.get("labels") or {}) | set(fixes.get("points") or {})),
                        "models": models, "t": (root / "request.json").stat().st_mtime})
        return sorted(out, key=lambda d: -d["t"])

    def known_labels(self, owner: str, folder: Path, phrases: dict[str, str], labels: list[str]) -> dict[str, int]:
        """Photos of the folder the teacher (or the user) already labeled, per class: a count's boxes or points for
        the class's phrase (phrases: class id -> English phrase), else a label among these very labels (the
        same number for every label)."""
        keys = [content_key(f) for f in _images(folder)]
        if phrases:
            return {c: len(set(self.notes(owner, keys, BOXES, p)) | set(self.notes(owner, keys, POINTS, p)))
                    for c, p in phrases.items()}
        return dict.fromkeys(labels, len(self.notes(owner, keys, CLASS, json.dumps(sorted(labels)))))

    def datasets(self, owner: str) -> list[dict]:
        out = []
        for ds, name, created, labeled in self.db.execute(
                "SELECT id, name, created, labeled FROM datasets WHERE owner = ? ORDER BY created DESC", (owner,)):
            splits = dict(self.db.execute("SELECT split, COUNT(*) FROM photos WHERE dataset = ? GROUP BY split", (ds,)))
            sample = [k for (k,) in self.db.execute("SELECT key FROM photos WHERE dataset = ? ORDER BY key LIMIT 4",
                                                    (ds,))]
            labels = [l for (l,) in self.db.execute("SELECT DISTINCT label FROM photos WHERE dataset = ? AND label "
                                                    "IS NOT NULL ORDER BY label", (ds,))]
            out.append({"id": ds, "name": name, "created": created, "labeled": bool(labeled),
                        "photos": sum(splits.values()), "splits": splits, "labels": labels, "sample": sample,
                        "jobs": self._used("dataset", ds), "profile": self.profile(owner, ds)})
        return out

    def objects(self, owner: str) -> list[dict]:
        rows = self.db.execute("SELECT id, name, concept, source, file, thumb, kept, info, created FROM objects "
                               "WHERE owner = ? ORDER BY created DESC", (owner,)).fetchall()
        return [{"id": oid, "name": name, "concept": concept, "source": source, "has_file": file is not None,
                 "thumb": thumb is not None, "kept": bool(kept), **json.loads(info), "created": created,
                 "jobs": self._used("object", oid)}
                for oid, name, concept, source, file, thumb, kept, info, created in rows]

    def scenes(self, owner: str) -> list[dict]:
        rows = self.db.execute("SELECT id, name, thumb, info, created FROM scenes WHERE owner = ? "
                               "ORDER BY created DESC", (owner,)).fetchall()
        return [{"id": sid, "name": name, "thumb": thumb is not None, **json.loads(info), "created": created}
                for sid, name, thumb, info, created in rows]

    def kits(self, owner: str) -> list[dict]:
        rows = self.db.execute("SELECT id, name, refs, created, updated FROM kits WHERE owner = ? "
                               "ORDER BY updated DESC", (owner,)).fetchall()
        return [{"id": kid, "name": name, **json.loads(refs), "created": created, "updated": updated}
                for kid, name, refs, created, updated in rows]

    def for_chat(self, owner: str) -> tuple[str, set[str]]:
        """What the interview is told of the library (interview.LIBRARY): every kit with its items, and the
        datasets outside kits. Objects outside kits are left out (hundreds of them, kept from earlier jobs).
        Returns the text and the ids it lists."""
        datasets = {d["id"]: d for d in self.datasets(owner)}
        objects = {o["id"]: o for o in self.objects(owner)}

        def ds_line(d):
            labels = f", labels {', '.join(d['labels'])}" if d["labels"] else ", no labels"
            seen = f". {d['profile']['summary']}" if d["profile"] else ""
            done = "".join(f'. The teacher labeled it before for model "{l["job"]}" (still "raw" for a new model: '
                           f'{TASK_NAME.get(l["task"], l["task"])}, classes {", ".join(l["classes"])}'
                           f'{", background " + l["background"] if l["background"] else ""}; {l["checked"]} photos '
                           f'checked by the user; 3D models kept: ' + (", ".join(
                               f'{k} {m["kept"]} (search {m["queries"]})' for k, m in l["models"].items()) or "none")
                           + ")" for l in self.labelings(d["id"])[:2])
            return f'dataset "{d["name"]}" (id {d["id"]}, {d["photos"]} photos{labels}{done}{seen})'

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

    def photo_file(self, owner: str, key: str) -> Path | None:
        row = self.db.execute("SELECT p.file FROM photos p JOIN datasets d ON d.id = p.dataset "
                              "WHERE d.owner = ? AND p.key = ? LIMIT 1", (owner, key)).fetchone()
        return self.files / row[0] if row else None

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
    return {"job": job.name, "title": title.read_text().strip() if title.exists() else None, "created": created,
            "running": running, "task": spec.get("task"),
            "hardware": spec.get("hardware"), "classes": [c.get("id") for c in spec.get("classes", [])],
            "model": model}
