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

from .workspace import IMAGE_EXT, Job, content_key

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
"""


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
             job.stage_dir("qa_assets") / "assets.json"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


class Library:
    def __init__(self, home: Path, cache_root: Path):
        self.root = home / "library"
        self.files = self.root / "files"
        self.cache_root = cache_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / "library.db"), check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
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
        if same is None:
            same = _id(owner, "dataset", *keys)
            if self._removed(owner, same):  # the owner deleted it: a later sync does not bring it back
                return
            created = min((p.stat().st_mtime for p in (job.spec_path, job.root / "request.json") if p.exists()),
                          default=time.time())
            self.db.execute("INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, ?, ?)",
                            (same, owner, job.name.replace("_", " "), created, int(bool(ud.get("labeled")))))
            for key, f, split, label in rows:
                rel = f"photos/{key[:2]}/{key}{f.suffix.lower()}"
                _store(f, self.files / rel)
                self.db.execute("INSERT OR IGNORE INTO photos VALUES (?, ?, ?, ?, ?)", (same, key, rel, split, label))
        self.db.execute("INSERT OR IGNORE INTO uses VALUES (?, 'dataset', ?)", (job.name, same))

    def _take_objects(self, owner: str, job: Job) -> None:
        converted = _load(job.stage_dir("convert_assets") / "assets.json") or []
        kept = {a["uid"] for a in _load(job.stage_dir("qa_assets") / "assets.json") or []}
        qa_ran = (job.stage_dir("qa_assets") / "assets.json").exists()
        for a in converted:
            uid = a["uid"]
            oid = _id(owner, "object", uid)
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
                self.db.execute("DELETE FROM photos WHERE dataset = ?", (item,))
                self.db.execute("DELETE FROM datasets WHERE owner = ? AND id = ?", (owner, item))
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

    # Reading out -----------------------------------------------------------------------------------

    def _used(self, kind: str, item: str) -> list[str]:
        return [j for (j,) in self.db.execute("SELECT job FROM uses WHERE kind = ? AND item = ? ORDER BY job",
                                              (kind, item)).fetchall()]

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
                        "jobs": self._used("dataset", ds)})
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
    report = _load(job.stage_dir("export") / "report.json")
    model = None
    if report:
        test = report.get("shipped_test") or {}
        real = report.get("real_test") or report.get("real_accuracy") or {}
        model = {"metric": real.get("metric") or report.get("metric"), "score": test.get("score", real.get("mean")),
                 "kb": report.get("model_kb"), "hardware": report.get("hardware")}
    return {"job": job.name, "created": created, "running": running, "task": spec.get("task"),
            "hardware": spec.get("hardware"), "classes": [c.get("id") for c in spec.get("classes", [])],
            "model": model}
