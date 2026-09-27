"""Live 3D model search: Sketchfab's search engine, downloads from the Objaverse mirror.

Objaverse is a snapshot of ~800k Sketchfab models with open licenses, mirrored on Hugging Face, and its
uids are Sketchfab uids. So any object the user asks for can be searched by text on Sketchfab (no API
key needed to search) and the hits that the mirror has can be downloaded without a key either. Hits
are ranked by Sketchfab's relevance; asset QA (renders checked by a vision model) removes the bad ones.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SEARCH_URL = "https://api.sketchfab.com/v3/search"
# Licenses that allow commercial use and derivatives, as Sketchfab labels them.
LICENSES = {"CC Attribution": "by", "CC Attribution-ShareAlike": "by-sa", "CC0 Public Domain": "cc0"}
PAGE = 24


@dataclass(frozen=True)
class Hit:
    uid: str
    name: str
    license: str
    author: str
    source: str
    query: str


def get_json(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "sim2chip"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def search(query: str, pages: int = 4, fetch: Callable[[str], dict] = get_json) -> list[Hit]:
    """Downloadable models with an open license, in Sketchfab's relevance order."""
    params = {"type": "models", "q": query, "downloadable": "true", "count": PAGE}
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
    hits = []
    for _ in range(pages):
        data = fetch(url)
        for r in data.get("results", []):
            lic = LICENSES.get((r.get("license") or {}).get("label", ""))
            if lic and r.get("isDownloadable") and not r.get("isAgeRestricted"):
                hits.append(Hit(r["uid"], r.get("name", ""), lic, (r.get("user") or {}).get("username", ""),
                                f"https://sketchfab.com/3d-models/{r['uid']}", query))
        url = data.get("next")
        if not url:
            break
    return hits


class Mirror:
    """The set of uids the Objaverse mirror can serve (object-paths.json.gz, fetched once into the cache)."""

    def __init__(self, objaverse_dir: Path | None = None, uids: set[str] | None = None):
        if uids is None:
            from .workspace import objaverse_at
            objaverse_dir = Path(objaverse_dir)
            objaverse = objaverse_at(objaverse_dir)
            objaverse._load_object_paths()  # downloads the index if missing
            with gzip.open(objaverse_dir / "hf-objaverse-v1" / "object-paths.json.gz") as f:
                uids = set(json.load(f))
        self.uids = uids

    def __contains__(self, uid: str) -> bool:
        return uid in self.uids


def find(queries: list[str], count: int, mirror: Mirror, fetch: Callable[[str], dict] = get_json,
         pages: int = 4, exclude: set[str] = frozenset()) -> list[Hit]:
    """Up to `count` distinct models for a class, interleaving the queries so each one contributes."""
    per_query = [[h for h in search(q, pages, fetch) if h.uid in mirror] for q in queries]
    out, seen = [], set(exclude)
    for rank in range(max((len(p) for p in per_query), default=0)):
        for hits in per_query:
            if rank < len(hits) and hits[rank].uid not in seen and len(out) < count:
                seen.add(hits[rank].uid)
                out.append(hits[rank])
    return out
