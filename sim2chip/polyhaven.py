"""Poly Haven (CC0): sky maps (HDRIs) and ground textures chosen by tags, for scenes built from data.

    https://api.polyhaven.com/assets?t=hdris|textures   {id: {categories, tags, ...}}
    https://api.polyhaven.com/files/<id>                 download URLs per map and resolution

An asset matches when every requested tag is among its categories or tags. For skies the spec's weather
adds one more condition (any of the weather's categories), except indoors where weather means nothing.
"""
from __future__ import annotations

import random
import urllib.request
from pathlib import Path
from typing import Callable

from .asset_search import get_json

API = "https://api.polyhaven.com"
WEATHER = {"clear": {"clear"}, "cloudy": {"partly cloudy", "overcast"}, "evening": {"sunrise-sunset"},
           "storm": {"overcast"}, "night": {"night"}}
SKY_RES, GROUND_RES = "2k", "1k"


def catalog(kind: str, fetch: Callable[[str], dict] = get_json) -> dict[str, dict]:
    return fetch(f"{API}/assets?t={kind}")


def categories(kind: str, fetch: Callable[[str], dict] = get_json) -> list[str]:
    """Category names with at least 5 assets, most common first."""
    counts = fetch(f"{API}/categories/{kind}")
    return [c for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 5 and c != "all"
            and not c.startswith("collection")]


def _labels(a: dict) -> set[str]:
    return {x.lower() for x in a.get("categories", []) + a.get("tags", [])}


def match(assets: dict[str, dict], tags: list[str], any_of: set[str] | None = None) -> list[str]:
    want = {t.lower() for t in tags}
    return sorted(i for i, a in assets.items()
                  if want <= _labels(a) and (not any_of or any_of & _labels(a)))


def pick_skies(assets: dict[str, dict], tags: list[str], weather: list[str], n: int, seed: int) -> list[str]:
    want = {t.lower() for t in tags}
    any_of = None if "indoor" in want else set().union(*(WEATHER.get(w, set()) for w in weather))
    # Photo studios are backdrops with colored lights, not places: only when asked for.
    ids = [i for i in match(assets, tags, any_of) if "studio" in want or "studio" not in _labels(assets[i])]
    random.Random(seed).shuffle(ids)
    return ids[:n]


def pick_textures(assets: dict[str, dict], tags: list[str], n: int, seed: int) -> list[str]:
    ids = match(assets, tags)
    random.Random(seed).shuffle(ids)
    return ids[:n]


def download(asset_id: str, kind: str, dest: Path, fetch: Callable[[str], dict] = get_json) -> Path:
    """Downloads one sky (.hdr) or ground texture (diffuse .jpg) into dest/, once."""
    files = fetch(f"{API}/files/{asset_id}")
    if kind == "hdris":
        url = files["hdri"][SKY_RES]["hdr"]["url"]
    else:
        url = files["Diffuse"][GROUND_RES]["jpg"]["url"]
    out = dest / url.rsplit("/", 1)[1]
    if not out.exists():
        dest.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "sim2chip"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(out)
    return out
