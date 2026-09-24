"""Find and download 3D models for each class, plus the sky maps of the scene.

Downloads go to the shared cache; the job only keeps a manifest.
Output: fetch_assets/manifest.json  [{uid, class, category, name, license, author, source, glb}]
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from ..scenes.skies import SKIES
from .base import Context, Stage

# Licenses that allow commercial use and derivatives (attribution is listed in the export report).
OK_LICENSES = {"by", "by-sa", "cc0"}


class FetchAssets(Stage):
    name = "fetch_assets"
    runtime = "cpu"
    uses_cache = True
    uses = ("classes", "scene")
    version = "2"  # v2: glb paths relative to the cache root

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    def run(self, ctx: Context) -> dict:
        root = ctx.cache.objaverse
        root.mkdir(parents=True, exist_ok=True)
        lvis = self.objaverse_lvis(ctx)
        manifest = lvis + self.live_search(ctx, taken={m["uid"] for m in lvis})
        manifest = self.download(ctx, manifest)
        n_sky = self.skies(ctx)
        out = ctx.out(self.name)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        per_class = {}
        for m in manifest:
            per_class[m["class"]] = per_class.get(m["class"], 0) + 1
        ctx.log(f"{len(manifest)} models {per_class}, {n_sky} skies")
        return {"models": per_class}

    @staticmethod
    def _objaverse(ctx: Context):
        import objaverse
        objaverse.BASE_PATH = str(ctx.cache.objaverse)
        objaverse._VERSIONED_PATH = str(ctx.cache.objaverse / "hf-objaverse-v1")
        return objaverse

    def objaverse_lvis(self, ctx: Context) -> list[dict]:
        wanted = [c for c in ctx.spec.classes if c.assets and c.assets.source == "objaverse_lvis"]
        if not wanted:
            return []
        objaverse = self._objaverse(ctx)
        lvis = objaverse.load_lvis_annotations()
        unknown = [cat for c in wanted for cat in c.assets.categories if cat not in lvis]
        if unknown:
            raise ValueError(f"categories not in Objaverse-LVIS: {unknown}")
        anns = objaverse.load_annotations([u for c in wanted for cat in c.assets.categories for u in lvis[cat]])

        # Same selection rule as the spike: one shuffle per category with a seeded global generator,
        # so the same spec picks the same models.
        random.seed(ctx.spec.generation.seed)
        manifest = []
        for c in wanted:
            cats = c.assets.categories
            per_cat = [c.assets.count // len(cats) + (i < c.assets.count % len(cats)) for i in range(len(cats))]
            for cat, n in zip(cats, per_cat):
                ok = [u for u in lvis[cat] if (anns.get(u, {}).get("license") or "").lower() in OK_LICENSES]
                random.shuffle(ok)
                for u in ok[:n]:
                    a = anns[u]
                    manifest.append({"uid": u, "class": c.id, "category": cat, "name": a.get("name"),
                                     "license": a.get("license"), "author": a.get("user", {}).get("username"),
                                     "source": a.get("viewerUrl")})
                ctx.log(f"{c.id:16s} {cat:22s} {len(ok):4d} with an open license, took {min(len(ok), n)}")
        return manifest

    def live_search(self, ctx: Context, taken: set[str]) -> list[dict]:
        from .. import asset_search

        wanted = [c for c in ctx.spec.classes if c.assets and c.assets.source == "search"]
        if not wanted:
            return []
        self._objaverse(ctx)
        mirror = asset_search.Mirror(ctx.cache.objaverse)
        manifest = []
        for c in wanted:  # a model found for two classes goes to the first one only
            hits = asset_search.find(c.assets.queries, c.assets.count, mirror, exclude=taken)
            taken |= {h.uid for h in hits}
            ctx.log(f"{c.id:16s} search {c.assets.queries}: {len(hits)} models (open license, in the mirror)")
            manifest += [{"uid": h.uid, "class": c.id, "category": h.query, "name": h.name, "license": h.license,
                          "author": h.author, "source": h.source} for h in hits]
        return manifest

    def download(self, ctx: Context, manifest: list[dict]) -> list[dict]:
        objaverse = self._objaverse(ctx)
        paths = objaverse.load_objects(uids=[m["uid"] for m in manifest], download_processes=8)
        for m in manifest:
            # Relative to the cache root, where convert_assets resolves it.
            m["glb"] = str(Path(paths[m["uid"]]).resolve().relative_to(ctx.cache.root.resolve())) \
                if m["uid"] in paths else None
        missing = [m["uid"] for m in manifest if not m["glb"]]
        if missing:
            ctx.log(f"{len(missing)} models failed to download and are dropped")
        return [m for m in manifest if m["glb"]]

    def skies(self, ctx: Context) -> int:
        d = ctx.cache.hdri
        d.mkdir(parents=True, exist_ok=True)
        wanted = [s for s in SKIES if s.weather in ctx.spec.scene.weather]
        if not wanted:
            raise ValueError(f"scene.weather {ctx.spec.scene.weather} matches no sky")
        for s in wanted:
            f = d / s.file
            if not f.exists():
                ctx.log(f"downloading sky {s.file}")
                tmp = f.with_suffix(".part")
                urllib.request.urlretrieve(s.url, tmp)
                if tmp.stat().st_size < 10_000:  # the bucket answers wrong paths with a small XML error
                    tmp.unlink()
                    raise RuntimeError(f"sky download failed: {s.url}")
                tmp.rename(f)
        return len(wanted)
