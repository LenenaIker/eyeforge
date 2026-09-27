"""Find and download 3D models for each class, plus the sky maps of the scene.

Downloads go to the shared cache; the job only keeps a manifest. Models generated from photos
(generate_assets) are added to it.
Output: fetch_assets/manifest.json  [{uid, class, category, name, license, author, source, glb, role?}]
        (role "prop" for background props: `class` is then the prop id)
        fetch_assets/scene.json     {"skies": [...], "ground": [...], "walls": [...]}  cache-relative paths
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
# Live search results are noisy (skulls, toys, fragments): fetch this many candidates per wanted model and
# let asset QA keep the best `count`.
SEARCH_OVERSAMPLE = 2


class FetchAssets(Stage):
    name = "fetch_assets"
    runtime = "cpu"
    uses_cache = True
    uses_db = True
    uses = ("classes", "props", "scene")
    after = ("generate_assets",)
    # v2: glb paths relative to cache root; v3: search oversamples; v4: scene.json; v5: asset DB.
    # Generated models need no bump: generate_assets only enters the fingerprint when it is enabled.
    version = "5"

    def enabled(self, spec) -> bool:
        return any(c.assets for c in spec.classes)

    @staticmethod
    def searched(spec) -> list[dict]:
        """Everything found by live search, in priority order: search classes first, then props."""
        return [{"id": c.id, "queries": c.assets.queries, "count": c.assets.count} for c in spec.classes
                if c.assets and c.assets.source == "search"] + \
               [{"id": p.id, "queries": p.queries, "count": p.count, "role": "prop"} for p in spec.props]

    def run(self, ctx: Context) -> dict:
        root = ctx.cache.objaverse
        root.mkdir(parents=True, exist_ok=True)
        lvis = self.objaverse_lvis(ctx)
        manifest = lvis + self.live_search(ctx, taken={m["uid"] for m in lvis})
        manifest = self.download(ctx, manifest)
        generated = ctx.job.stage_dir("generate_assets") / "manifest.json"
        if generated.exists():
            manifest += json.loads(generated.read_text())
        scene = self.scene_assets(ctx)
        out = ctx.out(self.name)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (out / "scene.json").write_text(json.dumps(scene, indent=1))
        n_sky = len(scene["skies"])
        per_class = {}
        for m in manifest:
            per_class[m["class"]] = per_class.get(m["class"], 0) + 1
        ctx.log(f"{len(manifest)} models {per_class}, {n_sky} skies, {len(scene['ground'])} ground textures")
        return {"models": per_class}

    @staticmethod
    def _objaverse(ctx: Context):
        from ..workspace import objaverse_at
        return objaverse_at(ctx.cache.objaverse)

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

        wanted = self.searched(ctx.spec)
        if not wanted:
            return []
        from ..asset_db import AssetDB

        self._objaverse(ctx)
        mirror = asset_search.Mirror(ctx.cache.objaverse)
        db = AssetDB(ctx.cache.db)
        manifest = []
        for c in wanted:  # a model found for two classes (or a class and a prop) goes to the first one only
            qs, count = c["queries"], c["count"]
            # Known good models for these queries need no review: take them first. Known bad ones are skipped.
            good = [asset_search.Hit(r["uid"], r.get("name", ""), r.get("license", ""), r.get("author", ""),
                                     r.get("source", ""), r["concept"])
                    for q in qs for r in db.known(q, "keep") if r["uid"] in mirror and r["uid"] not in taken]
            good = list({h.uid: h for h in good}.values())[:count]
            bad = {r["uid"] for q in qs for r in db.known(q, "discard")}
            need = (count - len(good)) * SEARCH_OVERSAMPLE
            found = asset_search.find(qs, need, mirror, exclude=taken | bad | {h.uid for h in good}) if need else []
            hits = good + found
            taken |= {h.uid for h in hits}
            ctx.log(f"{c['id']:16s} search {qs}: {len(good)} known good, {len(found)} new candidates"
                    f" ({len(bad)} known bad skipped)")
            role = {"role": c["role"]} if "role" in c else {}
            manifest += [{"uid": h.uid, "class": c["id"], "category": h.query, "name": h.name, "license": h.license,
                          "author": h.author, "source": h.source, **role} for h in hits]
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

    def scene_assets(self, ctx: Context) -> dict:
        """Sky maps and ground textures for the render, downloaded to the cache."""
        from .. import polyhaven

        sc, seed = ctx.spec.scene, ctx.spec.generation.seed
        rel = lambda p: str(p.resolve().relative_to(ctx.cache.root.resolve()))  # noqa: E731
        if sc.sky_tags or sc.template != "road":
            ids = polyhaven.pick_skies(polyhaven.catalog("hdris"), sc.sky_tags, sc.weather, 12, seed)
            if not ids:
                raise ValueError(f"no Poly Haven sky matches tags {sc.sky_tags} and weather {sc.weather}")
            ctx.log(f"skies for {sc.sky_tags or 'any place'} + {sc.weather}: {', '.join(ids)}")
            skies = [rel(polyhaven.download(i, "hdris", ctx.cache.hdri / "polyhaven")) for i in ids]
        else:
            skies = [rel(ctx.cache.hdri / s.file) for s in self._road_skies(ctx)]
        ground = []
        if sc.ground_tags:
            ids = polyhaven.pick_textures(polyhaven.catalog("textures"), sc.ground_tags, 8, seed)
            if not ids:
                raise ValueError(f"no Poly Haven texture matches ground_tags {sc.ground_tags}")
            ctx.log(f"ground textures for {sc.ground_tags}: {', '.join(ids)}")
            ground = [rel(polyhaven.download(i, "textures", ctx.cache.root / "textures")) for i in ids]
        out = {"skies": skies, "ground": ground}
        if sc.wall_tags:  # buildings and rooms of scene.contexts: a few textures per wall material
            cat, ids = polyhaven.catalog("textures"), []
            for w in sc.wall_tags:
                ids += polyhaven.pick_textures(cat, [w, "wall"], 4, seed)
            ctx.log(f"wall textures for {sc.wall_tags}: {', '.join(ids)}")
            out["walls"] = [rel(polyhaven.download(i, "textures", ctx.cache.root / "textures")) for i in ids]
        return out

    def _road_skies(self, ctx: Context) -> list:
        """The curated skies of the road template (NVIDIA's bucket), as in the spike."""
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
        return wanted
