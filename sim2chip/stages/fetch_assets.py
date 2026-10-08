"""Find and download 3D models for each class, plus the sky maps of the scene.

Downloads go to the shared cache; the job only keeps a manifest. Models generated from photos
(generate_assets) and the user's own models from their library (spec `models`, files in the job's input) are
added to it.
Output: fetch_assets/manifest.json  [{uid, class, category, name, license, author, source, glb, role?}]
        (role "prop" for background props: `class` is then the prop id)
        fetch_assets/scene.json     {"skies": [...], "ground": [...], "walls": [...]}  cache-relative paths
"""
from __future__ import annotations

import gzip
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
# Name, license, author and page of every Objaverse-LVIS model, in hf-objaverse-v1/ of the cache. Objaverse
# keeps them in 160 metadata files of 3.5 MB; reading licenses for three classes took 110 of them and 130 s
# of each cloud run (2026-10-04). This file is built once from those and then travels through the cache bucket.
LVIS_META = "lvis-meta.json.gz"
OBJAVERSE_INDEXES = ("object-paths.json.gz", "lvis-annotations.json.gz", LVIS_META)


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
        return spec.synthetic

    @staticmethod
    def searched(spec) -> list[dict]:
        """Everything found by live search, in priority order: search classes first, then props."""
        return [{"id": c.id, "queries": c.assets.queries, "count": c.assets.count} for c in spec.classes
                if c.assets and c.assets.source == "search"] + \
               [{"id": p.id, "queries": p.queries, "count": p.count, "role": "prop"} for p in spec.props if p.queries]

    @staticmethod
    def library(ctx: Context) -> list[dict]:
        """The user's own models (spec `models`): copied into the cache, where convert_assets reads every model,
        with the name, license and author the library knew (objects.json next to the files)."""
        import shutil
        out = []
        wanted = [(c.id, c.assets.models, None) for c in ctx.spec.classes if c.assets and c.assets.models] + \
                 [(p.id, p.models, "prop") for p in ctx.spec.props if p.models]
        for cls, models, role in wanted:
            for rel in models:
                src = ctx.job.resolve_input(rel)
                info = json.loads((src.parent / "objects.json").read_text()).get(src.stem, {}) \
                    if (src.parent / "objects.json").exists() else {}
                dst = ctx.cache.root / "library" / f"{src.stem}.glb"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                out.append({"uid": src.stem, "class": cls, "category": info.get("concept", cls),
                            "name": info.get("name"), "license": info.get("license"), "author": info.get("author"),
                            "source": info.get("url"), "glb": str(dst.relative_to(ctx.cache.root)),
                            **({"role": role} if role else {})})
        if out:
            ctx.log(f"{len(out)} models from the user's library")
        return out

    def run(self, ctx: Context) -> dict:
        root = ctx.cache.objaverse
        root.mkdir(parents=True, exist_ok=True)
        lvis = self.objaverse_lvis(ctx)
        manifest = lvis + self.live_search(ctx, taken={m["uid"] for m in lvis})
        manifest = self.download(ctx, manifest) if manifest else []
        manifest += self.library(ctx)
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
        objaverse = objaverse_at(ctx.cache.objaverse)
        _from_bucket(ctx, [Path(objaverse._VERSIONED_PATH) / f for f in OBJAVERSE_INDEXES])
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
        anns = lvis_meta(objaverse, lvis)

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
        where = objaverse._load_object_paths()
        _from_bucket(ctx, [Path(objaverse._VERSIONED_PATH) / where[m["uid"]] for m in manifest if m["uid"] in where])
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
        cached = lambda p: _from_bucket(ctx, [p])  # noqa: E731
        if sc.sky_tags or sc.template != "road":
            ids = polyhaven.pick_skies(polyhaven.catalog("hdris"), sc.sky_tags, sc.weather, 12, seed)
            if not ids:
                raise ValueError(f"no Poly Haven sky matches tags {sc.sky_tags} and weather {sc.weather}")
            ctx.log(f"skies for {sc.sky_tags or 'any place'} + {sc.weather}: {', '.join(ids)}")
            skies = [rel(polyhaven.download(i, "hdris", ctx.cache.hdri / "polyhaven", cached=cached)) for i in ids]
        else:
            skies = [rel(ctx.cache.hdri / s.file) for s in self._road_skies(ctx)]
        ground, extra = [], {}
        if sc.ground_tags:
            ids = polyhaven.pick_textures(polyhaven.catalog("textures"), sc.ground_tags, 8, seed)
            if not ids:
                raise ValueError(f"no Poly Haven texture matches ground_tags {sc.ground_tags}")
            ctx.log(f"ground textures for {sc.ground_tags}: {', '.join(ids)}")
            ground = [rel(polyhaven.download(i, "textures", ctx.cache.root / "textures", cached=cached)) for i in ids]
            if sc.ground_patches or sc.edge_tags:
                ground, extra = self.mixed_ground(ctx, ids), {"ground_tile_m": _ground_mix_size()}
        out = {"skies": skies, "ground": ground, **extra}
        if sc.wall_tags:  # buildings and rooms of scene.contexts: a few textures per wall material
            cat, ids = polyhaven.catalog("textures"), []
            for w in sc.wall_tags:
                ids += polyhaven.pick_textures(cat, [w, "wall"], 4, seed)
            ctx.log(f"wall textures for {sc.wall_tags}: {', '.join(ids)}")
            out["walls"] = [rel(polyhaven.download(i, "textures", ctx.cache.root / "textures", cached=cached)) for i in ids]
        return out

    def mixed_ground(self, ctx: Context, base_ids: list[str]) -> list[str]:
        """Layouts of the base ground with patches and strips (ground_mix.py), at the textures' real size."""
        from .. import ground_mix, polyhaven

        sc, seed = ctx.spec.scene, ctx.spec.generation.seed
        cat, dest = polyhaven.catalog("textures"), ctx.cache.root / "textures"

        def get(ids):  # (file, side in meters); Poly Haven gives the real size in millimeters
            return [(polyhaven.download(i, "textures", dest, cached=lambda p: _from_bucket(ctx, [p])), (cat[i].get("dimensions") or [2000])[0] / 1000)
                    for i in ids]

        patches, edges = [], []
        for tags, into in ((sc.ground_patches, patches), (sc.edge_tags, edges)):
            for t in tags:
                ids = polyhaven.pick_textures(cat, [t], 3, seed)
                if not ids:
                    raise ValueError(f"no Poly Haven texture matches ground tag {t!r}")
                into += get(ids)
        files = ground_mix.make(get(base_ids), patches, edges, dest / "mix", seed, sc.patch_cover,
                                sc.patch_size_m, sc.edge_share)
        ctx.log(f"mixed ground: {len(files)} layouts, patches {sc.ground_patches}, strips {sc.edge_tags}")
        return [str(f.resolve().relative_to(ctx.cache.root.resolve())) for f in files]

    def _road_skies(self, ctx: Context) -> list:
        """The curated skies of the road template (NVIDIA's bucket), as in the spike."""
        d = ctx.cache.hdri
        d.mkdir(parents=True, exist_ok=True)
        wanted = [s for s in SKIES if s.weather in ctx.spec.scene.weather]
        if not wanted:
            raise ValueError(f"scene.weather {ctx.spec.scene.weather} matches no sky")
        _from_bucket(ctx, [d / s.file for s in wanted])
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


def _ground_mix_size() -> float:
    from ..ground_mix import SIZE_M
    return SIZE_M


def _from_bucket(ctx: Context, files: list[Path]) -> None:
    from ..cloud_sync import from_cache_bucket
    if n := from_cache_bucket(ctx.cache.root, files):
        ctx.log(f"{n} files from the cache bucket")


def lvis_meta(objaverse, lvis: dict[str, list[str]]) -> dict[str, dict]:
    """Objaverse annotations of every LVIS model, only the fields the manifest keeps (see LVIS_META)."""
    path = Path(objaverse._VERSIONED_PATH) / LVIS_META
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    anns = objaverse.load_annotations(sorted({u for uids in lvis.values() for u in uids}))
    meta = {u: {"name": a.get("name"), "license": a.get("license"),
                "user": {"username": a.get("user", {}).get("username")}, "viewerUrl": a.get("viewerUrl")}
            for u, a in anns.items()}
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt") as f:
        json.dump(meta, f, separators=(",", ":"))
    tmp.rename(path)
    return meta
