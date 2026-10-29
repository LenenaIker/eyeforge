"""A 3DGS .ply (gsplat's) to a NuRec USDZ that Isaac Sim 5.1 renders, like gs_usd.py, but through 3DGRUT's
transcoding path (PLY importer, attributes adapter, NuRec exporter) instead of its MixtureOfGaussians model: no
tracer is imported, so none of 3DGRUT's CUDA extensions is built. Runs in any Python with torch, usd-core,
hydra-core and 3DGRUT's checkout on the path (the route image on the scene VM):

    python experiments/sphere_walk/gs_usd_light.py <3dgrut checkout> <splat.ply> [<out.usdz>]"""
import sys
from pathlib import Path


def main():
    grut, src = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else src.with_suffix(".usdz")
    sys.path.insert(0, str(grut))
    from hydra import initialize_config_dir
    from hydra.compose import compose
    from threedgrut.export.adapter import AttributesExportAdapter
    from threedgrut.export.importers import PLYImporter
    from threedgrut.export.usd.nurec.exporter import NuRecExporter

    with initialize_config_dir(version_base=None, config_dir=str(grut / "configs")):
        conf = compose(config_name="apps/colmap_3dgut.yaml")
    importer = PLYImporter(max_sh_degree=conf.model.progressive_training.max_n_features)
    attrs, caps = importer.load(src)
    model = AttributesExportAdapter(attrs, caps, is_preactivation=importer.stores_preactivation)
    NuRecExporter(export_cameras=False).export(model, out, dataset=None, conf=conf)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
