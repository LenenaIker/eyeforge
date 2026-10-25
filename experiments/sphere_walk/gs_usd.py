"""A 3DGS .ply (gsplat's, gs_train.py) to a NuRec USDZ that Isaac Sim 5.1 renders, with 3DGRUT's exporter. Its own
ply_to_usd.py script asks for training cameras (2026-10-20), which a .ply does not carry. Run in 3DGRUT's venv
(source <3dgrut>/.venv/bin/activate, from the 3dgrut folder: its configs are found from there):

    python experiments/sphere_walk/gs_usd.py <splat.ply> [<out.usdz>]"""
import sys
from pathlib import Path

from hydra.compose import compose
from hydra import initialize_config_dir

from threedgrut.export import NuRecExporter
from threedgrut.model.model import MixtureOfGaussians

GRUT = Path(sys.modules["threedgrut"].__file__).resolve().parents[1]


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".usdz")
    with initialize_config_dir(version_base=None, config_dir=str(GRUT / "configs")):
        conf = compose(config_name="apps/colmap_3dgut.yaml")
    model = MixtureOfGaussians(conf)
    model.init_from_ply(str(src), init_model=False)
    NuRecExporter(export_cameras=False).export(model, out, dataset=None, conf=conf)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
