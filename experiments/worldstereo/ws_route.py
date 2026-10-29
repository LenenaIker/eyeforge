"""The WorldStereo scene route (recipe F3c): one photo to a walkable NuRec splat for Isaac Sim. The code is in wsroute/;
the closed experiments that led here, frozen, in closed/ (docs/pruebas_reconstruccion_worldstereo.md).

Run the driver in the edge-sdg env (boto3 for the cloud steps); it calls each step in its own interpreter
(wsroute/paths.py lists the variables):

    python experiments/worldstereo/ws_route.py init --photo photo21 [--pano-seed 1] [--spots "9,-4;8,-4"]
    python experiments/worldstereo/ws_route.py status --case e2e_photo21
    python experiments/worldstereo/ws_route.py run --case e2e_photo21                    # up to the walks' sheet
    python experiments/worldstereo/ws_route.py run --case e2e_photo21 --until bundle     # pays two VM runs
    python experiments/worldstereo/ws_route.py remote <photo file> --photo P --case C   # all of it on the VM

Every step can also run alone, in its interpreter: `<python> ws_route.py <step> --case <case> [...]`."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsroute import config, erase  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=[
        "init", "status", "run",
        "things", "masks", "split", "publish", "caption", "ingest", "pano", "pano_paste", "walks", "result_sheet",
        "cloud", "vm", "depth", "views", "train", "audit", "plan", "stage", "grid", "sheet", "compare",
        "pseudo", "fix", "fixer_views", "bundle", "remote"])
    ap.add_argument("what", nargs="?", help="vm: disk, prep, gpu, setup, stop or status; ingest, remote: the source "
                                            "photo")
    ap.add_argument("--case")
    ap.add_argument("--photo", help="init, erase steps: the photo's name in data/panogen_test/in/")
    ap.add_argument("--pano-seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=None, help="init, views: leg folders left out (prefixes)")
    ap.add_argument("--spots", default=None, help='init: "x,y;x,y" grid spots the sheets show')
    ap.add_argument("--until", default="walks", help="run: the last step (a cloud step pays a VM run)")
    ap.add_argument("--arm")
    ap.add_argument("--arms", help="compare: comma-separated arms")
    ap.add_argument("--tag", help="fix, fixer_views: the Fixer round (F3a, ...)")
    ap.add_argument("--step", type=int, help="pseudo: the round's number, from 1")
    ap.add_argument("--base", help="fixer_views: the views file the pseudo views are added to")
    ap.add_argument("--out", help="remote: the folder the bundle lands in (default the case's)")
    ap.add_argument("--resume", action="store_true", help="cloud: ship the generated legs, generate only new ones")
    a = ap.parse_args()

    if a.cmd == "init":
        case = a.case or erase.case_name(a.photo)
        cfg = config.load(case)
        cfg.photo = a.photo or cfg.photo
        cfg.pano_seed = a.pano_seed
        if a.skip is not None:
            cfg.skip_legs = a.skip
        if a.spots is not None:
            cfg.spots = config.spots_arg(a.spots)
        cfg.save()
        return print(cfg)
    if a.cmd in ("things", "masks", "split", "publish", "caption"):
        return getattr(erase, a.cmd)(a.photo, a.case or erase.case_name(a.photo))
    if a.cmd == "ingest":
        from wsroute import pano
        return pano.ingest(Path(a.what), a.photo)
    if a.cmd == "vm":
        from wsroute import vm
        from wsroute import scenevm
        return {"disk": vm.disk, "prep": vm.prep, "gpu": vm.gpu, "setup": scenevm.setup, "stop": vm.stop,
                "status": vm.status}[a.what]()
    if a.cmd == "remote":  # the whole route on the GPU VM (scenevm.py), through bundle
        from wsroute import scenevm
        return scenevm.remote(Path(a.what), a.photo, a.case, Path(a.out) if a.out else None)

    cfg = config.load(a.case)
    if a.cmd == "status":
        from wsroute import steps
        steps.status(cfg)
    elif a.cmd == "run":
        from wsroute import steps
        steps.run(cfg, a.until)
    elif a.cmd == "pano":
        from wsroute import pano
        pano.pano(cfg.case, cfg.pano_seed)
    elif a.cmd == "pano_paste":
        from wsroute import pano
        pano.pano_paste(cfg.case, cfg.pano_seed)
    elif a.cmd == "walks":
        from wsroute import walks
        walks.prep(cfg.case, cfg.pano)
    elif a.cmd == "result_sheet":
        from wsroute import walks
        walks.result_sheet(cfg.case)
    elif a.cmd == "cloud":
        from wsroute import vm
        vm.run(cfg.case, cfg.gpu_hours, a.resume)
    elif a.cmd == "depth":
        from wsroute import depth
        depth.depth(cfg.case)
        depth.anchor(cfg.case)
        depth.consistency(cfg.case)
    elif a.cmd == "views":
        from wsroute import splat
        splat.views(cfg.case, config.arm_views(a.arm), tuple(cfg.skip_legs if a.skip is None else a.skip))
    elif a.cmd == "train":
        from wsroute import splat
        splat.train(cfg.case, a.arm)
    elif a.cmd == "audit":
        from wsroute import nbv
        nbv.audit(cfg.case, a.arm, cfg.radius)
    elif a.cmd == "plan":
        from wsroute import nbv
        nbv.plan(cfg.case, a.arm, cfg.radius, cfg.nbv_legs, cfg.nbv_max_unknown, cfg.nbv_min_gain, cfg.nbv_prefix)
    elif a.cmd == "stage":
        from wsroute import nbv
        nbv.stage(cfg.case, a.arm, cfg.nbv_prefix)
    elif a.cmd == "grid":
        from wsroute import conflict
        conflict.grid(cfg.case, a.arm, cfg.radius)
    elif a.cmd == "sheet":
        from wsroute import conflict
        conflict.sheet(cfg.case, a.arm, cfg.spots, cfg.radius)
    elif a.cmd == "compare":
        from wsroute import conflict
        conflict.compare(cfg.case, a.arms.split(","), cfg.spots, cfg.radius)
    elif a.cmd == "pseudo":
        from wsroute import fixer
        fixer.pseudo(cfg.case, a.arm, a.tag or cfg.fixer_arms()[a.step - 1], a.step, cfg.fixer_rounds,
                     tuple(cfg.fixer_walked), cfg.fixer_views)
    elif a.cmd == "fix":
        from wsroute import fixer
        fixer.fix(cfg.case, a.tag, cfg.fixer_timestep)
    elif a.cmd == "fixer_views":
        from wsroute import fixer
        fixer.views(cfg.case, a.tag, cfg.fixer_timestep, a.base)
    elif a.cmd == "bundle":
        from wsroute import bundle
        bundle.bundle(cfg.case, a.arm or cfg.final_arm, cfg.radius, cfg.photo)


if __name__ == "__main__":
    main()
