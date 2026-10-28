"""Entry point that runs one stage inside its runtime (Isaac Sim Python, TensorFlow env or plain Python).

    python -m sim2chip.run_stage <stage> --job <job dir> --cache <cache dir> --fingerprint <fp>

On success it writes job/<stage>/_done.json with the fingerprint. The caller checks that file instead
of the exit code, because Isaac Sim's app.close() ends the process with its own exit code.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


GPU_STAGES = ("label", "scene_render", "generate_assets")  # their tools need the GPU whatever TensorFlow does


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fingerprint", required=True)
    args, _ = p.parse_known_args()

    from . import spec as spec_mod
    from .stages import BY_NAME, Context
    from .workspace import Cache, Job

    stage = BY_NAME[args.stage]
    # A GPU TensorFlow cannot use (Blackwell, backends.nebius.TF_ON_CPU): TensorFlow's stages run on the CPUs. Set
    # before anything imports TensorFlow; the teachers and the renderers keep the GPU.
    if os.environ.get("SIM2CHIP_TF_CPU") and stage.runtime in ("tf", "cpu") and stage.name not in GPU_STAGES:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    job = Job(args.job)
    ctx = Context(spec_mod.load(job.spec_path), job, Cache(args.cache))
    app = None
    if stage.runtime == "isaac":
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
    ok = False
    try:
        info = stage.run(ctx)
        job.mark_done(stage.name, args.fingerprint, info)
        ok = True
    except Exception:
        # Print before app.close(): Isaac Sim exits the process and would swallow the traceback.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        if app is not None:
            app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
