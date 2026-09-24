"""Command line interface.

    sim2chip run specs/lora_alarm_traffic.json --input mio_tcd=data/real/train
    sim2chip run specs/lora_alarm_traffic.json --from train
    sim2chip status lora_alarm_traffic
    sim2chip stages specs/lora_alarm_traffic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import backends, spec as spec_mod
from .pipeline import Pipeline, StageFailed
from .workspace import Cache, Job, default_home, link_input, open_job


def _pipeline(args, create: bool) -> Pipeline:
    spec = spec_mod.load(args.spec)
    home = Path(args.home)
    name = args.job or spec.name
    job = open_job(home, name) if create else Job(home / "jobs" / name)
    return Pipeline(spec, job, Cache(home / "cache"))


def cmd_run(args) -> int:
    pipe = _pipeline(args, create=True)
    for item in args.input:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--input expects NAME=PATH, got '{item}'")
        link_input(pipe.job, name, Path(path))
    print(f"job {pipe.job.root}  backend {args.backend}")
    try:
        pipe.run(backends.get(args.backend, verbose=args.verbose), start=args.start, only=args.only,
                 force=args.force)
    except StageFailed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    report = pipe.job.stage_dir("export") / "report.json"
    if report.exists():
        r = json.loads(report.read_text())
        print(f"model: {r['model_path']}  ({r['model_kb']} KB)")
        if r.get("real_accuracy"):
            a = r["real_accuracy"]
            print(f"real test accuracy (int8): {a['mean']:.3f} +/- {a['std']:.3f} over seeds {a['seeds']}")
    return 0


def cmd_stages(args) -> int:
    pipe = _pipeline(args, create=False)
    for p in pipe.plan():
        state = "done" if p.action == "skip" else "pending"
        print(f"{p.stage.name:15s} {p.stage.runtime:6s} {p.stage.gpu or '':4s} {state}")
    return 0


def cmd_status(args) -> int:
    job = Job(Path(args.home) / "jobs" / args.job)
    if not job.spec_path.exists():
        raise SystemExit(f"no job at {job.root}")
    args.spec, args.job = job.spec_path, job.name
    return cmd_stages(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim2chip")
    p.add_argument("--home", default=str(default_home()), help="folder with jobs/ and cache/ (env SIM2CHIP_HOME)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline for a spec")
    r.add_argument("spec")
    r.add_argument("--job", help="job name (default: the spec name)")
    r.add_argument("--input", action="append", default=[], metavar="NAME=PATH",
                   help="expose a user folder as input/NAME inside the job")
    r.add_argument("--backend", default="local", choices=["local", "nebius"])
    r.add_argument("--from", dest="start", help="rerun from this stage on")
    r.add_argument("--only", help="run only this stage (upstream must be done)")
    r.add_argument("--force", action="store_true", help="rerun every stage")
    r.add_argument("-v", "--verbose", action="store_true", help="show the full output of every stage")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("stages", help="list the stages of a spec and whether they are done")
    s.add_argument("spec")
    s.add_argument("--job")
    s.set_defaults(func=cmd_stages)

    st = sub.add_parser("status", help="stage status of an existing job")
    st.add_argument("job")
    st.set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (spec_mod.SpecError, ValueError, KeyError, FileExistsError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
