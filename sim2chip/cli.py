"""Command line interface.

    sim2chip run specs/examples/lora_alarm_traffic.json --input mio_tcd=data/real/train
    sim2chip run specs/examples/lora_alarm_traffic.json --from train
    sim2chip status lora_alarm_traffic
    sim2chip stages specs/examples/lora_alarm_traffic.json
    sim2chip interview
    sim2chip describe-scene data/real/maceta --classes "flower pots"
    sim2chip match specs/examples/pots_count.json --job maceta_counter_11
    sim2chip serve --backend nebius
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    if why := spec_mod.no_photos(spec_mod.load(args.spec)):
        raise SystemExit(why)
    pipe = _pipeline(args, create=True)
    for item in args.input:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--input expects NAME=PATH, got '{item}'")
        link_input(pipe.job, name, Path(path))
    kw = {}
    if args.backend == "nebius":
        kw = {"dry_run": args.dry_run, "max_usd": args.max_usd}
    elif args.dry_run:
        raise SystemExit("--dry-run is only for the nebius backend")
    backend = backends.get(args.backend, verbose=args.verbose, **kw)
    pipe = backend.prepare(pipe)
    print(f"job {pipe.job.root}  backend {args.backend}")
    try:
        backend.run(pipe, start=args.start, only=args.only, force=args.force)
    except StageFailed as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    report = pipe.job.stage_dir("export") / "report.json"
    fresh = any(p.stage.name == "export" and p.action == "skip" for p in pipe.plan())
    if fresh and report.exists():
        r = json.loads(report.read_text())
        print(f"model: {report.parent / r['model_path']}  ({r['model_kb']} KB)")
        if r.get("real_test") or r.get("real_accuracy"):  # real_accuracy: reports before 2026-09-27
            a = r.get("real_test") or r["real_accuracy"]
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


def cmd_interview(args) -> int:
    from . import interview
    from .llm import Client

    home = Path(args.home)
    client = Client(usage_log=home / "logs" / "llm_usage.jsonl")
    model = client.model_for("interview")
    print(f"EyeForge interview with {model} (Nebius Token Factory). Ctrl+C to quit.\n")
    objaverse_dir = home / "cache" / "objaverse"
    cats = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    import uuid
    ranker = interview.photo_ranker(client, client.model_for("scene"))
    spec = interview.run(client, model, cats, ask=input, tell=print, scene_check=interview.check_scene,
                         photo_ranker=ranker, uploads=home / "uploads" / f"cli-{uuid.uuid4().hex[:8]}",
                         scene_profiler=interview.scene_profiler(client, client.model_for("scene")))
    if spec is None:
        return 1
    out = Path(args.out or f"specs/{spec.name}.json")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists, use --out or --overwrite")
    out.write_text(spec.to_json() + "\n")
    print(f"\nsaved {out}. Next: sim2chip run {out}")
    return 0


def cmd_describe_scene(args) -> int:
    from . import scene_profile
    from .llm import Client

    client = Client(usage_log=Path(args.home) / "logs" / "llm_usage.jsonl")
    prof, text = scene_profile.profile_folder(Path(args.folder), args.classes, client, client.model_for("scene"))
    print(text)
    print(json.dumps(prof, indent=2))
    return 0


def cmd_match(args) -> int:
    from . import match
    from .llm import Client

    pipe = _pipeline(args, create=False)
    if not pipe.job.root.exists():
        raise SystemExit(f"no job at {pipe.job.root}: run the pipeline up to the render first")
    # The vision model reasons before answering: a comparison takes minutes, not seconds.
    client = Client(usage_log=pipe.job.logs / "llm_usage.jsonl", timeout=match.LLM_TIMEOUT)
    backend = backends.get("local", verbose=args.verbose)
    match.run(pipe.spec, pipe.job, pipe.cache, backend, client, rounds=args.rounds, frames=args.frames,
              locked=tuple(args.lock))
    print(f"proposed spec: {pipe.job.root / 'match' / 'spec.json'}")
    return 0


def cmd_serve(args) -> int:
    from .config import env
    from .server import serve
    from .auth import Auth
    auth = Auth.from_env() if args.public else None
    key = env("SIM2CHIP_WEB_KEY") if args.public and auth is None else None
    if args.public and auth is None and not key:
        raise SystemExit("--public needs a login (SIM2CHIP_PUBLIC_URL, SIM2CHIP_SESSION_SECRET and Google or GitHub "
                         "OAuth clients, see sim2chip/auth.py) or an access key (SIM2CHIP_WEB_KEY)")
    if not args.public and args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit(f"serving on {args.host} lets others in: add --public (access key, uploads only)")
    url = f"http://{args.host}:{args.port}" + ("/?key=<SIM2CHIP_WEB_KEY>" if key else "")
    print(f"EyeForge web on {url}  (backend {args.backend}{', public' if args.public else ''}"
          + (f", login with {' and '.join(auth.providers)})" if auth else ")"))
    daily = env("SIM2CHIP_DAILY_USD")
    if args.public and not env("SIM2CHIP_HOME"):
        raise SystemExit("--public needs SIM2CHIP_HOME on a disk that survives restarts (jobs, buckets, spend)")
    until = env("SIM2CHIP_JUDGE_UNTIL")  # end of the judging, ISO date or time (UTC): judges are free until then
    judge_until = datetime.fromisoformat(until).replace(tzinfo=timezone.utc).timestamp() if until else None
    serve(Path(args.home), args.backend, args.host, args.port, args.max_usd, public=args.public, key=key,
          max_runs=args.max_runs, daily_usd=float(daily) if daily else (30.0 if args.public else None), auth=auth,
          judge_until=judge_until, contact=env("SIM2CHIP_CONTACT"))
    return 0


def _accounts(args):
    from .accounts import Accounts
    from .server import ACCOUNTS_DB
    db = Path(args.home) / ACCOUNTS_DB
    if not db.exists():
        raise SystemExit(f"no accounts yet in {db}")
    return Accounts(db)


def cmd_accounts(args) -> int:
    accs = _accounts(args)
    if args.action == "list":
        for a in accs.all():
            flags = " ".join(f for f, on in (("judge:" + str(a.judge), a.judge), ("banned", a.banned)) if on)
            print(f"{a.email or '-':40} {a.provider:7} ${accs.balance(a.id):8.2f}  {flags}")
        return 0
    if not args.email:
        raise SystemExit(f"accounts {args.action} needs an email")
    value = {"judge": "manual", "unjudge": None, "ban": 1, "unban": 0}[args.action]
    n = accs.set_flag(args.email, "judge" if "judge" in args.action else "banned", value)
    print(f"{args.action}: {n} account(s) with {args.email}" + ("" if n else " (it must sign in once first)"))
    return 0 if n else 1


def cmd_credits(args) -> int:
    accs = _accounts(args)
    found = accs.by_email(args.email)
    if not found:
        raise SystemExit(f"no account with {args.email}: it must sign in once first")
    for a in found:
        accs.add(a.id, args.amount, "grant", args.note)
        print(f"{a.email} ({a.provider}): ${accs.balance(a.id):.2f}")
    return 0


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
    r.add_argument("--from", dest="start", help="rerun this stage and the stages that depend on it")
    r.add_argument("--only", help="run only this stage (upstream must be done)")
    r.add_argument("--force", action="store_true", help="rerun every stage")
    r.add_argument("-v", "--verbose", action="store_true", help="show the full output of every stage")
    r.add_argument("--dry-run", action="store_true", help="nebius: print the cloud commands, create nothing")
    r.add_argument("--max-usd", type=float, help="nebius: spend cap for this run (env SIM2CHIP_MAX_USD, default 5)")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("stages", help="list the stages of a spec and whether they are done")
    s.add_argument("spec")
    s.add_argument("--job")
    s.set_defaults(func=cmd_stages)

    st = sub.add_parser("status", help="stage status of an existing job")
    st.add_argument("job")
    st.set_defaults(func=cmd_status)

    iv = sub.add_parser("interview", help="talk to Nemotron to write a spec")
    iv.add_argument("--out", help="where to save the spec (default specs/<name>.json)")
    iv.add_argument("--overwrite", action="store_true")
    iv.set_defaults(func=cmd_interview)

    ds = sub.add_parser("describe-scene", help="propose scene.contexts, wall_tags and railings from photos")
    ds.add_argument("folder")
    ds.add_argument("--classes", required=True, help="what the camera looks for, e.g. \"flower pots\"")
    ds.set_defaults(func=cmd_describe_scene)

    mt = sub.add_parser("match", help="bring the synthetic scenes closer to the user's photos (trial renders)")
    mt.add_argument("spec")
    mt.add_argument("--job", help="job with the assets ready (default: the spec name)")
    mt.add_argument("--rounds", type=int, default=3)
    mt.add_argument("--frames", type=int, default=48, help="frames per trial render")
    mt.add_argument("--lock", action="append", default=[], metavar="LEVER",
                    help="a lever the user gave as a fact, never changed (e.g. camera.height_m)")
    mt.add_argument("-v", "--verbose", action="store_true")
    mt.set_defaults(func=cmd_match)

    sv = sub.add_parser("serve", help="web UI and API: interview, photos, run, progress, download")
    sv.add_argument("--backend", default="local", choices=["local", "nebius"])
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--public", action="store_true",
                    help="for a server others reach: access key (SIM2CHIP_WEB_KEY), uploaded photos only, run limit")
    sv.add_argument("--max-runs", type=int, default=2, help="--public: runs at the same time (GPU quota)")
    sv.add_argument("--max-usd", type=float, help="nebius: spend cap per run")
    sv.set_defaults(func=cmd_serve)

    ac = sub.add_parser("accounts", help="public server accounts: list, mark a judge (free use), pause or unpause")
    ac.add_argument("action", choices=["list", "judge", "unjudge", "ban", "unban"])
    ac.add_argument("email", nargs="?", help="the account's email (every account with it)")
    ac.set_defaults(func=cmd_accounts)

    cr = sub.add_parser("credits", help="add dollars to the balance of an account of the public server")
    cr.add_argument("action", choices=["grant"])
    cr.add_argument("email")
    cr.add_argument("amount", type=float, help="dollars (things are charged by use, about 1 $ a run)")
    cr.add_argument("--note", default=None)
    cr.set_defaults(func=cmd_credits)

    args = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up when piped to tee or a log file
    try:
        return args.func(args)
    except (spec_mod.SpecError, ValueError, KeyError, FileExistsError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
