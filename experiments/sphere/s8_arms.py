"""S.8 of the sphere route (docs/roadmap_photosphere.md): does the student count better with the spheres' frames?
Arms on one count job of the user's photos (bins_job.py): photos only, and photos plus the frames S.6 made on the
spheres (s6_sheet.py), 3 runs of 5 seeds each (PicoDet's noise, roadmap_photo2scene). The frames go in as the
job's scene_from_photo output, so the product's render stage takes them as it takes the user's place: the
similarity filter, then points and training. Frames the VLM review rejected stay out.

    python experiments/sphere/s8_arms.py --job bins_count --places data/sphere_test/PXL_* [--runs 3]

Appends to <home>/arms.log and writes <home>/s8_<job>.json with each run's test score.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from sim2chip import spec as spec_mod
from sim2chip.pipeline import Pipeline
from sim2chip.workspace import Cache, open_job

ROOT = Path(__file__).resolve().parents[2]
HOME = Path("/media/lenena-iker/Crucial/runs/eyeforge/p2s_test")
SCORE = re.compile(r"real test (?:points F1|accuracy \(int8\):) ([0-9.]+)")


def frames_from(places: list[Path], out: Path) -> list[dict]:
    """The spheres' final frames with their labels as scene_from_photo/frames.json entries, review rejects out."""
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in places:
        labels = json.loads((d / "render" / "labels.json").read_text())
        rv = d / "render" / "review.json"
        review = json.loads(rv.read_text())["views"] if rv.exists() else {}
        for name, boxes in sorted(labels.items()):
            if not review.get(name, {"keep": True})["keep"]:
                continue
            src = d / "render" / f"final_{name}.png"
            dst = out / d.name / f"{name}.png"
            dst.parent.mkdir(exist_ok=True)
            shutil.copyfile(src, dst)
            frames.append({"file": str(dst.relative_to(out)), "scene": d.name,
                           "boxes": [[b["label"], *[round(c) for c in b["box"]]] for b in boxes]})
    return frames


def inject(spec_path: Path, job_name: str, places: list[Path]) -> int:
    """The frames as this job's finished scene_from_photo stage (its fingerprint, so the pipeline keeps it)."""
    spec = spec_mod.load(spec_path)
    job = open_job(HOME, job_name)
    pipe = Pipeline(spec, job, Cache(HOME / "cache"))
    out = job.stage_dir("scene_from_photo")
    shutil.rmtree(out, ignore_errors=True)
    frames = frames_from(places, out)
    (out / "frames.json").write_text(json.dumps(frames, indent=1))
    (out / "scenes.json").write_text("[]")
    job.mark_done("scene_from_photo", pipe.fingerprints()["scene_from_photo"],
                  {"frames": len(frames), "objects": sum(len(f["boxes"]) for f in frames), "built": 0})
    return len(frames)


def run(spec_path: Path, job: str, start: str, log: Path) -> float | None:
    with open(log, "w") as f:
        subprocess.run([sys.executable, "-m", "sim2chip", "--home", str(HOME), "run", str(spec_path), "--job", job,
                        "--from", start], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    m = SCORE.findall(log.read_text())
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="bins_count")
    ap.add_argument("--places", nargs="+", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=["photos", "sphere"])
    args = ap.parse_args()
    base = json.loads((HOME / "jobs" / args.job / "spec.json").read_text())
    results = {}
    for arm in args.arms:
        spec = {**base, "generation": {**base["generation"], "photo_scenes": arm == "sphere"}}
        path = HOME / "jobs" / args.job / f"spec_{arm}.json"
        path.write_text(json.dumps(spec, indent=1))
        if arm == "sphere":
            n = inject(path, args.job, [p.resolve() for p in args.places])
            print(f"{n} sphere frames in", flush=True)
        for r in range(1, args.runs + 1):
            log = HOME / f"arm_{args.job}_s8_{arm}_{r}.log"
            # ingest's split is the same for both arms (it reads no generation settings); the frames come in at
            # render, which reruns the similarity filter
            start = "render" if arm == "sphere" else "ingest" if r == 1 else "train"
            score = run(path, args.job, start, log)
            results.setdefault(arm, []).append(score)
            line = f"S.8 {args.job} {arm} run {r}: {score}"
            print(line, flush=True)
            with open(HOME / "arms.log", "a") as f:
                f.write(line + "\n")
    (HOME / f"s8_{args.job}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
