"""Blob vs peak decoding of count grids on FSC-147 geese: the target grids themselves (ceiling) and a job's
trained int8 models, scored against the dataset's points.  python experiments/count_decode.py JOB"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import numpy as np
from count_decode_lib import peaks
from fsc147_count import job_dir, truth_of
from sim2chip import counting
from sim2chip import student as tu

job = job_dir(sys.argv[1]); labels = ["empty", "goose"]; size = 224; grid = counting.grid_size(size)
xt, mt, names, _ = tu.load_points(job / "ingest" / "test", labels, size)
info = json.loads((job / "ingest" / "test" / "points.json").read_text()); truth = truth_of()
tt = [truth[info[n]["key"]] for n in names]
# Ceiling: the truth itself on the grid.
tgt = []
for t in tt:
    g = np.zeros((grid, grid, 2), np.float32); g[..., 0] = 1
    for x, y, *_ in t:
        i, j = min(int(y * grid), grid - 1), min(int(x * grid), grid - 1); g[i, j] = (0, 1)
    tgt.append(g)
for name, dec in (("blob", counting.decode), ("peak", peaks)):
    print("ceiling", name, counting.score([dec(g, labels, .5) for g in tgt], tt, grid))
for d in sorted((job / "train").glob("seed_*")):
    grids = [np.asarray(g) for g in tu.predict_int8_grid((d / "model_int8.tflite").read_bytes(), xt)]
    for name, dec in (("blob", counting.decode), ("peak", peaks)):
        c = {t: counting.score([dec(g, labels, t) for g in grids], tt, grid) for t in counting.THRESHOLDS}
        b = max(c, key=lambda t: c[t]["f1"])
        print(d.name, name, " ".join(f"{t}:{c[t]['f1']:.3f}" for t in c), "| best", b, c[b])
