"""plan.json of the student camera's shots in the orbit clip (run/anim.json): what each flash captures, at the
spec's 4:3 camera (60 degrees), with the bins alive at that moment. Rendered by walkscene like the orbit."""
import json
import sys
from pathlib import Path

ROOT = Path("/home/lenena-iker/work/eyeforge")
sys.path.insert(0, str(ROOT))
from sim2chip.walkscene import w2c  # noqa: E402

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((RUN / "plan.json").read_text())
anim = json.loads((RUN / "anim.json").read_text())
views = []
for k, s in enumerate(anim["shots"]):
    frame = plan["views"][min(int(round(s["t"] * anim["fps"])), len(plan["views"]) - 1)]
    views.append({"name": f"{k:04d}", "scene": frame["scene"], "class": "bin", "objects": frame["objects"],
                  "w2c": w2c(s["pos"], s["yaw"], s["pitch"]).round(6).tolist()})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plan.json").write_text(json.dumps({"size": [640, 480], "fov_deg": 60.0, "views": views, "missed": 0}, indent=1))
(OUT / "scenes.json").write_text((RUN / "scenes.json").read_text())
print(len(views), "shots")
