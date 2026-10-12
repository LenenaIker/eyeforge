"""photo2scene's street and room routes in the product (sim2chip/photo2scene/place.py, roadmap C.1): no step is
run, the tests write what the steps would."""
import json
from pathlib import Path

import pytest
from PIL import Image

from sim2chip.photo2scene import STEPS, Stop, place


class FakeFlow:
    """Runs no step; records which ones were asked for."""
    def __init__(self, d, codes=None):
        self.d, self.codes, self.ran, self.said = d, codes or {}, [], []

    def run(self, step, python, script, args, outputs=(), ok=(0,)):
        self.ran.append((step, python, script))
        return self.codes.get(step, 0)

    def json(self, name):
        return json.loads((self.d / name).read_text())

    def log(self, m):
        self.said.append(m)


def write(d: Path, name: str, value):
    (d / name).parent.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(value))


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "in.jpg"
    Image.new("RGB", (64, 48)).save(p)
    return p


def routed(d, route, template="crossroads", built=True, why=()):
    write(d, "observe.json", {"final": {"place": "village_street", "objects": ["car"], "ground": ["road"],
                                        "surroundings": ["house"], "weather": "cloudy"}})
    write(d, "route.json", {"route": route, "template": template, "template_built": built,
                            "hdri_group": "countryside", "why": list(why)})


def test_a_photo_out_of_scope_makes_no_place(tmp_path, photo):
    d = tmp_path / "s"
    d.mkdir()
    routed(d, "out_of_scope", "none", False, ["night"])
    with pytest.raises(Stop, match="out of scope: night"):
        place.build(FakeFlow(d, {"1.2_route": 3}), photo, [])


@pytest.mark.parametrize("route,template,built,why", [
    ("support", "none", False, "working surface"),
    ("village_street_day", "square", False, "no template builds a square"),
])
def test_routes_this_module_does_not_build_stop_before_any_model(tmp_path, photo, route, template, built, why):
    d = tmp_path / "s"
    d.mkdir()
    routed(d, route, template, built)
    flow = FakeFlow(d)
    with pytest.raises(Stop, match=why):
        place.build(flow, photo, [])
    assert [s for s, *_ in flow.ran] == ["1.1_observe", "1.2_route"]
    assert (d / "photo.png").exists()


def test_compose_keeps_the_views_the_check_passed_without_the_photo_camera(tmp_path):
    d = tmp_path
    routed(d, "village_street_day")
    write(d, "extended/views.json", {"kept": [1, 3]})
    write(d, "populated/populate.json", {"placed_per_view": [2, 3, 0, 1]})
    flow = FakeFlow(d)
    # compose() clears the job's last things first: the step outputs appear after that, as the steps would write.
    real_run = flow.run

    def run(step, *a, **kw):
        if step == "3.5_populate":
            write(d, "populated/populate.json", {"placed_per_view": [2, 3, 0, 1]})
        if step == "3.5_labels":
            write(d, "populated/isaac/labels.json", {f"render_{k:02d}.png": [{"box": [0, 0, 1, 1]}] for k in range(4)})
            for k in range(4):
                (d / f"populated/isaac/render_{k:02d}.png").write_bytes(b"")
        return real_run(step, *a, **kw)
    flow.run = run
    frames = place.compose(flow, 7)
    assert [f["file"] for f in frames] == ["populated/isaac/render_01.png", "populated/isaac/render_03.png"]
    assert [s for s, *_ in flow.ran] == ["3.5_populate", "3.5_render", "3.5_labels"]


def test_the_steps_need_nothing_from_the_experiments():
    """The product runs its steps from its own package (roadmap C.1)."""
    for f in STEPS.glob("*.py"):
        code = [line for line in f.read_text().splitlines() if line.lstrip().startswith(("import", "from", "sys.path"))]
        assert not [line for line in code if "experiments" in line], f.name
