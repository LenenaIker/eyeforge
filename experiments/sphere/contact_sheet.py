"""Ground-contact check (user, 2026-10-17): each place's proxy from above in three versions side by side:
    MoGe as it was (proxy_before_walls.npz), the ground-contact distances (paved ground on the plane, walls at
    their base's distance), and the contact plus the plane straightening;
with how many pixels the contact set and the walls the straightening found.

    python experiments/sphere/contact_sheet.py data/sphere_test/PXL_*
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from walls_sheet import above  # noqa: E402

from sim2chip.photo2scene.sphere import geometry  # noqa: E402

TOP = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("places", nargs="+", type=Path)
    args = ap.parse_args()
    panels = []
    for d in args.places:
        old = json.loads((d / "geometry.json").read_text())
        kw = dict(capture_height_m=old.get("height_given_m"), proxy_depth=old.get("proxy_from"),
                  proxy_stuff=old.get("proxy_ground_from"))
        contact = geometry.solve(d, **kw, straighten_walls=False)
        shutil.copy(d / "proxy.npz", d / "proxy_contact.npz")
        both = geometry.solve(d, **kw)  # the default: contact and walls
        h = both["camera_height_m"]
        views = [("MoGe as it was", np.load(d / "proxy_before_walls.npz"), ()),
                 ("ground contact", np.load(d / "proxy_contact.npz"), ()),
                 ("ground contact + plane straightening", np.load(d / "proxy.npz"), both["walls"])]
        c = contact["contact"]
        text = (f"{d.name}: paved ground on the plane {c.get('paved_px', 0)} px; walls at their base's distance "
                f"{c.get('structure_px', 0)} px in {c.get('columns', 0)} columns; then {len(both['walls'])} walls "
                f"straightened")
        print(text, flush=True)
        panel = Image.new("RGB", (3 * TOP + 40, TOP + 70), "white")
        draw = ImageDraw.Draw(panel)
        for k, (name, proxy, walls) in enumerate(views):
            panel.paste(above(proxy["vertices"][~proxy["ground"]], h, walls), (k * (TOP + 20), 34))
            draw.text((k * (TOP + 20) + 8, 6), name, fill=(0, 0, 0), font_size=24)
        draw.text((8, TOP + 40), text, fill=(0, 0, 0), font_size=22)
        panels.append(panel)
    out = Image.new("RGB", (panels[0].width, sum(p.height + 20 for p in panels)), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + 20
    path = args.places[0].parent / "sheet_contact.jpg"
    out.save(path, quality=85)
    print(path)


if __name__ == "__main__":
    main()
