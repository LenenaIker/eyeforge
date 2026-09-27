"""Isaac Sim's Asset Converter for the models glb2usd cannot read (convert_assets' fallback).

    <Isaac Sim python.sh> -m sim2chip.isaac_convert pairs.json   # [["in.glb", "out.usd"], ...]
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from .workspace import say


async def convert(src: str, dst: str) -> bool:
    import omni.kit.asset_converter as converter

    c = converter.AssetConverterContext()
    c.ignore_materials = False
    c.ignore_animations = True
    c.ignore_camera = True
    c.ignore_light = True
    c.embed_textures = True
    task = converter.get_instance().create_converter_task(src, dst, None, c)
    ok = await task.wait_until_finished()
    if not ok:
        say(f"conversion error {src} {task.get_error_message()}")
    return ok


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        loop = asyncio.get_event_loop()
        pairs = json.loads(Path(sys.argv[1]).read_text())
        ok = sum(loop.run_until_complete(convert(src, dst)) for src, dst in pairs)
        say(f"Asset Converter: {ok} of {len(pairs)} models converted")
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
