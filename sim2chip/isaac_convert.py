"""Isaac Sim's Asset Converter for the models glb2usd cannot read (convert_assets' fallback).

    <Isaac Sim python.sh> -m sim2chip.isaac_convert '[["in.glb", "out.usd"], ...]'
"""
from __future__ import annotations

import asyncio
import json
import sys


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
        print("[sim2chip] conversion error", src, task.get_error_message(), flush=True)
    return ok


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        loop = asyncio.get_event_loop()
        pairs = json.loads(sys.argv[1])
        ok = sum(loop.run_until_complete(convert(src, dst)) for src, dst in pairs)
        print(f"[sim2chip] Asset Converter: {ok} of {len(pairs)} models converted", flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
