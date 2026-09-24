"""Sky maps (HDRI) from NVIDIA's public asset bucket, tagged by weather so the spec can choose."""
from __future__ import annotations

from dataclasses import dataclass

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies"


@dataclass(frozen=True)
class Sky:
    file: str
    weather: str  # clear | cloudy | evening | storm

    @property
    def url(self) -> str:
        return f"{BUCKET}/{self.weather.capitalize()}/{self.file}"


SKIES = [Sky(f, "clear") for f in ("lakeside.hdr", "mealie_road.hdr", "noon_grass.hdr", "qwantani.hdr",
                                   "signal_hill_sunrise.hdr", "syferfontein_18d_clear.hdr")] + \
        [Sky(f, "cloudy") for f in ("between_bridges.hdr", "champagne_castle_1.hdr",
                                    "kloofendal_48d_partly_cloudy.hdr", "sunflowers.hdr", "table_mountain_1.hdr",
                                    "white_cliff_top.hdr")] + \
        [Sky("evening_road_01.hdr", "evening"), Sky("approaching_storm.hdr", "storm")]
